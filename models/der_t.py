# Please note that the current implementation of DER only contains the dynamic expansion process, since masking and pruning are not implemented by the source repo.
import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 0.1
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4


class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        args['depth_resnet'] = 10
        args['width_resnet'] = 64
        args['final_size'] = 2
        self._network = DERNet(args, False)
        #t
        self.teach_args = copy.deepcopy(args)
        if args['dataset'] == 'cifar100':
            self.t_dual = 4
        else:
            self.t_dual = 3
        self.teach_args['depth_resnet'] = 18
        self.teach_args['width_resnet'] = 42
        self.teach_args['final_size'] = 1
        self._teach_network = IncrementalNet(self.teach_args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        #t
        self._teach_network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("Main model's params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.train_loader_t=self.train_loader

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        #t
        self._teach_network.to(self._device)
        
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=init_epoch, eta_min=1e-5)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=1e-5)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        #t
        teach_optimizer = optim.SGD(
            self._teach_network.parameters(),
            momentum=0.9,
            lr=init_lr,
            weight_decay=init_weight_decay,
        )
        teach_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=teach_optimizer, T_max=init_epoch, eta_min=1e-5)

        for epoch in range(init_epoch):
            self._teach_network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(self.train_loader_t):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._teach_network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                teach_optimizer.zero_grad()
                loss.backward()
                teach_optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            teach_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(self._teach_network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(self.train_loader_t),
                    train_acc,
                    test_acc,
                )
                logging.info(info)

        for epoch in range(init_epoch):
            self.train()
            self._teach_network.eval() #t
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                #t
                loss_kdt = _KD_loss(
                    logits,
                    self._teach_network(inputs)["logits"],
                    self.t_dual,
                )
                loss = (loss_kdt + loss_clf)*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        #t
        teach_optimizer = optim.SGD(
            self._teach_network.parameters(),
            momentum=0.9,
            lr=lrate,
            weight_decay=weight_decay,
        )
        teach_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=teach_optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            self._teach_network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(self.train_loader_t):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._teach_network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                teach_optimizer.zero_grad()
                loss.backward()
                teach_optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            teach_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(self._teach_network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(self.train_loader_t),
                    train_acc,
                    test_acc,
                )
                logging.info(info)

        for epoch in range(epochs):
            self.train()
            self._teach_network.eval() #t
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes + 1 > 0,
                    aux_targets - self._known_classes + 1,
                    0,
                )
                loss_aux = F.cross_entropy(aux_logits, aux_targets)
                #t
                loss_kdt_aux = _KD_loss(
                    aux_logits[:, 1: ],
                    self._teach_network(inputs)["logits"][:, self._known_classes: ],
                    self.t_dual,
                )
                loss_kdt = _KD_loss(
                    logits,
                    self._teach_network(inputs)["logits"],
                    self.t_dual,
                )
                loss = (loss_kdt + loss_clf)*0.5 + (loss_aux + loss_kdt_aux)*0.5

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 4:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_aux / len(train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(info)

    def confusion_matrix(self, task_num, file_id):
        total_class = 100
        model = self._network
        loader = self.test_loader
        model.eval()
        matrix = np.zeros((total_class, total_class))
        matrix_task = np.zeros((task_num, task_num))
        for i, (k, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probs, 1)
                predicted = predicted.to('cpu')
            for p,t in zip(predicted, targets):
                matrix[t, p] += 1
                matrix_task[t*task_num//total_class, p*task_num//total_class] +=1

        import pandas as pd
        df = pd.DataFrame(matrix)
        df.to_csv('./csv/cm_' + file_id +'.csv', index=False, header=False)
        df_task = pd.DataFrame(matrix_task)
        df_task.to_csv('./csv/cm_task_' + file_id + '.csv', index=False, header=False)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
