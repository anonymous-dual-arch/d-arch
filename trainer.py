import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np


def train(args, file_id):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    aia=0.0

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        aia = _train(args, file_id)
    return aia


def _train(args, file_id):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    
    if not os.path.exists('./log'):
       os.makedirs('./log')
    if not os.path.exists('./csv'):
       os.makedirs('./csv')

    logfilename = './log/%s.txt'%(file_id)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    for task in range(data_manager.nb_tasks):
                
        logging.info("Main model's params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        if hasattr(model, '_teach_network'):
            if model._teach_network is not None:
                logging.info("Teacher model's params: {}".format(count_parameters(model._teach_network)))
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        class_count = args["init_cls"] + task*args["increment"]
        if args['dataset'] == 'imagenet100' and class_count >= 100:
            break

    logging.info("Main model's params: {}".format(count_parameters(model._network)))
    logging.info(
        "Trainable params: {}".format(count_parameters(model._network, True))
    )
    if hasattr(model, '_teach_network'):
        if model._teach_network is not None:
            logging.info("Teacher model's params: {}".format(count_parameters(model._teach_network)))

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        print('Accuracy Matrix (CNN):')
        print(np_acctable)

        new_curve = []
        af_curve = []
        for i in range(task + 1):
            new_curve.append(np_acctable[i, i])
            if i!= 0:
                af_curve.append(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, i])[:i]))
        logging.info("CNN NewAcc curve: {}".format(new_curve))
        logging.info("CNN Forgetting curve: {}".format(af_curve))

        forgetting = af_curve[-1]
        aan = np.mean(new_curve)
        logging.info('AAN (CNN): {}'.format(aan))
        logging.info('Forgetting (CNN): {}'.format(forgetting))

    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        print('Accuracy Matrix (NME):')
        print(np_acctable)

        new_curve = []
        af_curve = []
        for i in range(task + 1):
            new_curve.append(np_acctable[i, i])
            if i!= 0:
                af_curve.append(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, i])[:i]))
        logging.info("NME NewAcc curve: {}".format(new_curve))
        logging.info("NME Forgetting curve: {}".format(af_curve))

        forgetting = af_curve[-1]
        aan = np.mean(new_curve)
        logging.info('AAN (NME): {}'.format(aan))
        logging.info('Forgetting (NME): {}'.format(forgetting))

    task_num = task + 1
    model.confusion_matrix(task_num, file_id)


    return float(sum(cnn_curve["top1"])/len(cnn_curve["top1"]))

    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
