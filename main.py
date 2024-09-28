import json
import argparse
import os
import sys
import multiprocessing
from datetime import datetime
from trainer import train

default_config = './exps/der_t.json'

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Json file of settings.')
    parser.add_argument('--gpu_', type=int, default=None,
                        help='ID of used gpu, the default value is defined in the config file.')
    parser.add_argument('--dataset_', type=str, default=None,
                        help='Name of dataset, the default value is defined in the config file.')
    parser.add_argument('--task_num_', type=int, default=None,
                        help='Number of tasks, the default value is defined in the config file.')

    return parser

class TrainModel(object):
    def __init__(self):
        self.file_id = None

    def process(self):
        args = setup_parser().parse_args()
        param = load_json(args.config)
        args = vars(args)  # Converting argparse Namespace to a dict.
        args.update(param)  # Add parameters from json
        
        if args['gpu_']:
            args['device'] = [str(args['gpu_'])]
        if args['dataset_']:
            args['dataset'] = args['dataset_']
        if args['task_num_']:
            args['init_cls'] = 100//args['task_num_']
            args["increment"] = 100//args['task_num_']
        

        self.file_id = args["model_name"].replace("_t", "-dual_arch") + "-" + args["dataset"] + "-inc" + str(args["increment"])

        aia = 0.0
        aia = train(args, self.file_id)
        return round(aia,3)


    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()


if __name__ == '__main__':
    m=TrainModel()
    m.process()
