import argparse
import os
import socket
import sys

import psutil
import setproctitle
import torch.nn
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.cwd(),"./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.cwd(),"")))
from data_preprocessing.molecule.data_loader import * #TODO_rec_ver
from model.ngcf import ngcfnet #TODO
from training.ngcf_trainer import ngcftrainer #TODO
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed

def add_args(parser):
    #Training set
    parser.add_argument('--model', type='str', default='ngcf', metavar='N',
                        help='neural network used in training' )

    parser.add_argument('--dataset', type=str, default='gowalla', metavar='N',
                        help='amazon-book,yelp2018,gowalla')

    parser.add_argument('--data_dir', type=str, default='./../../../data/gowalla',
                        help='data directory')

    parser.add_argument('--')



def load_data(args, dataset_name):
    if (args.dataset != 'amazon-book') and (args.dataset != 'gowalla') and (args.dataset != 'yelp2018') :
        raise Exception("no such dataset!")

    compact = (args.model == 'gowalla')
    logging.info("load_data. dataset_name = %s" % dataset_name)
    #자이제시작이야내꿈을
    _, feature_matrices, labels = get_data(args.data_dir)

def creat_model(args, model_name, feat_dim, label_num, output_dim):

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):

if __name__ == "__main__":

