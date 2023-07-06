import argparse
import os
import socket
import sys

import psutil
import setproctitle
import torch.nn
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(),"")))
from data_preprocessing.recommandation.data_loader import *
from model.ngcf import NGCF
from training.ngcf_trainer import NGCFTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed

def adding_args(parser):
    #training
    parser.add_argument('--model', type=str, default='ngcf', metavar='N',
                        help='neural network used in training' )

    parser.add_argument('--dataset', type=str, default='ml-100k', metavar='N',
                        help='amazon-book,ml-100k,gowalla')

    parser.add_argument('--data_path', type=str, default='./../../../data/ml-100k',
                        help='data directory')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='client number in total')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--backend', type=str, default='MPI', metavar='N',
                        help='idontknow')

    parser.add_argument('--partition_method', type=str, default='hetro', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    #model related
    parser.add_argument('--epochs', type=int, default=20, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=20,
                        help='number of client epoch')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size')

    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu number per server default 4')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    args = parser.parse_args()
    return args



def load_data(args, dataset_name, generator):
    if (args.dataset != 'amazon-book') and (args.dataset != 'gowalla') and (args.dataset != 'ml-100k') :
        raise Exception("no such dataset!")

    compact = (args.model == 'ngcf')
    logging.info("load_data. dataset_name = %s" % dataset_name)
    unif = True if args.partition_method == "homo" else False

    #데이터 불러오기
    user_num, item_num = data_generator.n_users, data_generator.n_items
    norm_adj = generator.get_adj_mat()

    train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global, \
    data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict = generator.load_partition_data(
        args.client_num_in_total,
        uniform=unif)

    dataset = [train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global,
               data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict]

    return dataset, user_num, item_num, norm_adj


def create_model(args, model_name, user_num, item_num, adj_mat):
    logging.info("create_model. model_name = %s" % (model_name))
    if model_name == 'ngcf':
        model = NGCF(user_num, item_num, adj_mat, args) #init 여기서
        trainer = NGCFTrainer(model, args)
    else:
        raise Exception("such model does not exist !")
    logging.info("model created")
    return model, trainer

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):

    if process_ID == 0:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        return device

    process_gpu_dict = dict()

    for worker_index in range(fl_worker_num):
        gpu_index = worker_index % gpu_num_per_machine #gpu가 1이면 나머지는 다 0
        process_gpu_dict[worker_index] = gpu_index

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") #str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")

    return device


def post_complete_message_to_sweep_process(args):
    logging.info("post_complete_message_to_sweep_process")
    pipe_path = "./fed_rec"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(args)))


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()
    # parse python script input parameters
    parser = argparse.ArgumentParser(description="Run fed-ngcf")
    args = adding_args(parser)

    # customize the process name
    str_process_name = "fed:" + "hetero ngcf 10" #str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    #seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    args.device = device
    data_generator = Data_generator(args.data_path, args.batch_size)
    args.data_generator = data_generator

    #load_data
    dataset, n_user, n_item, norm_adj = load_data(args, args.dataset, data_generator)

    [train_data_num, val_data_num, test_data_num, train_data_global, val_data_global, test_data_global,
     data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict] = dataset

    #create_model
    model, trainer = create_model(args, args.model, n_user, n_item, norm_adj)

    data_generator.print_statistics()

    # trainer.train(train_data_local_dict[4], device, args) #트레이너 잘 돌아감
    # trainer.test(test_data_local_dict[0], device, args) #test 돌아감 값이 들어감
    # trainer.test_on_the_server(train_data_local_dict,test_data_local_dict, device, args)
    # exit(0)

    FedML_FedAvg_distributed(process_id, worker_number, device, comm,
                             model, train_data_num, train_data_global, test_data_global,
                             data_local_num_dict, train_data_local_dict, test_data_local_dict, args,
                             trainer)

    if process_id == 0:
        post_complete_message_to_sweep_process(args)

#끝