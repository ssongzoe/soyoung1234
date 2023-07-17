import yaml
import secrets
import itertools
import argparse
import os
import random as rd
import numpy as np

TOKEN_LENGTH = 6

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--numfile", type=int, default=10)
args = parser.parse_args()
# {test_type}_{data}_{token}.yamln

with open("base.yaml") as fd:
    base_config = yaml.load(fd, Loader=yaml.FullLoader)

#-------------------------------------correct below--------------------------------------------
#config variation set
#should set same key of config.yaml
var_keys = ["test_type",
            "dataset",
            "send_telegram_after_complete",
            "seed",
            "bpr_reg",
            "ssl_reg",
            "proto_reg"
            ]
case_test_type = ["LossContribution"]
case_dataset = ["ml-100k"] #, "gowalla", "yelp2018"
case_telegram = [True]
case_seed = rd.sample(range(10000), 3) #also act as repeatition of test
case_bpr = [0.1, 0.01, 0.001, 0.0001]
case_ssl = [1e-5, 1e-6, 1e-7, 1e-8]
case_proto = [8e-5, 8e-6, 8e-7, 8e-8]

#should match order of config key
var_set = [case_test_type, case_dataset, case_telegram, case_seed, case_bpr, case_ssl, case_proto]
#----------------------------------------------------------------------------------------------
assert len(var_keys) == len(var_set)

file_index = 1
num_file_in_dir = args.numfile

for var_values in itertools.product(*var_set):
    token = secrets.token_hex(TOKEN_LENGTH//2)
    base_config["config_token"] = token
    
    for key, val in zip(var_keys, var_values):
        base_config[key] = val
        # if key == "dataset":
        #     base_config["data_path"] = f"./../../../data/{val}"

    # if var_values[-1] != var_values[-2]:
    #     continue
    if not os.path.exists(f"{args.path}{file_index}"):
        os.makedirs(f"{args.path}{file_index}")
    
    with open(os.path.join(f"{args.path}{file_index}", f"{base_config['test_type']}_{base_config['dataset']}_{token}.yaml"), "w") as fd:
        yaml.dump(base_config, fd)

    if len(os.listdir(f"{args.path}{file_index}")) > num_file_in_dir:
        file_index += 1