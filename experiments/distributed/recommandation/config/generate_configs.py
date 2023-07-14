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
args = parser.parse_args()
# {test_type}_{data}_{token}.yamln

if not os.path.exists(args.path):
    os.makedirs(args.path)

with open("base.yaml") as fd:
    base_config = yaml.load(fd, Loader=yaml.FullLoader)

#-------------------------------------correct below--------------------------------------------
#config variation set
#should set same key of config.yaml
var_keys = ["test_type",
            "dataset",
            "send_telegram_after_complete",
            "seed",
            "use_SL"
            ]
case_test_type = ["SLOn"]
case_dataset = ["ml-100k"] #, "gowalla", "yelp2018"
case_telegram = [True]
case_seed = rd.sample(range(10000), 5) #also act as repeatition of test
case_SL = [True]

#should match order of config key
var_set = [case_test_type, case_dataset, case_telegram, case_seed, case_SL]
#----------------------------------------------------------------------------------------------
assert len(var_keys) == len(var_set)

for var_values in itertools.product(*var_set):
    token = secrets.token_hex(TOKEN_LENGTH//2)
    base_config["config_token"] = token
    for key, val in zip(var_keys, var_values):
        base_config[key] = val
        # if key == "dataset":
        #     base_config["data_path"] = f"./../../../data/{val}"
    with open(os.path.join(args.path, f"{base_config['test_type']}_{base_config['dataset']}_{token}.yaml"), "w") as fd:
        yaml.dump(base_config, fd)