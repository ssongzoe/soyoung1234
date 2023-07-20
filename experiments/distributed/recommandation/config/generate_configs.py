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
var_keys = [
    "test_type",
    "dataset",
    "send_telegram_after_complete",
    "seed",
    "sl_use_user_gdve_plus",
    "sl_use_item_gdve_plus",
    "use_CL"
]
case_test_type = ["Test"]
case_dataset = ["ml-100k"] #, "gowalla", "yelp2018"
case_telegram = [True]
case_seed = rd.sample(range(10000), 1) #also act as repeatition of test
case_gdve_plus = [True, False]
case_CL = [False]

#should match order of config key
var_set = [case_test_type, case_dataset, case_telegram, case_seed, case_gdve_plus, case_gdve_plus, case_CL]
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
    
    if base_config["sl_use_user_gdve_plus"] != base_config["sl_use_item_gdve_plus"]:
        continue

    # if var_values[-1] != var_values[-2]:
    #     continue
    if not os.path.exists(f"{args.path}_{file_index}"):
        os.makedirs(f"{args.path}_{file_index}")
    
    with open(os.path.join(f"{args.path}_{file_index}", f"{base_config['test_type']}_{base_config['dataset']}_{token}.yaml"), "w") as fd:
        yaml.dump(base_config, fd)

    if len(os.listdir(f"{args.path}_{file_index}")) > num_file_in_dir:
        file_index += 1