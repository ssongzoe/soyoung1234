import yaml
import secrets
import itertools
import argparse

TOKEN_LENGTH = 6

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".")
# {test_type}_{data}_{token}.yamln

with open("base.yaml") as fd:
    base_config = yaml.load(fd, Loader=yaml.FullLoader)

#-------------------------------------correct below--------------------------------------------
#config variation set
#should match order of config key
var_keys = ["test_type", "dataset", "use_SL"]
#should set same key of config.yaml
case_test_type = ["noSL"]
case_dataset = ["ml-100k", "gowalla"] #, "gowalla", "yelp2018"
case_use_SL = [True, False]

var_set = [case_test_type, case_dataset, case_use_SL]
#----------------------------------------------------------------------------------------------
assert len(var_keys) == len(var_set)

for var_values in itertools.product(*var_set):
    token = secrets.token_hex(TOKEN_LENGTH//2)
    base_config["config_token"] = token
    for key, val in zip(var_keys, var_values):
        base_config[key] = val
        # if key == "dataset":
        #     base_config["data_path"] = f"./../../../data/{val}"
    with open(f"{base_config['test_type']}_{base_config['dataset']}_{token}.yaml", "w") as fd:
        yaml.dump(base_config, fd)