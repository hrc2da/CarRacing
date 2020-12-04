from data_paths import dqn_roots, human_roots, human_designs, hybrid_roots, hybrid_designs, bo_roots, bo_designs, anonymous_keys, default_design, default_roots
from betterplotter import get_filenames, get_matrix, get_rewards
import json
import pickle as pkl
import itertools
import glob
import pymongo
import datetime
from pymongo import MongoClient, ReturnDocument
from config import Config
from bson.objectid import ObjectId
import csv

output_dict = dict()
output_list = list()

dqn_reward_files = [get_filenames(f,"*/total_rewards.pkl") for f in dqn_roots]
# default_rewards = get_matrix(default_reward_files)
dqn_rewards = [get_rewards(f[:3]) for f in dqn_reward_files]

# get some friendly names for keys
with open('flaskapp/nfl_names.csv') as infile:
    friendlynames = list(csv.reader(infile))

# get the db'ed results, then get the ones stored in files
client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
experiments = db.experiments

finished = list(experiments.find({"finished_training": {"$exists": True}}))


def get_type_key(experiment_type):
    if experiment_type== "human":
        return "human"
    if experiment_type == "bo_hybrid_select_features":
        return "hybrid_hum_select"    # human selects features to optimize
    if experiment_type == "full_bo_on_redesign":
        return "hybrid_hum_init"    # full bo on human's design
    if experiment_type == "full_bo":
        return "bo"
    if experiment_type == "bo_hybrid_select_features_start_from_redesign":
        return "hybrid_hum_init_select"    # human selects features to optimize but start from human's design
    if experiment_type == "bo_hybrid_select_features_by_confidence_start_from_user_design":
        return "hybrid_hum_confidence"    # human selects features NOT to optimize by confidence
    else:
        raise ValueError(f"Unexpected Experiment Type {experiment_type}")
     

for i,car in enumerate(finished):
    exp_type = get_type_key(car['experiment_type'])
    key = f'{friendlynames[i][0].lower().replace(" ","")}_{exp_type}'

    output_dict[key] = {
        'type': exp_type,
        'design': car['final_design'],
        'results': [dqn_rewards[1][:250] + r for r in car['trial_rewards']]
    }

    

for i,car in enumerate(finished):
    exp_type = get_type_key(car['experiment_type'])
    key = f'{friendlynames[i][0].lower().replace(" ","")}_{exp_type}'
    output_list.append({
        'key': key,
        'type': exp_type,
        'design': car['final_design'],
        'results': [dqn_rewards[1][:250] + r for r in car['trial_rewards']]
    })


import pdb; pdb.set_trace()



hybrid_reward_files = {k:list(itertools.chain(*[get_filenames(v, "*rewards_flask.pkl") for v in root_list])) for k,root_list in hybrid_roots.items()}
hybrid_rewards = {k:get_matrix(filenames) for k,filenames in hybrid_reward_files.items()}

hybrid_design_files = {k: glob.glob(design_file)[0] for k,design_file in hybrid_designs.items()}
hybrid_designs = dict()
for k,design_file in hybrid_design_files.items():
    with open(design_file) as infile:
        hybrid_designs[k] = json.load(infile)


for k,design in hybrid_designs.items():
    key = anonymous_keys[k] + "_Hybrid"
    if "color" not in design.keys():
        design["color"] = "0000cc"
    output_dict[key] = {
        'type': 'hybrid_hum_select',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in hybrid_rewards[k].tolist()]
    }



for k,design in hybrid_designs.items():
    key = anonymous_keys[k] + "_Hybrid"
    output_list.append({
        'key': key,
        'type': 'hybrid_hum_select',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in hybrid_rewards[k].tolist()]
    })








# human_files = [get_filenames(f,"*rewards_flask.pkl") for f in list(itertools.chain(*list(human_roots.values())))]
# hybrid_files = [get_filenames(f,"*rewards_flask.pkl") for f in list(itertools.chain(*list(hybrid_roots.values())))]

human_reward_files = {k:list(itertools.chain(*[get_filenames(v, "*rewards_flask.pkl") for v in root_list])) for k,root_list in human_roots.items()}
human_rewards = {k:get_matrix(filenames) for k,filenames in human_reward_files.items()}
human_design_files = {k: glob.glob(design_file)[0] for k,design_file in human_designs.items()}
human_designs = dict()
for k,design_file in human_design_files.items():
    with open(design_file) as infile:
        human_designs[k] = json.load(infile)


for k,design in human_designs.items():
    key = anonymous_keys[k] + "_Human"
    output_dict[key] = {
        'type': 'human',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in human_rewards[k].tolist()]
    }


for k,design in human_designs.items():
    key = anonymous_keys[k] + "_Human"
    output_list.append({
        'key': key,
        'type': 'human',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in human_rewards[k].tolist()]
    })


bo_reward_files = {k:list(itertools.chain(*[get_filenames(v, "*rewards_flask.pkl") for v in root_list])) for k,root_list in bo_roots.items()}
bo_rewards = {k:get_matrix(filenames) for k,filenames in bo_reward_files.items()}

bo_design_files = {k: glob.glob(design_file)[0] for k,design_file in bo_designs.items()}
bo_designs = dict()
for k,design_file in bo_design_files.items():
    with open(design_file) as infile:
        bo_designs[k] = json.load(infile)


for k,design in bo_designs.items():
    key = anonymous_keys[k] + "_BO"
    output_dict[key] = {
        'type': 'bo',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in bo_rewards[k].tolist()]
    }


for k,design in bo_designs.items():
    key = anonymous_keys[k] + "_BO"
    output_list.append({
        'key': key,
        'type': 'bo',
        'design': design,
        'results': [dqn_rewards[1][:250] + h for h in bo_rewards[k].tolist()]
    })




default_reward_files = [get_filenames(f,"*/*rewards_flask.pkl") for f in default_roots][0]
default_rewards = get_matrix(default_reward_files)

with open(default_design) as infile:
    default_car = json.load(infile)

output_dict["Default_Car"] = {
    'type': 'default',
    'design': default_car,
    'results': [dqn_rewards[1][:250] + h for h in default_rewards.tolist()]
}

output_list.append({
    'key': 'Default_Car',
    'type': 'default',
    'design': default_car,
    'results': [dqn_rewards[1][:250] + h for h in default_rewards.tolist()]
})


with open('flaskapp/d3plots/car_dict.json', 'w') as outfile:
    json.dump(output_dict, outfile)

with open('flaskapp/d3plots/car_list.json', 'w') as outfile:
    json.dump(output_list, outfile)


import pdb; pdb.set_trace()



