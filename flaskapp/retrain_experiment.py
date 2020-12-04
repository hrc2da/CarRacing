# rerun the training stage of a particular experiment

# how to use this script:
# choose the user to run on and the treatment.

import pickle as pkl
import json
import os
import sys
import glob
from utils import config2car, car2config
from shutil import copy as shutil_copy
sys.path.append('/home/dev/scratch/cars/carracing_clean')
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from pilotsessions import users, sessions
from pyvirtualdisplay import Display
import pymongo
import datetime
from pymongo import MongoClient, ReturnDocument
from config import Config
from bson.objectid import ObjectId
# pull the final_design from an experiment

# specify user_id
testing = False
run_on = 11 #8 george gonzalo

# optimize over the non-selected confident features
user_id = users[run_on]
session_override_id = sessions[run_on]

if testing == True:
    user_id = "4248bf467c7b4a27afaaca841634a028"
    session_override_id = "5f6d0b7b673446edf0d88f1d"


# specify experiment_type
experiment_type = "human"
# experiment_type = "bo_hybrid_select_features"
# experiment_type = "full_bo_on_redesign"
# experiment_type = "full_bo"
# experiment_type = "bo_hybrid_select_features_start_from_redesign"
# experiment_type = "bo_hybrid_select_features_by_confidence_start_from_user_design"

# specify which one if there's more than one
print(f'Running {experiment_type} Retrain for User: {user_id}, Session: {session_override_id}')
# add the results to trial_paths

# replace trial_rewards with [trial_rewards] and append the new trial rewards
# ^^ consider scripting this and refactoring the code to add it as a list of lists

# get the experiment





#eventually we want to get this from the db but I think the initial_agent actually got overwritten for all the db entries, so I need to fix that first.
default_agent_path = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_1_0902_2232_avg_dqn_ep_0.h5'
num_episodes = 250

display = Display(visible=0, size=(1400,900))
display.start()

client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]

experiments = db.experiments

experiment = experiments.find_one({"user_id":user_id, "session_id":ObjectId(session_override_id), "experiment_type": experiment_type})

if experiment is None:
    raise ValueError("No experiment of that type found!")
experiment_id = experiment['_id']
car_config = experiment['final_design']

# with open(design_json, 'r') as jsonfile:
#     car_config = json.load(jsonfile)

# setup the output paths for agent and rewards pickle
train_dir = experiment["trial_paths"][0]
train_dir_root = experiment["trial_paths"][0]
# if 
counter = 1
while(os.path.isdir(train_dir)):
    train_dir = f'{train_dir_root}_extra_training_{counter}'
    counter += 1
# make a copy of the default agent to this path
agent_file = 'copy of the default agent in this path'
os.makedirs(train_dir)
shutil_copy(default_agent_path,train_dir)
agent_file = str(os.path.join(train_dir,os.path.basename(default_agent_path)))
# run X episodes
driver = DQNAgent(num_episodes, agent_file, car_config, replay_freq=50, lr=0.001, train_dir = str(train_dir))
rewards = driver.train()
# dump agent and rewards

final_agent_path = os.path.join(train_dir,"final_agent.h5")
driver.model.save(final_agent_path)
timestamp = datetime.datetime.utcnow()
experiment = experiments.find_one_and_update({'_id':experiment_id},
    {'$push': {'trial_rewards': rewards.tolist(), 'trial_paths': train_dir},
    '$set': {'last_modified': timestamp}}
)







##################################################
# Testing that pickled design is the same as json
##################################################
# with open(design_json, 'r') as jsonfile:
#     design_dict = json.load(jsonfile)
#     design_arr = car2config(design_dict)

# with open(design_pkl, 'rb') as pklfile:
#     design_compare_arr = pkl.load(pklfile)[-1]
#     design_compare_dict = config2car(design_compare_arr)


# # design_keys = list(design_dict.keys())
# # design_compare_keys = list(design_compare_dict.keys())
# # for i,k in enumerate(design_keys):
# #     assert k == design_compare_keys[i]
# for i in range(len(design_arr)):
#     assert design_arr[i] == design_compare_arr[i]
# for k,v in design_dict.items():
#     try:
#         assert v == design_compare_dict[k]
#     except AssertionError as e:
#         print(f'ASSERTION FAIL: {k}:{v} != {design_compare_dict[k]}')
#         # raise e

# print("the json is the same as the pickle!")