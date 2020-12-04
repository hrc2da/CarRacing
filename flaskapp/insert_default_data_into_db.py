import os
import glob
import sys
sys.path.append('/home/dev/scratch/cars/carracing_clean')
import re
import pickle as pkl
import json
import pymongo
from copy import deepcopy
from pymongo import MongoClient
from config import Config
from pilotsessions import users, sessions, session2keydict
from data_paths import human_roots, hybrid_roots, bo_roots, hybrid_designs, bo_designs
from bson.objectid import ObjectId 
from utils import config2car, car2config, chopshopfeatures2indexlist
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from pilotsessions import users, sessions
from pyvirtualdisplay import Display
import pymongo
import gym
from gym import wrappers
# from run_bo_designer import fill_out_config

num_test_drives = 10

display = Display(visible=0, size=(1400,900))
display.start()

def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)

client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]

all_sessions = db.sessions
experiments = db.experiments


# # get the roots for multiple agents at 500 with the defafult car
# default_roots = glob.glob("/home/dev/scratch/cars/carracing_clean/flaskapp/static/default_design_for_experiments/extra_training_run_with_this_design_*")
# default_roots.sort()

# default_design_file = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/car_config.json"
# # create a single experiment and insert each root as a trial path with results

# first_final_agent = glob.glob(f'{default_roots[0]}/*/*.h5')[0]

# experiment_results = []
# for p in default_roots:
#     reward_file = glob.glob(p+'/*/*rewards_flask.pkl')[0]
#     with open(reward_file,'rb') as infile:
#         rewards = pkl.load(infile)
#         assert rewards.shape == (250,)
#         experiment_results.append(list(rewards))


# with open(default_design_file,'r') as infile:
#     default_design = json.load(infile)

# new_experiment = {
#     "time_created": None,
#     "user_id": 'no_redesign',
#     "session_id": None,
#     "experiment_type": 'benchmark',
#     "garbage": False,
#     "finished_training": True,
#     "final_design": default_design,
#     "finished_designing": None,
#     "initial_agent": '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_1_0902_2232_avg_dqn_ep_0.h5',
#     "initial_design": default_design,
#     "last_modified": None,
#     "question_answers": None,
#     "ran_bo": False,
#     "started_designing": None,
#     "tested_design": [],
#     "tested_results": [],
#     "trial_paths": default_roots, # get this from datapaths
#     "trial_rewards": experiment_results, #experiment_results, # get this from pickle file in datapaths
#     "final_agent": first_final_agent, #final_agent,# get this from datapaths as well (will need to figure out where the h5 is; I need this agent to run the "final test drives")
#     # "finished_training": null ACTUALLY DELETE THIS KEY
#     # "final_test_drive_results": [],
#     # "final_test_drive_vids": [],
#     # "pre_train_test_drive_results": [],
#     # "pre_train_test_drive_vids": [],
#     "notes": "This is the benchmark data with NO redesign."
# }

# experiments.insert_one(new_experiment)
e = list(experiments.find({"user_id":"no_redesign"}))[0]
eid = e['_id']
design = e['final_design']
for trial_path in e['trial_paths']:
    agent = glob.glob(f'{trial_path}/*/joe*.h5')[0] #double check this please
    driver = DQNAgent(1, agent, design, replay_freq=50, lr=0.001)
    vid_dir = os.path.join(os.path.split(agent)[0], 'final_test_drives') # put it in the directory with the first run to avoid confusion with other non-db runs
    os.makedirs(vid_dir)
    videos = []
    results = []
    for i in range(num_test_drives):
        driver.env.seed(i)
        video_filename = f'final_test_drive_{i}.mp4'
        reset_driver_env(driver,vid_dir)
        result = driver.play_one(train=False,video_path=video_filename,eps=0.01)
        videos.append(str(os.path.join(vid_dir,video_filename)))
        results.append(result[0])
        driver.env.close()
    experiments.find_one_and_update({'_id':eid},{'$push':{'final_test_drive_vids':videos,'final_test_drive_results':results}})