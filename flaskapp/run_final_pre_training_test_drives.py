# run n=10 test drives with each car and the default agent

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


import gym
from gym import wrappers

start=1
end=10

num_test_drives = 10

display = Display(visible=0, size=(1400,900))
display.start()

client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]

experiments = db.experiments
finished = list(experiments.find({"finished_training": {"$exists": True}, "pre_train_test_drive_results": {"$exists": False}}))


def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)
import pdb; pdb.set_trace()
# experiment = experiments.find_one({"user_id":user_id, "session_id":ObjectId(session_override_id), "experiment_type": experiment_type})
for e in finished:
    eid = e['_id']
    design = e['final_design']
    agent = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_1_0902_2232_avg_dqn_ep_0.h5' #e['final_agent']
    final_agent = e['final_agent']
    driver = DQNAgent(1, agent, design, replay_freq=50, lr=0.001)
    vid_dir = os.path.join(os.path.split(final_agent)[0], 'pre_train_test_drives') # put it in the directory with the first run to avoid confusion with other non-db runs
    os.makedirs(vid_dir)
    videos = []
    results = []
    for i in range(num_test_drives):
        driver.env.seed(i)
        video_filename = f'pre_train_test_drive_{i}.mp4'
        reset_driver_env(driver,vid_dir)
        result = driver.play_one(train=False,video_path=video_filename,eps=0.01)
        videos.append(str(os.path.join(vid_dir,video_filename)))
        results.append(result[0])
        driver.env.close()
    experiments.find_one_and_update({'_id':eid},{'$set':{'pre_train_test_drive_vids':videos,'pre_train_test_drive_results':results}})
