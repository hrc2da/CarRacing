# run the bayesopt designer over the full set of config vars, then train for N steps
import sys
import os
import uuid
from copy import deepcopy
from shutil import copy as shutil_copy
sys.path.append('/home/dev/scratch/cars/carracing_clean')
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from config import Config
from pyvirtualdisplay import Display
import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
import gym
from gym import wrappers
import time
import datetime

from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import glob
from skopt import gp_minimize

from utils import feature_labels, feature_ranges, car2config, config2car, densities, wheelmoment, blacklist, chopshopfeatures2indexlist, feature_labels, featuremask2names

n_design_steps = 30

# activate carracing_clean20
testing = False
run_on = 0 # amit
experiment_type = "full_bo"
# user_id = users[run_on]
# session_override_id = sessions[run_on]

# For full bo (not redesigned) we use both the default agent and the default design to start.
# So there is no real need to attach to a particular session. Instead, we attach to a specific
# baseyopt_only session that gives the path to both the default agent and design
user_id = "bayesopt_only"
session_override_id = "5fae021ecf951ee063d0412f"

print(f'Running {experiment_type} User: {user_id}, Session: {session_override_id}')

display = Display(visible=0, size=(1400,900))
display.start()

def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)

# first connect to pymongo and get all the sessions that have a complete status
client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
experiments = db.experiments

experiment = experiments.insert_one({
    "time_created": datetime.datetime.utcnow(),
    "user_id": user_id, 
    "session_id": ObjectId(session_override_id), 
    "experiment_type": experiment_type,
    "garbage": testing
    })
experiment_id = experiment.inserted_id

sessions = db.sessions
if session_override_id is not None:
    session = sessions.find_one({"user_id": user_id, "_id": ObjectId(session_override_id)},sort=[("_id",pymongo.ASCENDING)])

# else:
#     session = sessions.find_one({"user_id": user_id,"status":"complete"},sort=[("_id",pymongo.ASCENDING)])
# get the agent to optimize for
agent_full_path = session['agent']
selected_features = session['selected_features']
base_config = car2config(session['initial_design'])

new_session_dir = os.path.join(Config.FILE_PATH,user_id,"bo_sessions_all_features",str(session["_id"])) # session_id is an ObjectID, not str
train_dir = os.path.join(new_session_dir,f'{n_design_steps}_step_policy_training')
counter = 0
while(os.path.isdir(train_dir)):
    train_dir = os.path.join(new_session_dir,f'{n_design_steps}_step_policy_training_{counter}')
    counter += 1
os.makedirs(train_dir)
shutil_copy(agent_full_path,new_session_dir)
agent_file = str(os.path.join(new_session_dir,os.path.basename(agent_full_path)))

num_episodes = session['n_training_episodes']

# initialize a driver for now
driver = DQNAgent(num_episodes, agent_file, base_config, train_dir=train_dir) 

mask_indices = []
# for attribute,value in base_config.items():
#     if user_config[attribute] != value:
#         try:
#             mask_indices.append(feature_labels.index(attribute))
#         except ValueError as e:
#             mask_indices = []
#             raise(e)
for index,value in enumerate(base_config):
    if index not in blacklist: #user_config[index] != value:
        try:
            mask_indices.append(index)
        except ValueError as e:
            mask_indices = []
            raise(e)

def fill_out_config(config):
    global base_config
    global mask_indices
    full_config = deepcopy(base_config)
    for i,idx in enumerate(mask_indices):
        full_config[idx] = config[i]
    return full_config


def test_drive(config):
    global base_config
    global driver
    if len(config) < len(base_config):
        full_config = fill_out_config(config)
    else:
        full_config = config
    driver.carConfig = config2car(full_config)
    return -driver.play_one(eps=0.01,train=False)[0]

bo_timestamps = [time.time()]
bo_time_intervals = [0]
def timing_callback(res):
    global bo_timestamps
    ts = time.time()
    bo_time_intervals.append(ts-bo_timestamps[-1])
    print(f'BO Step {len(bo_timestamps)-1} Time: {bo_time_intervals[-1]}, Result: {res["fun"]}')
    bo_timestamps.append(ts)
    
    

def design_step(x0,y0,iters=15,seed=42, acq_func="gp_hedge", kappa=1.96,mask_eps=0.1):
    global feature_ranges
    global mask_indices
    # masked_designbounds = deepcopy(designbounds)
    # for i,bound in enumerate(masked_designbounds):
    #     if i not in mask_indices:
    #         bnd = current_config[i]
    #         if type(bnd) == Real:
    #             masked_designbounds[i] = (bnd-mask_eps,bnd+mask_eps)
    #         else:
    #             masked_designbounds[i] = (bnd-1,bnd+1)
    masked_designbounds = [b for i,b in enumerate(feature_ranges) if i in mask_indices]
    x0_masked = [list(np.array(design)[mask_indices]) for design in x0]
    # reinstantiate the bopt every time because we may want to filter the features we care about
    res = gp_minimize(test_drive, masked_designbounds, acq_func=acq_func, n_calls=iters, x0=x0_masked, y0=y0, n_random_starts=5, random_state=seed, n_jobs=8, kappa=kappa,callback=timing_callback)
    for i in range(len(x0)):
        res.x_iters[i] = x0[i]
    for i in range(len(x0),len(res.x_iters)):
        res.x_iters[i] = fill_out_config(res.x_iters[i])
    res.x = fill_out_config(res.x)
    return res

def policy_step(config,episodes=15):
    global driver
    # retrain the driver on the current car for n_episodes
    driver.num_episodes = episodes
    if config is not None:
        driver.carConfig = config
    return driver.train()

#insert the initial design and the trial
timestamp = datetime.datetime.utcnow()
experiment = experiments.find_one_and_update({'_id':experiment_id},
    {'$set': {'initial_design': session['initial_design'],
                'selected_features': featuremask2names(mask_indices),
                'initial_agent': agent_file,
                'trial_paths': [train_dir],
                'started_bo': timestamp,
                'last_modified':timestamp}},
    return_document = ReturnDocument.AFTER 
)


# do a test drive with the base config, x0, to get an initial reward value, y0
x0 = [base_config]
init_result = test_drive(base_config)
y0 = np.array([init_result])
results = design_step(x0,y0,acq_func='LCB',iters=n_design_steps) #,acq_func=acq_func,kappa=4)
best_config = results.x
print(f'Best design reward: {results.fun}')
# if x0 is None:
x0 = results.x_iters
y0 = results.func_vals
print(f'x0:{len(x0)},y0:{y0.shape}')

# insert into db x0 as bo_designs, y0 as bo_rewards, best_config as final_design
# set ran_bo True
timestamp = datetime.datetime.utcnow()
experiment = experiments.find_one_and_update({'_id':experiment_id},
    {'$set': {'bo_designs': [config2car(x) for x in x0],
                'bo_rewards': y0.tolist(),
                'final_design': config2car(best_config),
                'last_modified': timestamp,
                'finished_bo': timestamp,
                'ran_bo': True}},
    return_document = ReturnDocument.AFTER 
)

with open(f'{train_dir}/designs.pkl','wb+') as design_dump:
    pkl.dump([x0,y0,best_config,bo_time_intervals],design_dump) # these designs should be in order, check y0 for rewards

# now retrain with the best design and see how it goes
rewards = policy_step(config2car(best_config),num_episodes)

# dump the latest model
final_agent_path = os.path.join(train_dir,"final_agent.h5")
driver.model.save(final_agent_path)
timestamp = datetime.datetime.utcnow()
experiment = experiments.find_one_and_update({'_id':experiment_id},
    {'$set': {'trial_rewards': rewards.tolist(),
                'final_agent': final_agent_path,
                'finished_training': timestamp,
                'last_modified': timestamp}}

)    





