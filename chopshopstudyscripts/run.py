'''
Run N test drives for a given agent, design pair.
Optionally create a new experiment.
'''

import sys
import os
import uuid
from copy import deepcopy
from shutil import copy as shutil_copy
sys.path.append('/home/dev/scratch/cars/carracing_clean')
sys.path.append('/home/ubuntu/chopshop/carracing')
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
import csv
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import glob
import json
from skopt import gp_minimize

from utils import feature_labels, feature_ranges, car2config, config2car, densities, wheelmoment, blacklist, chopshopfeatures2indexlist, feature_labels, featuremask2names

import tensorboard
from tensorboardX import SummaryWriter
import argparse

def get_nfl_name(n):
    with open('nfl_names.csv', 'r') as infile:
        names = list(csv.reader(infile))
        return names[n][0].replace(' ','_') 



def create_experiment(experiment_type, user_id=None, session_id=None, experiment_seed=None):
    '''
    Create a new experiment NOT RUNNING ANYTHING HERE!

    # note: if you want to run multiple iterations of the same experiment, then create a new experiment for each one
    # for example, to run five iterations of training for one design, make five experiments with that design (session_id)
    # for the bo experiments, if session_id is None, then BO hasn't run yet, so create a new session

    Experiment Types:
    * h1control: default agent, default design
    * h1human: default agent, human design
    * h2control: retrain agent, default design
    * h2human: retrain agent, human design
    * h3bo: default agent, bo design
    * h4bo: retrain agent, bo design

    '''
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    session_id = ObjectId(session_id)
    friendly_name = get_nfl_name(experiments.count())
    params = {
        "time_created": datetime.datetime.utcnow(),
        "friendly_name": friendly_name,
        "user_id": user_id, 
        "session_id": session_id, # will create one if none
        "experiment_type": experiment_type,
        "experiment_seed": experiment_seed
    }
    experiment = experiments.insert_one(params)
    experiment_id = experiment.inserted_id

    # file structure:
    # root/experiments/<treatment>/<experiment_id>
    experiment_path = os.path.join(Config.FILE_PATH,'experiments',experiment_type,str(session_id),str(experiment_id))
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    default_design_path = os.path.join(Config.FILE_PATH,Config.DEFAULT_PATH,Config.DEFAULT_DESIGN)
    with open(default_design_path,'r') as designfile:
        default_design = json.load(designfile)
    shared_agent_path = os.path.join(Config.FILE_PATH,Config.DEFAULT_PATH,Config.DEFAULT_AGENT)
    default_agent = os.path.join(experiment_path,"default_agent.h5") # make a copy! (to prevent i/o errors w/ many jobs at once)
    shutil_copy(shared_agent_path,default_agent)
    if experiment_type == 'h1control':
        # default agent, default design
        params["user_id"] = "control"
        params['final_agent'] = params['initial_agent'] = str(default_agent)
        params['final_design'] = params['initial_design'] = default_design
        params['ran_bo'] = False
        params['tested_designs'] = [] # these are designs tested during designing, so leave blank
        params['tested_results'] = [] # ditto
        params['test_drive_vids'] = []
        params['test_drive_results'] = []


    elif experiment_type == 'h1human':
        # default agent, human design
        sessions = db.sessions
        session = sessions.find_one({'_id': ObjectId(session_id)})
        params['final_agent'] = params['initial_agent'] = default_agent
        params['initial_design'] = default_design
        params['final_design'] = session['final_design']
        params['ran_bo'] = False
        params['tested_designs'] = session['tested_designs'] # this is kind of unecessary, consider striking to save space
        params['tested_results'] = session['tested_results']
        params['test_drive_vids'] = []
        params['test_drive_results'] = []

    elif experiment_type == 'h2control':
        # retrain the agent, default design
        params["user_id"] = "control"
        params['initial_agent'] = default_agent
        params['final_agent'] = ''
        params['final_design'] = params['initial_design'] = default_design
        params['ran_bo'] = False
        params['tested_designs'] = [] # these are designs tested during designing, so leave blank
        params['tested_results'] = [] # ditto
        params['test_drive_vids'] = []
        params['test_drive_results'] = []

    elif experiment_type == 'h2human':
        # retrain the agent, human design
        sessions = db.sessions
        session = sessions.find_one({'_id': session_id})
        params['initial_agent'] = default_agent
        params['final_agent'] = ''
        params['initial_design'] = default_design
        params['final_design'] = session['final_design']
        params['ran_bo'] = False
        params['tested_designs'] = session['tested_designs'] # this is kind of unecessary, consider striking to save space
        params['tested_results'] = session['tested_results']
        params['test_drive_vids'] = []
        params['test_drive_results'] = []

    elif experiment_type == 'h3bo':
        # default agent, bo design
        user_id = "bo"
        sessions = db.sessions
        timestamp = datetime.datetime.utcnow()
        session = sessions.insert_one({'_id': session_id, 
                                        'user_id': user_id,
                                        'agent': default_agent,
                                        'time_created': timestamp,
                                        'last_modified': timestamp,
                                        'initial_design': default_design,
                                        'final_design': {},
                                        'n_training_episodes': Config.N_TRAINING_EPISODES,
                                        'file_path': os.path.join(Config.FILE_PATH,user_id,str(session_id))}) #not sure about this, may put in experiments dir instead
        
        params["user_id"] = user_id
        params['final_agent'] = params['initial_agent'] = default_agent
        params['initial_design'] = default_design
        params['final_design'] = {}
        params['ran_bo'] = False
        params['tested_designs'] = [] # these are designs tested during designing, so leave blank
        params['tested_results'] = [] # ditto
        params['test_drive_vids'] = []
        params['test_drive_results'] = []

    elif experiment_type == 'h4bo':
        # retrain agent, bo design
        user_id = "bo"
        params["user_id"] = user_id
        params['initial_agent'] = default_agent
        params['final_agent'] = ''
        params['initial_design'] = default_design
        params['final_design'] = {}
        params['ran_bo'] = False
        params['tested_designs'] = [] # these are designs tested during designing, so leave blank
        params['tested_results'] = [] # ditto
        params['test_drive_vids'] = []
        params['test_drive_results'] = []
        params['test_drive_starts'] = []
        params['test_drive_ends'] = []
    experiments.find_one_and_update({"_id": experiment_id},{"$set":params})
    return experiment_id

def run_experiment(experiment_id):
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    # find experiment by id
    experiment = experiments.find_one({"_id": experiment_id})
    
    if experiment['experiment_type'] == 'h1control':
        # don't run anything ,the experiment is done, just need to run test drives
        print("H1 Control has nothing to run")
        return True
    elif experiment['experiment_type'] == 'h1human':
        print("H1 Human has nothing to run")
        return True
    elif experiment['experiment_type'] == 'h2control':
        # run the training, if it hasn't run
        if experiment['final_agent'] == '':
            run_training(experiment)
        else:
            print("H2 Control agent has already been trained. To retrain, create a new experiment.")
            return True
    elif experiment['experiment_type'] == 'h2human':
        # run the training, if it hasn't run
        if experiment['final_agent'] == '':
            run_training(experiment)
        else:
            print("H2 Human agent has already been trained. To retrain, create a new experiment.")
            return True
    elif experiment['experiment_type'] == 'h3bo':
        print("got h3bo")
        # check for design, if no design run the bo
        if experiment['final_design'] == {}:
            run_bo(experiment)
        else:
            print("Current experiment H3 BO has a design. To redesign, create a new experiment.")
            return True
    elif experiment['experiment_type'] == 'h4bo':
        # check for design, if no design run the bo
        if experiment['final_design'] == {}:
            run_bo(experiment)
        if experiment['final_agent'] == '':
            run_training(experiment)
        else:
            print("H4 BO has a design and final agent. To redesign/retrain, create a new experiment.")
            return True


def timing_callback(res):
    # global bo_timestamps
    ts = time.time()
    # bo_time_intervals.append(ts-bo_timestamps[-1])
    # print(f'BO Step {len(bo_timestamps)-1} Time: {bo_time_intervals[-1]}, Result: {res["fun"]}')
    # bo_timestamps.append(ts)
    return res
    

def fill_out_config(config, base_config, mask_indices):
    full_config = deepcopy(base_config)
    for i,idx in enumerate(mask_indices):
        full_config[idx] = config[i]
    return full_config


def test_drive(config, driver, base_config, mask_indices):
    # takes config and base config as arrays (not dicts)
    if len(config) < len(base_config):
        full_config = fill_out_config(config, base_config, mask_indices)
    else:
        full_config = config
    driver.carConfig = config2car(full_config)
    return -driver.play_one(eps=0.01,train=False)[0]

def design_step(x0,y0,driver,base_config,mask_indices,iters=15,seed=42, acq_func="gp_hedge", kappa=1.96,mask_eps=0.1):
    masked_designbounds = [b for i,b in enumerate(feature_ranges) if i in mask_indices]
    x0_masked = [list(np.array(design)[mask_indices]) for design in x0]
    res = gp_minimize(lambda x: test_drive(x,driver,base_config, mask_indices), masked_designbounds, acq_func="EI", n_calls=iters, n_initial_points=5, x0=x0_masked, y0=y0, random_state=seed, n_jobs=8, kappa=kappa)
    for i in range(len(x0)):
        res.x_iters[i] = x0[i]
    for i in range(len(x0),len(res.x_iters)):
        res.x_iters[i] = fill_out_config(res.x_iters[i],base_config,mask_indices)
    res.x = fill_out_config(res.x,base_config,mask_indices)
    return res

def policy_step(config, driver, episodes=15):
    driver.num_episodes = episodes
    if config is not None:
        driver.carConfig = config
    return driver.train()

def setup_training_paths(experiment):
    agent = experiment['initial_agent']
    # user_id = experiment['user_id']
    # session = experiment['session']
    train_dir = os.path.join(os.path.split(agent)[0], 'training') 
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # this is a bit paranoid, but make a copy of the agent file
    shutil_copy(agent,train_dir)
    agent_file = str(os.path.join(train_dir,os.path.basename(agent)))
    return train_dir, agent_file


def run_training(experiment, seed=None, n=Config.N_TRAINING_EPISODES):
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    design = experiment['final_design']
    experiment_seed = experiment['experiment_seed']
    train_dir, agent_file = setup_training_paths(experiment)
    driver = DQNAgent(n, agent_file, design, train_dir=train_dir) 
    driver.env.seed(experiment_seed)
    rewards = policy_step(design, driver, n)

    final_agent_path = os.path.join(train_dir,"final_agent.h5")
    driver.model.save(final_agent_path)
    timestamp = datetime.datetime.utcnow()
    experiment = experiments.find_one_and_update({'_id':experiment['_id']},
        {'$set': {'training_rewards': rewards.tolist(),
                    'final_agent': final_agent_path,
                    'finished_training': timestamp,
                    'last_modified': timestamp}}

    )  

def run_bo(experiment):
    print("running bo")
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    sessions = db.sessions
    session = sessions.find_one({"_id": experiment["session_id"]})
    initial_design = experiment["initial_design"]
    base_config = car2config(initial_design)
    agent = experiment['initial_agent'] # bo runs before any retrain
    mask_indices = []
    for index,value in enumerate(base_config):
        if index not in blacklist: #user_config[index] != value:
            try:
                mask_indices.append(index)
            except ValueError as e:
                mask_indices = []
                raise(e)

    x0 = [base_config]
    driver = DQNAgent(1, agent, initial_design)
    init_result = test_drive(base_config, driver, base_config, mask_indices)
    y0 = np.array([init_result])    
    seed = int(experiment['_id'].generation_time.timestamp())
    results = design_step(x0,y0,driver,base_config,mask_indices,acq_func='EI',iters=Config.N_DESIGN_STEPS, seed=seed)
    best_config = results.x
    print(f'Best design reward: {results.fun}')
    # if x0 is None:
    x0 = results.x_iters
    y0 = results.func_vals
    print(f'x0:{len(x0)},y0:{y0.shape}')

    # insert into db x0 as bo_designs, y0 as bo_rewards, best_config as final_design
    # set ran_bo True
    timestamp = datetime.datetime.utcnow()
    
    experiment = experiments.find_one_and_update({'_id':experiment['_id']},
        {'$set': {'bo_designs': [config2car(x) for x in x0],
                    'bo_rewards': y0.tolist(),
                    'final_design': config2car(best_config),
                    'last_modified': timestamp,
                    'finished_bo': timestamp,
                    'seed': seed,
                    'ran_bo': True}},
        return_document = ReturnDocument.AFTER 
    )
    session = sessions.find_one_and_update({'_id':session['_id']},
        {'$set': {'tested_designs': [config2car(x) for x in x0],
                    'tested_results': y0.tolist(),
                    'tested_videos': [], # empty for now, try to write these in the eval callback somehow
                    'final_design': config2car(best_config),
                    'last_modified': timestamp,
                    'finished_bo': timestamp,
                    'ran_bo': True}},
        return_document = ReturnDocument.AFTER 
    )
    return


def run_test_drives(experiment_id, seeds, n=10):
    start_time = time.time()
    display = Display(visible=0, size=(1400,900))
    display.start()
    client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
    db = client[Config.DB_NAME]
    experiments = db.experiments
    experiment = experiments.find_one({"_id": experiment_id})
    def reset_driver_env(driver, vid_path):
        env = gym.make('CarRacingTrain-v1')
        driver.env = wrappers.Monitor(env, vid_path, force=False, 
                    resume = True, video_callable=lambda x : True, mode='evaluation',
                    write_upon_reset=False)
    eid = experiment['_id']
    design = experiment['final_design']
    agent = experiment['final_agent']

    driver = DQNAgent(1, agent, design)
    # put it in the directory with the first run to avoid confusion with other non-db runs
    vid_dir = os.path.join(os.path.split(agent)[0], 'test_drives') 
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)
    existing_vids = glob.glob(f"{vid_dir}/*.mp4")
    n_prior = len(existing_vids)
    videos = []
    results = []
    for i in range(n):
        
        video_filename = f'test_drive_{i+n_prior}.mp4'
        reset_driver_env(driver,vid_dir)
        driver.env.seed(seeds[i])
        driver.env.reset()
        result = driver.play_one(train=False,video_path=video_filename,eps=0.01)
        videos.append(str(os.path.join(vid_dir,video_filename)))
        results.append(result[0])
        driver.env.close()
    end_time = time.time()
    # append to db
    experiments.find_one_and_update({'_id':eid},{'$push':{'test_drive_vids':videos,'test_drive_results':results,'test_drive_starts':start_time, 'test_drive_ends':end_time}})
    

def main(experiment_type, user_id, session_id, experiment_seed=None):

    test_drive_seeds = [x*10 + 1 for x in range(Config.N_TEST_DRIVES)]

    if experiment_type == 'h1control':
        experiment_id = create_experiment(experiment_type, experiment_seed=experiment_seed)
        run_experiment(experiment_id)
        run_test_drives(experiment_id,test_drive_seeds,Config.N_TEST_DRIVES)
    elif experiment_type == 'h1human':
        experiment_id = create_experiment(experiment_type, user_id, session_id, experiment_seed=experiment_seed)
        run_experiment(experiment_id)
        run_test_drives(experiment_id,test_drive_seeds,Config.N_TEST_DRIVES)
    elif experiment_type == 'h2control':
        experiment_id = create_experiment(experiment_type, experiment_seed=experiment_seed)
        run_experiment(experiment_id)
        run_test_drives(experiment_id,test_drive_seeds,Config.N_TEST_DRIVES)
    elif experiment_type == 'h2human':
        experiment_id = create_experiment(experiment_type, user_id, session_id, experiment_seed=experiment_seed)
        run_experiment(experiment_id)
        run_test_drives(experiment_id,test_drive_seeds,Config.N_TEST_DRIVES)
    elif experiment_type == 'h3bo':
        experiment_id = create_experiment(experiment_type, experiment_seed=experiment_seed)
        run_experiment(experiment_id)
        run_test_drives(experiment_id,test_drive_seeds,Config.N_TEST_DRIVES)
    else:
        raise(NotImplementedError)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--experiment_type', '-e')
    parser.add_argument('--user', '-u')
    parser.add_argument('--session', '-s')
    parser.add_argument('--random_seed', '-r')
    # for h1 types, create experiment and run test drives
    args = parser.parse_args()
    if args.random_seed is not None:
        args.random_seed = int(args.random_seed)
    main(args.experiment_type, args.user, args.session, args.random_seed)