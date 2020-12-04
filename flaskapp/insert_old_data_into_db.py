import os
import glob
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
# from run_bo_designer import fill_out_config

def fill_out_config(config, mask_indices, base_config):
    full_config = deepcopy(base_config)
    for i,idx in enumerate(mask_indices):
        full_config[idx] = config[i]
    return full_config


client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]

all_sessions = db.sessions
experiments = db.experiments

old_sessions = [ObjectId(s) for s in sessions[:8]]
old_users = users[:8]

# get the session data for all the old (pre-experiment in db) users
old_sessions_records = all_sessions.find({"_id": {"$in": old_sessions}})

import pdb; pdb.set_trace()
# for each of the 8 old users, create a "human" experiment for them
'''
fields:
    - _id
    - time_created: null
    - user_id: session['user_id']
    - session_id: session['_id']
    - experiment_type : 'human'
    - garbage: false
    - final_design: session['final_design']
    - finished_designing: null
    - initial_agent: session['agent']
    - initial_design: session['initial_design']
    - last_modified: null
    - question_answers: session['question_answers']
    - ran_bo: false
    - started_designing: null
    - tested_design: session['tested_designs']
    - tested_results: session['tested_results']
    - trial_paths: [] # get this from datapaths
    - trial_rewards: [] # get this from pickle file in datapaths
    - final_agent: # get this from datapaths as well (will need to figure out where the h5 is; I need this agent to run the "final test drives")
    - finished_training: null ACTUALLY DELETE THIS KEY
    - final_test_drive_results: []
    - final_test_drive_vids: []
    - pre_train_test_drive_results: []
    - pre_train_test_drive_vids: []


'''
# for session in old_sessions_records:
#     roots = human_roots[session2keydict[str(session['_id'])]]
#     experiment_paths = [os.path.split(roots[0])[0]]
#     final_agent = glob.glob(roots[0]+"/*.h5")[0]
#     # repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]+".h5")]
#     repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]) if re.match(rf'{roots[1][:-3]}.*/[0-9,_]+',g)]
#     experiment_paths += repeats
#     experiment_results = []
#     for p in experiment_paths:
#         reward_file = glob.glob(p+'/*/*rewards_flask.pkl')[0]
#         with open(reward_file,'rb') as infile:
#             rewards = pkl.load(infile)
#             assert rewards.shape == (250,)
#             experiment_results.append(list(rewards))

#     new_experiment = {
#         "time_created": None,
#         "user_id": session['user_id'],
#         "session_id": session['_id'],
#         "experiment_type": 'human',
#         "garbage": False,
#         "final_design": session['final_design'],
#         "finished_designing": None,
#         "initial_agent": session['agent'],
#         "initial_design": session['initial_design'],
#         "last_modified": None,
#         "question_answers": session['question_answers'],
#         "selected_features": session['selected_features']
#         "ran_bo": False,
#         "started_designing": None,
#         "tested_design": session['tested_designs'],
#         "tested_results": session['tested_results'],
#         "trial_paths": experiment_paths, # get this from datapaths
#         "trial_rewards": experiment_results, # get this from pickle file in datapaths
#         "final_agent": final_agent,# get this from datapaths as well (will need to figure out where the h5 is; I need this agent to run the "final test drives")
#         # "finished_training": null ACTUALLY DELETE THIS KEY
#         "final_test_drive_results": [],
#         "final_test_drive_vids": [],
#         "pre_train_test_drive_results": [],
#         "pre_train_test_drive_vids": [],
#         "notes": "Manually inserted early pilot data"
#     }
#     experiments.insert_one(new_experiment)

#     import pdb; pdb.set_trace()
##################################################################################
# for session in old_sessions_records:
#     user_name = session2keydict[str(session['_id'])]
#     roots = hybrid_roots[user_name]
#     experiment_paths = [os.path.split(roots[0])[0]]

#     bo_testing_file = os.path.join(experiment_paths[0],'designs.pkl')
#     with open(bo_testing_file,'rb') as infile:
#         bo_designs_tested = pkl.load(infile)
    

#     base_config = car2config(session['initial_design'])
#     if 'selected_features' in session.keys():
#         mask_indices = chopshopfeatures2indexlist(session['selected_features'])
#     else:
#         print("No Selected Features, using a Diff with the Default Design instead")
#         mask_indices = []
#         user_config = car2config(session['final_design'])
#         total_features = 0
#         for index,value in enumerate(base_config):
#             total_features += 1
#             if user_config[index] != value:
#                 try:
#                     mask_indices.append(index)
#                 except ValueError as e:
#                     mask_indices = []
#                     raise(e)
#         print(f"From the diff: {len(mask_indices)} out of {total_features} are different.")

#     tested_designs = [config2car(fill_out_config(d,mask_indices,base_config)) for d in bo_designs_tested[0]]
#     tested_results = list(bo_designs_tested[1])
    
#     final_agent = glob.glob(roots[0]+"/*.h5")[0]
#     # repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]+".h5")]
#     repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]) if re.match(rf'{roots[1][:-3]}.*/[0-9,_]+',g)]
#     experiment_paths += repeats
#     experiment_results = []
#     for p in experiment_paths:
#         reward_file = glob.glob(p+'/*/*rewards_flask.pkl')[0]
#         with open(reward_file,'rb') as infile:
#             rewards = pkl.load(infile)
#             assert rewards.shape == (250,)
#             experiment_results.append(list(rewards))

#     design_file = glob.glob(hybrid_designs[user_name])[0]
#     with open(design_file, 'r') as infile:
#         hybrid_design = json.load(infile)
    
#     new_experiment = {
#         "time_created": None,
#         "user_id": session['user_id'],
#         "session_id": session['_id'],
#         "experiment_type": 'hybrid_hum_select',
#         "garbage": False,
#         "finished_training": True,
#         "final_design": hybrid_design,
#         "finished_designing": None,
#         "initial_agent": session['agent'],
#         "initial_design": session['initial_design'],
#         "last_modified": None,
#         "question_answers": session['question_answers'],
#         "ran_bo": False,
#         "started_designing": None,
#         "tested_design": tested_designs,
#         "tested_results": tested_results,
#         "trial_paths": experiment_paths, # get this from datapaths
#         "trial_rewards": experiment_results, # get this from pickle file in datapaths
#         "final_agent": final_agent,# get this from datapaths as well (will need to figure out where the h5 is; I need this agent to run the "final test drives")
#         # "finished_training": null ACTUALLY DELETE THIS KEY
#         # "final_test_drive_results": [],
#         # "final_test_drive_vids": [],
#         # "pre_train_test_drive_results": [],
#         # "pre_train_test_drive_vids": [],
#         "notes": "Manually inserted early pilot data. Had to recreate feature masks (some from selected features, some from diff)"
#     }
    
#     experiments.insert_one(new_experiment)

for k,roots in bo_roots.items():
    # user_name = session2keydict[str(session['_id'])]
    # roots = hybrid_roots[user_name]
    experiment_paths = [os.path.split(roots[0])[0]]
    
    bo_testing_file = os.path.join(experiment_paths[0],'designs.pkl')
    with open(bo_testing_file,'rb') as infile:
        bo_designs_tested = pkl.load(infile)
    
    tested_designs = [config2car(d) for d in bo_designs_tested[0]]
    tested_results = list(bo_designs_tested[1])
    

    final_agent = glob.glob(roots[0]+"/*.h5")[0]
    # repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]+".h5")]
    repeats = [os.path.split(g)[0] for g in glob.glob(roots[1]) if re.match(rf'{roots[1][:-3]}.*/[0-9,_]+',g)]
    experiment_paths += repeats
    experiment_results = []
    for p in experiment_paths:
        reward_file = glob.glob(p+'/*/*rewards_flask.pkl')[0]
        with open(reward_file,'rb') as infile:
            rewards = pkl.load(infile)
            assert rewards.shape == (250,)
            experiment_results.append(list(rewards))

    design_file = glob.glob(bo_designs[k])[0]
    with open(design_file, 'r') as infile:
        bo_design = json.load(infile)

    initial_design_file = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/car_config.json'
    with open(initial_design_file, 'r') as infile:
        initial_design = json.load(infile)
    
    new_experiment = {
        "time_created": None,
        "user_id": k,
        "session_id": None,
        "experiment_type": 'full_bo',
        "garbage": False,
        "finished_training": True,
        "final_design": bo_design,
        "finished_designing": None,
        "initial_agent": '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_1_0902_2232_avg_dqn_ep_0.h5',
        "initial_design": initial_design,
        "last_modified": None,
        "question_answers": None,
        "ran_bo": False,
        "started_designing": None,
        "tested_design": tested_designs,
        "tested_results": tested_results,
        "trial_paths": experiment_paths, # get this from datapaths
        "trial_rewards": experiment_results, # get this from pickle file in datapaths
        "final_agent": final_agent,# get this from datapaths as well (will need to figure out where the h5 is; I need this agent to run the "final test drives")
        # "finished_training": null ACTUALLY DELETE THIS KEY
        # "final_test_drive_results": [],
        # "final_test_drive_vids": [],
        # "pre_train_test_drive_results": [],
        # "pre_train_test_drive_vids": [],
        "notes": "Manually inserted early pilot data for full bo. No session b/c these just start from default design."
    }
    # import pdb; pdb.set_trace()
    print(f"INSERTED {k}")
    experiments.insert_one(new_experiment)