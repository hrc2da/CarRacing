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
from pymongo import MongoClient
from bson.objectid import ObjectId
import gym
from gym import wrappers

from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import glob
from skopt import gp_minimize

from utils import feature_labels, feature_ranges, car2config, config2car


# user_id = 'a3476ba7c278432db0315eda9546b7a4' # amit
# session_override_id = "5f6d02c2673446edf0d88f1c" # first 250 session

user_id = 'e6900ed30d77497a97b8b9800d3becdf' #dan
session_override_id = '5f6cd5e2d8a6d9430d007bf3' #dan

densities = [7,10,15,18]

# feature_ranges = [(1e3,1e6), #eng_power          0
#                 (0.01,5), #wheel_moment        1
#                 (0.01,1e4), #friction_lim      2
#                 (10,100), #wheel_rad         3
#                 (10,100), #wheel_width       4
#                 # (4), #drive_train             5
#                 (5,300), #bumper_width1      5
#                 (5,300), #bumper_width2      6
#                 (0.1,2), #bumper_density    7
#                 (10,250), #hull2_width1      8
#                 (10,250), #hull2_width2      9
#                 (0.1,2), #hull2_density     10
#                 (10,250), #hull3_width1      11
#                 (10,250), #hull3_width2      12
#                 (10,250), #hull3_width3      13
#                 (10,250), #hull3_width4      14
#                 (0.1,2), #hull3_density     15
#                 (5,300), #spoiler_width1     16
#                 (5,300), #spoiler_width2     17
#                 (0.1,2), #spoiler_density   18
#                 (0.0,2), #steering_scalar  19
#                 (0.0,2), #rear_steering_scalar 20
#                 (0.0,2), #brake_scalar 21
#                 (5,200)] #max_speed 22
# feature_ranges = [(10000,600000), #eng_power          0
#                 (0.01,5), #wheel_moment        1 # mask this out!!!
#                 (100,10000), #friction_lim      2
#                 (10,80), #wheel_rad         3
#                 (5,80), #wheel_width       4
#                 # (4), #drive_train             5
#                 (5,300), #bumper_width1      5
#                 (5,300), #bumper_width2      6
#                 (0.1,2), #bumper_density    7
#                 (10,250), #hull2_width1      8
#                 (10,250), #hull2_width2      9
#                 (0.1,2), #hull2_density     10
#                 (10,250), #hull3_width1      11
#                 (10,250), #hull3_width2      12
#                 (10,250), #hull3_width3      13
#                 (10,250), #hull3_width4      14
#                 (0.1,2), #hull3_density     15
#                 (5,300), #spoiler_width1     16
#                 (5,300), #spoiler_width2     17
#                 (0.1,2), #spoiler_density   18
#                 (0.0,2), #steering_scalar  19
#                 (0.0,2), #rear_steering_scalar 20
#                 (0.0,2), #brake_scalar 21
#                 (5,200), #max_speed 22
#                 (0,16777215)] #color -- you need to set this up!!!!!!



# # feature_types = [float, float, float, int, int, int, int, float, int, int, float, int, int, int, int, float, int, int, float, float, float, float, int]
# feature_labels = ['eng_power','wheel_moment','friction_lim','wheel_rad','wheel_width',
# 'bumper_width1','bumper_width2','bumper_density','hull2_width1','hull2_width2','hull2_density',
# 'hull3_width1','hull3_width2','hull3_width3','hull3_width4','hull3_density',
# 'spoiler_width1','spoiler_width2','spoiler_density','steering_scalar','rear_steering_scalar','brake_scalar','max_speed']

# # assert len(feature_types) == len(feature_labels)

# def pack(config):

#     '''
#     "eng_power": eng_power,
#     "wheel_moment": wheel_moment,
#     "friction_lim": friction_lim,
#     "wheel_rad": wheel_rad,
#     "wheel_width": wheel_width,
#     "wheel_pos": wheel_pos,
#     "hull_poly1": hull_poly1,
#     "hull_poly2": hull_poly2,
#     "hull_poly3": hull_poly3,
#     "hull_poly4": hull_poly4,
#     "drive_train": drive_train,
#     "hull_densities": hull_densities
#     '''
#     packed_config = {}
#     packed_config['eng_power'] = config[0]
#     packed_config['wheel_moment'] = config[1]
#     packed_config['friction_lim'] = config[2]
#     packed_config['wheel_rad'] = config[3]
#     packed_config['wheel_width'] = config[4]
#     packed_config['drive_train'] = [0,0,1,1]
#     packed_config['bumper'] = {'w1':config[5], 'w2':config[6], 'd':config[7]}
#     packed_config['hull_poly2'] = {'w1':config[8],'w2':config[9],'d':config[10]}
#     packed_config['hull_poly3'] = {'w1':config[11],'w2':config[12],'w3':config[13],'w4':config[14],'d':config[15]}
#     packed_config['spoiler'] = {'w1':config[16], 'w2':config[17], 'd':config[18]}
#     packed_config['steering_scalar'] = config[19]
#     packed_config['rear_steering_scalar'] = config[20]
#     packed_config['brake_scalar'] = config[21]
#     packed_config['max_speed'] = config[22]
#     packed_config['color'] = config[23] #CONVERT THIS!!!!!!!
#     ;;;;
#     return packed_config

# def unpack(config):
#     '''
#         return an array version of an unparsed config
#     '''
#     unpacked_config = [0 for i in range(23)]
#     unpacked_config[0] = config['eng_power']    
#     unpacked_config[1] = config['wheel_moment']
#     unpacked_config[2] = config['friction_lim']
#     unpacked_config[3] = config['wheel_rad']
#     unpacked_config[4] = config['wheel_width']
#     # unpacked_config[5] = config['drive_train']
#     unpacked_config[5] = config['bumper']['w1']
#     unpacked_config[6] = config['bumper']['w2']
#     unpacked_config[7] = config['bumper']['d']
#     unpacked_config[8] = config['hull_poly2']['w1']
#     unpacked_config[9] = config['hull_poly2']['w2']
#     unpacked_config[10] = config['hull_poly2']['d']
#     unpacked_config[11] = config['hull_poly3']['w1']
#     unpacked_config[12] = config['hull_poly3']['w2']
#     unpacked_config[13] = config['hull_poly3']['w3']
#     unpacked_config[14] = config['hull_poly3']['w4']
#     unpacked_config[15] = config['hull_poly3']['d']
#     unpacked_config[16] = config['spoiler']['w1']
#     unpacked_config[17] = config['spoiler']['w2']
#     unpacked_config[18] = config['spoiler']['d']
#     unpacked_config[19] = config['steering_scalar']
#     unpacked_config[20] = config['rear_steering_scalar']
#     unpacked_config[21] = config['brake_scalar']
#     unpacked_config[22] = config['max_speed']
#     return unpacked_config
#     # return [self.problem.types[i].encode(unpacked_config[i]) for i in range(self.problem.nvars)]

# def parse_config(config):
#     l1 = 0
#     l2 = 0
#     spoiler_d = 35 #TODO: check that these match the poly's on the baseline car
#     bumper_d = 25
#     densities =[0,0,0,0]
#     # for reference, from the baseline car: 
#     # bumper is hull_poly1, spoiler is hull_poly4
#     # hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)]
#     # hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)]
#     # hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)]
#     # hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)]
#     bumper_y1 = 130
#     bumper_y2 = 110
#     h2_y1 = 120
#     h2_y2 = 20
#     h3_y1 = 20
#     h3_y2 = -10
#     h3_y3 = -40
#     h3_y4 = -90
#     spoiler_y1 = -120
#     spoiler_y2 = -90
#     if(config == {}):
#         return config
#     else:
#         if("bumper" in config.keys()):
#             bumper = config["bumper"]
#             config["hull_poly1"] = [(-bumper['w1']/2, bumper_y1),(bumper["w1"]/2, bumper_y1),( bumper["w2"]/2, bumper_y2),(-bumper["w2"]/2, bumper_y2)]
#             densities[0] = bumper["d"]
#             del config["bumper"]

#         if("hull_poly2" in config.keys()):
#             hull2 = config["hull_poly2"]
#             config["hull_poly2"] = [(-hull2['w1']/2, h2_y1),(hull2['w1']/2, h2_y1),(hull2['w2']/2, h2_y2),(-hull2['w2']/2, h2_y2)]
#             densities[1] = hull2['d']

#         if("hull_poly3" in config.keys()):
#             hull3 = config["hull_poly3"]
#             config["hull_poly3"] = [(hull3['w1']/2, h3_y1),(hull3['w2']/2, h3_y2),(hull3['w3']/2, h3_y3),(hull3['w4']/2, h3_y4), \
#                                     (-hull3['w4']/2, h3_y4),(-hull3['w3']/2, h3_y3),(-hull3['w2']/2, h3_y2),(-hull3['w1']/2, h3_y1)]
#             densities[2] = hull3['d']

#         if("spoiler" in config.keys()):
#             spoiler = config["spoiler"]
#             config["hull_poly4"] = [(-spoiler['w1']/2, spoiler_y1),(spoiler['w1']/2, spoiler_y1),(spoiler['w2']/2, spoiler_y2),(-spoiler['w2']/2, spoiler_y2)]
#             densities[3] = spoiler['d']
#             del config["spoiler"]

#         config["hull_densities"] = densities
#         config["wheel_pos"] = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)]
#     return config

# def unparse_config(config):
#     '''
#         translate a config in the carracing format back into the ga format (packed)
#         This returns a dict, for the GA we need an array.
#         to get all the way to the ga format, call unpack(unparse_config(config))
#         This is reverse-engineered from parse_config
#     '''
#     if(config == {}):
#         return config
#     else:
#         if("hull_poly1" in config.keys()):
#             coords = config["hull_poly1"]
#             config['bumper'] = {'w1':coords[1][0]*2, 'w2':coords[2][0]*2, 'd':config['hull_densities'][0]}
#         if("hull_poly2" in config.keys()):
#             coords = config["hull_poly2"]
#             config['hull_poly2'] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2,'d':config['hull_densities'][1]}
#         if("hull_poly3" in config.keys()):
#             coords = config["hull_poly3"]
#             config['hull_poly3'] = {'w1':coords[0][0]*2,'w2':coords[1][0]*2,'w3':coords[2][0]*2,'w4':coords[3][0]*2,'d':config['hull_densities'][2]}
#         if("hull_poly4" in config.keys()):
#             coords = config["hull_poly4"]
#             config["spoiler"] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2, 'd':config['hull_densities'][3]}
#         del config['hull_poly1']
#         del config['hull_poly4']
#     return config

# def config2car(config):
#     # converts a vector car config into a json
#     config = [float(c) for c in config]
#     return parse_config(pack(config))

# def car2config(car):
#     # global feature_types
#     # converts a json car config into a vector
#     return unpack(unparse_config(car))
#     # return [feature_types[i](val) for i, val in enumerate(unpack(unparse_config(car)))]

display = Display(visible=0, size=(1400,900))
display.start()

def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)

# first connect to pymongo and get all the sessions that have a complete status
client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
sessions = db.sessions

if session_override_id is not None:
    session = sessions.find_one({"user_id": user_id, "_id": ObjectId(session_override_id)},sort=[("_id",pymongo.ASCENDING)])
else:
    session = sessions.find_one({"user_id": user_id,"status":"complete"},sort=[("_id",pymongo.ASCENDING)])
# get the agent to optimize for
agent_full_path = session['agent']

base_config = car2config(session['initial_design'])

new_session_dir = os.path.join(Config.FILE_PATH,user_id,"bo_sessions_all_features",str(session["_id"])) # session_id is an ObjectID, not str
train_dir = os.path.join(new_session_dir,'policy_training')
os.makedirs(train_dir)
shutil_copy(agent_full_path,new_session_dir)
agent_file = str(os.path.join(new_session_dir,os.path.basename(agent_full_path)))

num_episodes = session['n_training_episodes']

# initialize a driver for now
driver = DQNAgent(num_episodes, agent_file, base_config, train_dir=train_dir) 

user_config = car2config(session['final_design'])
mask_indices = []
# for attribute,value in base_config.items():
#     if user_config[attribute] != value:
#         try:
#             mask_indices.append(feature_labels.index(attribute))
#         except ValueError as e:
#             mask_indices = []
#             raise(e)
for index,value in enumerate(base_config):
    if index not in densities: #user_config[index] != value:
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
    res = gp_minimize(test_drive, masked_designbounds, acq_func=acq_func, n_calls=iters, x0=x0_masked, y0=y0, n_random_starts=5, random_state=seed, n_jobs=8, kappa=kappa)
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



# do a test drive with the base config, x0, to get an initial reward value, y0
x0 = [base_config]
init_result = test_drive(base_config)
y0 = np.array([init_result])
results = design_step(x0,y0,acq_func='LCB',iters=250) #,acq_func=acq_func,kappa=4)
best_config = results.x
print(f'Best design reward: {results.fun}')
# if x0 is None:
x0 = results.x_iters
y0 = results.func_vals
print(f'x0:{len(x0)},y0:{y0.shape}')
with open(f'{train_dir}/designs.pkl','wb+') as design_dump:
    pkl.dump([x0,y0],design_dump) # these designs should be in order, check y0 for rewards

# now retrain with the best design and see how it goes
rewards = policy_step(config2car(best_config),num_episodes)
        



