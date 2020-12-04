import pickle as pkl
import json
import os
import sys
import glob
from utils import config2car, car2config
from shutil import copy as shutil_copy
sys.path.append('/home/dev/scratch/cars/carracing_clean')
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from pyvirtualdisplay import Display

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/bo_sessions/5f8df42ad5c68de5e8185f59/policy_training'
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/bo_sessions_all_features/5f8df42ad5c68de5e8185f59/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1102_2157/*car_config.json')[0]) #jihyun all_features done
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/bo_sessions_start_from_redesign/5f8df42ad5c68de5e8185f59/30_step_design_policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_0938/*car_config.json')[0]) #jihyun hybrid start from redesign

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/2a532da3d761421890cc5de28b3ff2f3/bo_sessions/5f90983b8dfbb43775149641/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1021_1946/*car_config.json')[0]) #anna 
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/2a532da3d761421890cc5de28b3ff2f3/bo_sessions_start_from_redesign/5f90983b8dfbb43775149641/30_step_design_policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_0938/*car_config.json')[0]) #anna hybrid start from redesign

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/bo_sessions/5f90fdee8dfbb43775149642/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1022_0356/*car_config.json')[0]) #nikhil
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/bo_sessions_all_features/5f90fdee8dfbb43775149642/policy_training_0'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1103_0033/*car_config.json')[0]) #nikhil 250 bo done
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/bo_sessions_all_features/5f90fdee8dfbb43775149642/policy_training_1_30stepbo'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1102_1932/*car_config.json')[0]) #nikhil 30 bo done
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/bo_sessions_start_from_redesign/5f90fdee8dfbb43775149642/30_step_design_policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_0915/*car_config.json')[0]) #nikhil 30 bo # hybrid start from redesign

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/9a5f5d937d79438daa2b52cb4ce26216/bo_sessions/5f87bcceb9d252ffa7e54beb/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1015_0252/*car_config.json')[0]) #yuhan
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/9a5f5d937d79438daa2b52cb4ce26216/bo_sessions_all_features/5f87bcceb9d252ffa7e54beb/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1021_1511/*car_config.json')[0]) #yuhan done
# skipping start from redesign because illegal for bo


# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b0e35b9e8db847d992fa81afa8851753/bo_sessions/5f87a26bb9d252ffa7e54bea/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1015_0145/*car_config.json')[0]) #swati
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b0e35b9e8db847d992fa81afa8851753/bo_sessions_all_features/5f87a26bb9d252ffa7e54bea/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1021_0323/*car_config.json')[0]) #swati done
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b0e35b9e8db847d992fa81afa8851753/bo_sessions_start_from_redesign/5f87a26bb9d252ffa7e54bea/30_step_design_policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_1002/*car_config.json')[0]) #swati hybrid start from redesign



# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b15a47a3828c43d79fa74ca0cffdeb53/bo_sessions/5f875228b9d252ffa7e54be9/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1014_2025/*car_config.json')[0]) #alap
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b15a47a3828c43d79fa74ca0cffdeb53/bo_sessions_all_features/5f875228b9d252ffa7e54be9/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1020_0237/*car_config.json')[0]) #alap done
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/b15a47a3828c43d79fa74ca0cffdeb53/bo_sessions_start_from_redesign/5f875228b9d252ffa7e54be9/30_step_design_policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_1002/*car_config.json')[0]) #alap hybrid start from redesign


# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/e6900ed30d77497a97b8b9800d3becdf/bo_sessions/5f6cd5e2d8a6d9430d007bf3/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1014_2032/*car_config.json')[0]) #dan
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/e6900ed30d77497a97b8b9800d3becdf/bo_sessions_all_features/5f6cd5e2d8a6d9430d007bf3/policy_training_30stepbo'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1103_1821/*car_config.json')[0]) #dan 30 bo

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions/5f6d02c2673446edf0d88f1c/policy_training'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1001_0528/*car_config.json')[0]) #amit
# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions_all_features/5f6d02c2673446edf0d88f1c/policy_training_30stepb0'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1103_1816/*car_config.json')[0]) #amit 30 bo
root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions_start_from_redesign/5f6d02c2673446edf0d88f1c/30_step_design_policy_training'
design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/1109_1004/*car_config.json')[0]) #amit 30 hybrid start from redesign

# root_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default_design_for_experiments'
# design_json = os.path.join(root_dir, glob.glob(f'{root_dir}/car_config.json')[0]) #Baseline!

# pull the final_design from an experiment
# specify user_id
# specify experiment_type
# specify which one if there's more than one

# add the results to trial_paths
# replace trial_rewards with [trial_rewards] and append the new trial rewards
# ^^ consider scripting this and refactoring the code to add it as a list of lists




design_pkl = os.path.join(root_dir,'designs.pkl')

default_agent_path = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_1_0902_2232_avg_dqn_ep_0.h5'
num_episodes = 250

display = Display(visible=0, size=(1400,900))
display.start()


with open(design_json, 'r') as jsonfile:
    car_config = json.load(jsonfile)

# setup the output paths for agent and rewards pickle
train_dir = os.path.join(root_dir,'extra_training_run_with_this_design_0')
# if 
counter = 1
while(os.path.isdir(train_dir)):
    train_dir = os.path.join(root_dir,f'extra_training_run_with_this_design_{counter}')
    counter += 1
# make a copy of the default agent to this path
agent_file = 'copy of the default agent in this path'
os.makedirs(train_dir)
shutil_copy(default_agent_path,train_dir)
agent_file = str(os.path.join(train_dir,os.path.basename(default_agent_path)))
# run X episodes
driver = DQNAgent(num_episodes, agent_file, car_config, replay_freq=50, lr=0.001, train_dir = str(train_dir))
driver.train()
# dump agent and rewards






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