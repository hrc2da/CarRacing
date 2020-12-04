import sys
import os
import uuid
from shutil import copy
sys.path.append('/home/dev/scratch/cars/carracing_clean')
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from config import Config
from pyvirtualdisplay import Display
from pymongo import MongoClient
import gym
from gym import wrappers

from bson.objectid import ObjectId

from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import glob

display = Display(visible=0, size=(1400,900))
display.start()

# user_id = 'a3476ba7c278432db0315eda9546b7a4' #amit
# session_override_id = "5f6d02c2673446edf0d88f1c" # first 250 session
# user_id = 'e6900ed30d77497a97b8b9800d3becdf' #dan
# session_override = '5f6cd5e2d8a6d9430d007bf3' #dan

# user_id = 'b15a47a3828c43d79fa74ca0cffdeb53' #alap
# session_override = '5f875228b9d252ffa7e54be9' #alap

# user_id = 'b0e35b9e8db847d992fa81afa8851753' #swati
# session_override = '5f87a26bb9d252ffa7e54bea' #swati

# user_id = '9a5f5d937d79438daa2b52cb4ce26216' #yuhan
# session_override = '5f87bcceb9d252ffa7e54beb' #yuhan


# session_override = False #"5f8df42ad5c68de5e8185f59"  #False
# session_override = "5f90fdee8dfbb43775149642" #nikhil
# session_override = "5f90983b8dfbb43775149641" #anna
# session_override = "5f8df42ad5c68de5e8185f59" #jihyun

# user_id = 'a3476ba7c278432db0315eda9546b7a4' #amit
# session_override = "5f6d02c2673446edf0d88f1c" #amit

def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)

# first connect to pymongo and get all the sessions that have a pending status
client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
sessions = db.sessions


# check the "current_jobs" record and see if any of the jobs set to pending haven't been started
# choose the first one, add it to the list of running jobs, make the right directories, and start it
jobs = db.jobs
job_id = None

if session_override:
  session_id = session_override
  session = sessions.find_one({"_id":ObjectId(session_id)})
  
else:
  for session in sessions.find({"status":"pending"}):
      job = jobs.find_one({"session":session["_id"]})
      if job is None:
          session_id = session["_id"]
          job_id = jobs.insert_one({"session":session_id,"status":"running"}).inserted_id
          break


  if job_id is None:
      print("No pending jobs to run!")
      sys.exit()

  job = jobs.find_one({"session":session_id})
  # a bit of paraonoia
  assert job is not None
  assert job['_id'] == job_id
  assert job['session'] == session_id
  assert job['session'] == session['_id']




# ok now get the stuff we need from the session
old_agent_file = session['agent']
car_config = session['final_design']
user_id = session['user_id']
try:
    num_episodes = session['n_training_episodes']
except KeyError:
    num_episodes = Config.N_TRAINING_EPISODES # fall back to config if not in db
# setup the directories
if session_override:
  print("overriding")
  new_session_id = f"session_{session_override}_human_design_retrain"
else:
  new_session_id = sessions.insert_one({ "user_id":user_id,
                    "status":"pre_design",
                    "tested_designs":[], 
                    "tested_videos":[],
                    "tested_results":[],
                    "initial_design": car_config,
                    "final_design": {},
                    "question_answers": {},
                    "initial_test_drives":[],
                    "n_training_episodes":num_episodes}).inserted_id

new_session_dir = os.path.join(Config.FILE_PATH,user_id,str(new_session_id)) # session_id is an ObjectID, not str
train_dir = os.path.join(new_session_dir,'training')
if session_override:
  counter = 0
  while(os.path.isdir(train_dir)):
    train_dir = os.path.join(new_session_dir,f'training_{counter}')
    counter += 1
os.makedirs(train_dir)
copy(old_agent_file,train_dir)
agent_file = str(os.path.join(train_dir,os.path.basename(old_agent_file)))

# only update the session if this is the first time running, subsequent runs are just for data collection

if session_override == False:
  sessions.update_one({"_id":new_session_id},{'$set':{'file_path':new_session_dir, 'agent':agent_file}})

driver = DQNAgent(num_episodes, agent_file, car_config, replay_freq=50, lr=0.001, train_dir = str(train_dir))
driver.train()



# when finished, set the job to complete and the status to complete
if session_override == False:
  sessions.update_one({'_id':session_id},{'$set':{'status':'complete'}})
  jobs.update_one({'_id':job_id},{'$set':{'status':'complete'}})


  # run ten test drives with the latest agent
  videos = []
  for i in range(Config.N_TEST_DRIVES):
      video_filename = uuid.uuid4().hex+'.mp4'
      video_path = str(os.path.join(new_session_dir,video_filename)).replace(f'{Config.FILE_PATH}/','')
      vid_dir = new_session_dir #.replace(f'{Config.FILE_PATH}/','')
      reset_driver_env(driver,vid_dir)
      result = driver.play_one(train=False,video_path=video_filename,eps=0.01)
      driver.env.close()
      videos.append(video_path)

  sessions.update_one({"_id":new_session_id},{'$set':{'initial_test_drives':videos,'status':'designing'}})

  # plot cumulative progress
  # reward_files = glob.glob(f'{Config.FILE_PATH}/{user_id}/*/training/*/total_rewards.pkl')
  reward_files = glob.glob(f'{Config.FILE_PATH}/{user_id}/*/training/*/*_rewards*.pkl')
  reward_files.sort()

  rewards = []
  for f in reward_files:
    with open(f, 'rb') as infile:
      data = pkl.load(infile)
      for d in data:
        rewards.append(d)

  def running_avg(totalrewards):
    # return totalrewards
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
    return running_avg


  plt.plot(running_avg(rewards))
  # plt.text(20,430,"Driver's Cumulative Performance While Learning to Drive", fontsize=15)
  plt.xlabel('Training Episode')
  plt.ylabel('Episode Score (Higher is Better)')
  plt.ylim(-100,600)
  for i in (250,500,750,1000):
    plt.axvline(i,color='0.5',linestyle='--')
  # plt.text(150,410,'First Redesign')
  # plt.text(380,410,'Second Redesign')
  # plt.text(650,410,'Third Redesign')
  plt.savefig(os.path.join(new_session_dir,'reward_plot.png'), bbox_inches='tight')
  reward_plot_path = f'{user_id}/{session_id}/reward_plot.png'
  sessions.update_one({"_id":new_session_id},{'$set':{'reward_plot':reward_plot_path}})