import sys, traceback
import random
sys.path.append('/home/dev/scratch/cars/carracing_clean')

from keras_trainer.avg_dqn import DQNAgent
from pyvirtualdisplay import Display
from copy import deepcopy
import time
import json

driver = DQNAgent(num_episodes=1, model_name='/home/dev/scratch/cars/carracing_clean/agents/pretrained_drivers/avg_dqn_ep_200.h5')
carfile = '/home/dev/scratch/cars/carracing_clean/agents/nice_200_car.json'
with open(carfile, 'r') as infile:
    car = json.load(infile)

rewards = []
# driver.carConfig = car
for i in range(10):
    total_reward,_ = driver.play_one(eps=0.0,train=False)
    rewards.append(total_reward)
print(rewards)
# with open('/home/dev/scratch/cars/carracing_clean/agents/test_drives_normal_200_car.json', 'w+') as outfile:
#     json.dump(rewards,outfile)