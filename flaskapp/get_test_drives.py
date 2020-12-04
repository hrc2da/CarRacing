import json
import sys
sys.path.append('/home/dev/scratch/cars/carracing_clean')
from keras_trainer.avg_dqn_updated_gym import DQNAgent
from config import Config
import gym
from gym import wrappers
from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400,900))
display.start()

def reset_driver_env(driver, vid_path):
    env = gym.make('CarRacingTrain-v1')
    driver.env = wrappers.Monitor(env, vid_path, force=False, resume = True, video_callable=lambda x : True, mode='evaluation', write_upon_reset=False)

agent_file = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/joe_dqn_only_2_0902_2232_avg_dqn_ep_0.h5'
with open('/home/dev/scratch/cars/carracing_clean/flaskapp/static/default/car_config.json','r') as infile:
    car_config = json.load(infile)
driver = DQNAgent(1, agent_file, car_config, replay_freq=50, lr=0.001)
videos = []
for i in range(Config.N_TEST_DRIVES):
    video_filename = f'test_drive_{i}.mp4'
    
    vid_dir = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/default2/' #.replace(f'{Config.FILE_PATH}/','')
    reset_driver_env(driver,vid_dir)
    result = driver.play_one(train=False,video_path=video_filename,eps=0.01)
    driver.env.close()