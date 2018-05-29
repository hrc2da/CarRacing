import gym
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import gym.spaces
import gym.wrappers as wrappers
from gym.wrappers.monitoring import stats_recorder, video_recorder
#from gym import wrappers
from datetime import datetime
import random
from pyvirtualdisplay import Display
#from xvfbwrapper import Xvfb
from sklearn.preprocessing import StandardScaler
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Merge
from keras.utils import np_utils
from keras.models import load_model
import os
from socketIO_client import SocketIO

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["DISPLAY"] = ':1'
orig = os.environ["DISPLAY"]

def transform(s):
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)

    # upper_field = observation[:84, :96] # this is the section of the screen that contains the track.
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 10x10 pixels
    upper_field_bw = upper_field_bw.astype('float')/255

    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]

#     print(car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255)
    car_field_t = [car_field_bw[:, 3].mean()/255, car_field_bw[:, 4].mean()/255, car_field_bw[:, 5].mean()/255, car_field_bw[:, 6].mean()/255]


    return bottom_black_bar_bw, upper_field_bw, car_field_t

# this function uses the bottom black bar of the screen and extracts steering setting, speed and gyro data
def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean()/255
    left_steering = a[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2

    left_gyro = a[6, 46:60].mean()/255
    right_gyro = a[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2

    speed = a[:, 0][:-2].mean()/255
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255


    return [steering, speed, gyro, abs1, abs2, abs3, abs4]


def create_nn():
    if os.path.exists('race-car.h5'):
        return load_model('race-car.h5')

    model = Sequential()
    model.add(Dense(512, kernel_initializer='lecun_uniform', input_shape=(111,)))# 7x7 + 3.  or 14x14 + 3
    model.add(Activation('relu'))

#     model.add(Dense(512, init='lecun_uniform'))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.3))

    model.add(Dense(11, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

#     rms = RMSprop(lr=0.005)
#     sgd = SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
# try "adam"
#     adam = Adam(lr=0.0005)
    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    #model.summary()

    return model


class Model:
	def __init__(self, env):
		self.env = env
		self.model = create_nn()  # one feedforward nn for all actions.

	def predict(self, s):
                print("SSSS",s)
                return self.model.predict(s.reshape(-1, 111), verbose=0)[0]

	def update(self, s, G):
		self.model.fit(s.reshape(-1, 111), np.array(G).reshape(-1, 11), nb_epoch=1, verbose=0)

	def sample_action(self, s, eps):
		qval = self.predict(s)
		if np.random.random() < eps:
			return random.randint(0, 10), qval
		else:
			return np.argmax(qval), qval


def convert_argmax_qval_to_env_action(output_value):
    # we reduce the action space to 15 values.  9 for steering, 6 for gaz/brake.
    # to reduce the action space, gaz and brake cannot be applied at the same time.
    # as well, steering input and gaz/brake cannot be applied at the same time.
    # similarly to real life drive, you brake/accelerate in straight line, you coast while sterring.

    gaz = 0.0
    brake = 0.0
    steering = 0.0

    # output value ranges from 0 to 10

    if output_value <= 8:
        # steering. brake and gaz are zero.
        output_value -= 4
        steering = float(output_value)/4
    elif output_value >= 9 and output_value <= 9:
        output_value -= 8
        gaz = float(output_value)/3 # 33%
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2 # 50% brakes
    else:
        print("error")

    white = np.ones((round(brake * 100), 10))
    black = np.zeros((round(100 - brake * 100), 10))
    brake_display = np.concatenate((black, white))*255

    white = np.ones((round(gaz * 100), 10))
    black = np.zeros((round(100 - gaz * 100), 10))
    gaz_display = np.concatenate((black, white))*255

    control_display = np.concatenate((brake_display, gaz_display), axis=1)

    # cv2.imshow('controls', control_display)
    # cv2.waitKey(1)

    return [steering, gaz, brake]



def reset_video_recorder_filename(filename,env):
    if env.video_recorder:
        env._close_video_recorder()
    env.video_recorder = video_recorder.VideoRecorder(
                env=env,
                path=os.path.join(env.directory,filename),
                metadata={'episode_id': env.episode_id},
                enabled=env._video_enabled(),
            )
    env.video_recorder.capture_frame()
            # Close any existing video recorder
            #         if self.video_recorder:
            #                     self._close_video_recorder()
            #
            #                             # Start recording the next video.
            #                                     #
            #                                             # TODO: calculate a more correct 'episode_id' upon merge
            #                                                     self.video_recorder = video_recorder.VideoRecorder(
            #                                                                 env=self.env,
            #                                                                             base_path=os.path.join(self.directory, '{}.video.{}.video{:06}'.format(self.file_prefix, self.file_infix, self.episode_id)),
            #                                                                                         metadata={'episode_id': self.episode_id},
            #                                                                                                     enabled=self._video_enabled(),
            #                                                                                                             )
            #                                                                                                                     self.video_recorder.capture_frame()



def play_one(env,model, eps, gamma, config,path=None,display=None):
    print("about to reset env")
    observation = env.reset(config,display)
    print("Made it past env reset with path:",path)
    if(path):
        reset_video_recorder_filename(path,env)
    #env.build_car(config)
    print("Made it past video recorder reset")
    done = False
    full_reward_received = False
    totalreward = 0
    totalfuelconsumed = 0
    totalgrasstraveled = 0
    iters = 0
    while not done:
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1
        argmax_qval, qval = model.sample_action(state, eps)
        # prev_state = state
        action = convert_argmax_qval_to_env_action(argmax_qval)
        # env.render()
        observation, reward, done, info = env.step(action)

        # a, b, c = transform(observation)
        # state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1

        # update the model
        # standard Q learning TD(0)
        # next_qval = model.predict(state)
        # G = reward + gamma*np.max(next_qval)
        # y = qval[:]
        # y[argmax_qval] = G
        # model.update(prev_state, y)
        totalreward += reward
        totalfuelconsumed += info["fuel"]
        totalgrasstraveled += info["grass"]
        iters += 1

        if iters > 300:
            print("This episode is stuck")
            break
    if(env.video_recorder):
        env._close_video_recorder()
    return totalreward, iters, totalfuelconsumed, totalgrasstraveled #Matt:weird order is to not break anything else

def parse_config(config):
    l1 = 0
    l2 = 0
    spoiler_d = 35
    bumper_d = 25
    densities =[0,0,0,0]
    if(config == {}):
        return config
    else:
        if("hull_poly1" in config.keys()):
            hull1_vars = config["hull_poly1"]
            l1 = hull1_vars['l']
            config["hull_poly1"] = [(-hull1_vars['w2']/2, 0),(hull1_vars['w2']/2, 0),(-hull1_vars['w1']/2, l1),(hull1_vars['w1']/2, l1)]
            densities[0] = hull1_vars['d']

        if("hull_poly2" in config.keys()):
            hull2_vars = config["hull_poly2"]
            l2 = hull2_vars['l']
            config["hull_poly2"] = [(-hull2_vars['w1']/2, 0),(hull2_vars['w1']/2, 0),(-hull2_vars['w2']/2, -l2),(hull2_vars['w2']/2, -l2)]
            densities[1] = hull2_vars['d']

        if("spoiler" in config.keys()):
            spoiler_w = config["spoiler"]['w']
            config["hull_poly3"] = [(-spoiler_w, -l2),(spoiler_w, -l2),(-spoiler_w, -l2-spoiler_d),(spoiler_w, -l2-spoiler_d)]
            densities[2] = config["spoiler"]['d']
            del config["spoiler"]

        if("bumper" in config.keys()):
            bumper_w = config["bumper"]['w']
            config["hull_poly4"] = [(-bumper_w, l1),(bumper_w, l1),( -bumper_w, l1+bumper_d),(bumper_w, l1+bumper_d)]
            densities[3] = bumper_w = config["bumper"]['d']
            del config["bumper"]

        config["hull_densities"] = densities

    return config

def init_buffer():
    #testing os level display fix
    global orig
    #testing os level display fix
    display = Display(visible=0,size=(1400,900))
    display.start()
    #orig = os.environ["DISPLAY"]
    return display

def kill_buffer(display):
    display.sendstop()
def run(config = {},session_id=None):
    env = gym.make('CarRacing-v1')
    env = wrappers.Monitor(env, 'monitor-folder', force=False, resume=True, video_callable= False, mode='evaluation')
    vector_size = 10*10 + 7 + 4
    model = Model(env)
    eps = 0.5/np.sqrt(1 + 900)
    gamma = 0.99

    parsed_config = parse_config(config)

    totalreward, iters, fuel, grass = play_one(env, model, eps, gamma, parsed_config)
    env.close()
    with open("/share/sandbox/carracing/logged_cars.txt", "a+") as carfile:
        carfile.write(str(config)+","+str(totalreward)+","+ str(fuel)+","+str(grass)+"\n")
    with SocketIO('127.0.0.1','5000') as SIO:
        SIO.emit('evaluated_car',{"session_id":session_id,"car":{'config':config,'reward':totalreward,'fuel':fuel, 'grass':grass}})
    print("REWARD: ", totalreward, " FUEL: ", fuel, "GRASS: ",grass)
    return [totalreward, fuel, grass]
def run_vid(config = {}):
    #testing os level display fix
    global orig
    os.environ["DISPLAY"] = orig
    #testing os level display fix

    display = Display(visible=0, size=(1400,900))
    display.start()
    env = gym.make('CarRacing-v1')
    env = wrappers.Monitor(env, 'monitor-folder', force=False, resume = True, video_callable=None, mode='evaluation')

    vector_size = 10*10 + 7 + 4

    model = Model(env)
    eps = 0.5/np.sqrt(1 + 900)
    gamma = 0.99

    parsed_config = parse_config(config)
    # parsed_config = config
    totalreward, iters, fuel, grass = play_one(env, model, eps, gamma, parsed_config)
    display.sendstop()
    return [totalreward, fuel, grass]

def run_unparsed(config = {}, filename=None,display=None):
    tempdisplay = None
#testing os level display fix
    global orig
    #testing os level display fix
    if display is None:
        os.environ["DISPLAY"] = ':1'
        tempdisplay = Display(visible=0, size=(1400,900))
        tempdisplay.start()
        #tempdisplay = Xvfb(width=1400,height=900)
        #tempdisplay.start()
    env = gym.make('CarRacing-v1')
    env = wrappers.Monitor(env, 'flaskapp/static', force=False, resume = True, video_callable=None, mode='evaluation')
    model = Model(env)
    eps = 0.5/np.sqrt(1 + 900)
    gamma = 0.99

    # parsed_config = parse_config(config)
    parsed_config = config
    totalreward, iters, fuel, grass = play_one(env, model, eps, gamma, parsed_config, filename);
    env.close()
    env = None
    model = None
    if tempdisplay is not None:
        tempdisplay.sendstop()
        tempdisplay = None
        os.environ["DISPLAY"]=':1'
        print("closed display")
    return [totalreward, fuel, grass]

def run_configs_from_file(filepath):
    with open(filepath,"r") as infile:
        configs = []
        for line in infile:
            configs.append(json.loads(line))
    for car in configs:
        print(car['reward'])
        run_unparsed(car['config'])
if __name__=='__main__':
    #disp = Display(visible=0,size=(1400,900))
    #disp.start()
    #init_buffer()
    run_unparsed(filename="testrun1.mp4",display=None)
    run_unparsed(filename="testrun2.mp4",display=None)
    #kill_buffer()
    #disp.sendstop()
