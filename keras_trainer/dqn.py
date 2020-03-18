import gym
import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, plot_model
from keras.models import load_model
from keras import backend as K
from pprint import pprint
import cv2
import ringbuffer

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  fname = os.path.join(os.getcwd(), "dqn_1000_running_avg.png")
  plt.savefig(fname)


def transform(s):
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)
    
    # upper_field = observation[:84, :96] # this is the section of the screen that contains the track.
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
#     cv2.imshow('video', upper_field_bw)
#     cv2.waitKey(1)
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

vector_size = 10*10 + 7 + 4


def create_nn(model_to_load):
    try:
        m = load_model(model_to_load)
        # K.set_value(m.optimizer.lr, 0.01) # set a higher LR for retraining
        print("Loaded pretrained model " + model_to_load)
        init_weights = m.get_weights()
        return m, init_weights
    except FileNotFoundError:
        print("Creating new network")
        model = Sequential()
    # 4 frames vertically concatenated
        model.add(Dense(512, input_shape=(4*111,), kernel_initializer="lecun_uniform"))# 7x7 + 3.  or 14x14 + 3 # a
        model.add(Activation('relu'))

        model.add(Dense(11, kernel_initializer="lecun_uniform"))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        adamax = Adamax() #Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        model.summary()
        
        return model

class DQNAgent():
    def __init__(self, num_episodes, model_name=None, carConfig=None):
        env = gym.make('CarRacing-v1')
        env = wrappers.Monitor(env, 'monitor-folder', force=True)
        self.carConfig = carConfig
        self.env = env
        self.gamma = 0.99
        self.model_name = model_name
        self.model, self.init_weights = create_nn(model_name)  # 4 consecutive steps, 111-element vector for each state
        self.model.summary()
        if not model_name:
            MEMORY_SIZE = 10000
        else:
            MEMORY_SIZE = 1000  # smaller memory for retraining
        self.memory = ringbuffer.RingBuffer(MEMORY_SIZE)
        self.num_episodes = num_episodes

    def predict(self, s):
        # print("shape for pred is ", np.reshape(s, (4*111,)).shape)
        return self.model.predict(np.reshape(s, (1, 4*111)), verbose=0)[0]

    def update(self, s, G, B):
        self.model.fit(s, np.array(G).reshape(-1, 11), batch_size=B, epochs=1, use_multiprocessing=True, verbose=0)

    def sample_action(self, s, eps):
        # print('THE SHAPE FOR PREDICTION: ', s.shape)
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0, 10), qval
        else:
            return np.argmax(qval), qval

    def convert_argmax_qval_to_env_action(self, output_value):
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

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        old_states = []
        old_state_preds = []
        for (old_state, argmax_qval, reward, next_state) in batch:
            next_state_pred = self.predict(next_state)
            max_next_pred = np.max(next_state_pred)
            old_state_pred = self.predict(old_state)
            target_q_value = reward + self.gamma * max_next_pred
            y = old_state_pred[:]
            y[argmax_qval] = target_q_value
            old_states.append(old_state)
            old_state_preds.append(y.reshape(1, 11))
        old_states = np.reshape(old_states, (batch_size, 111*4))
        old_state_preds = np.array(old_state_preds).reshape(batch_size, 11)
        self.model.fit(old_states, old_state_preds, batch_size=batch_size, epochs=1, verbose=0)



    def play_one(self, eps):
        if self.carConfig:
            print("TRAINING WITH CAR CONFIG: ")
            print(self.carConfig)
            observation = self.env.reset(self.carConfig)
        else: 
            observation = self.env.reset()
        done = False
        full_reward_received = False
        totalreward = 0
        iters = 0
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        stacked_state = np.array([state,state,state, state])
        # print('state shape: ', stacked_state.shape)
        while not done:
            # a, b, c = transform(observation)
            # state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            # print("STATE SHAPE: ", stacked_state.shape)
            argmax_qval, qval = self.sample_action(stacked_state, eps)
            prev_state = stacked_state
            action = self.convert_argmax_qval_to_env_action(argmax_qval)
            observation, reward, done, info = self.env.step(action)

            a, b, c = transform(observation)        
            curr_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            stacked_state = np.append(stacked_state[1:], [curr_state], axis=0) # appending the lastest frame, pop the oldest
            # print('state shape: ', stacked_state.shape)
            # add to memory
            self.memory.append((prev_state, argmax_qval, reward, stacked_state))
            # replay batch from memory every 20 steps
            REPLAY_FREQ = 32
            if iters % REPLAY_FREQ==0 and iters>10:
                try:
                    self.replay(32)
                except Exception as e: # will error if the memory size not enough for minibatch yet
                    print("error when replaying: ", e)
                    raise e

            totalreward += reward
            iters += 1
            
            if iters > 1500:
                print("This episode is stuck")
                break
        # for i in range(20):
        #     self.replay(100)
        return totalreward, iters

    def train(self, retrain=False):
        totalrewards = np.empty(self.num_episodes)
        for n in range(self.num_episodes):
            print("training ", str(n))
            eps = 0.5/np.sqrt(n + 1000)
            totalreward, iters = self.play_one(eps)
            totalrewards[n] = totalreward
            print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
            if n>0 and n%50==0 and not self.model_name:
                # save model
                trained_model = os.path.join(os.getcwd(),"dqn_trained_model_{}.h5".format(str(n)))
                self.model.model.save(trained_model)
        if self.model_name:
            print('saving: ', self.model_name)
            self.model.save(self.model_name)
        # print(self.model)
        # print("INITIAL WEIGHTS: ")
        # pprint(self.init_weights)
        # print("FINAL WEIGHTS: ")
        # pprint(self.model.get_weights())
        # print("DIFFERENCE")
        # weight_diff = abs(np.array(self.model.get_weights())-np.array(self.init_weights))
        # # pprint(weight_diff)
        # print("PERCENTAGE CHANGE layer 1: ")
        # layer1 = weight_diff[0]/self.init_weights[0] * 100
        # # pprint(layer1)
        # print("MAX of layer: ", np.max(layer1))
        # print("mean of layer: ", np.mean(layer1))

        # print("PERCENTAGE CHANGE layer 2: ")
        # layer2 = weight_diff[1]/self.init_weights[1] * 100
        # # pprint(layer2)
        # print("MAX of layer: ", np.max(layer2))
        # print("mean of layer: ", np.mean(layer2))

        # print("PERCENTAGE CHANGE layer 3: ")
        # layer3 = weight_diff[2]/self.init_weights[2] * 100
        # # pprint(layer3)
        # print("MAX of layer: ", np.max(layer3))
        # print("mean of layer: ", np.mean(layer3))


        # print("PERCENTAGE CHANGE layer 4: ")
        # layer4 = weight_diff[3]/self.init_weights[3] * 100
        # # pprint(layer4)
        # print("MAX of layer: ", np.max(layer4))
        # print("mean of layer: ", np.mean(layer4))


        if not self.model_name:
            plt.plot(totalrewards)
            rp_name = os.path.join(os.getcwd(), "dqn_1000_rewards.png")
            plt.title("Rewards")
            plt.savefig(rp_name)
            plt.close()
            plot_running_avg(totalrewards)
        self.env.close()

if __name__ == "__main__":
    trainer = DQNAgent(1000)
    trainer.train()
