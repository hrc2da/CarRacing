import gym
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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Merge
from keras.utils import np_utils, plot_model
from keras.models import load_model
from pprint import pprint
import cv2

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

env = gym.make('CarRacing-v1')
env = wrappers.Monitor(env, 'monitor-folder', force=True)
# obs = env.reset()
# gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)
# plt.show()
# print("screen size: ", gray.shape)

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
    
        
#     cv2.imshow('sensors', speed_display)
#     cv2.waitKey(1)

    
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

vector_size = 10*10 + 7 + 4


def create_nn(input_shape):

    model = Sequential()
    # add convolutional layers
    model.add(Conv2D(32, 2, strides=(4,4),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape))

        # Third convolutional layer
    model.add(Conv2D(64, 1, strides=(1,1),
                            padding='valid',
                            activation='relu',
                            input_shape=input_shape))
        # Flatten the convolution output
    model.add(Flatten())
 # a # a
    model.add(Dense(512, kernel_initializer="lecun_uniform"))# 7x7 + 3.  or 14x14 + 3 # a
    model.add(Activation('relu'))

    model.add(Dense(11, kernel_initializer="lecun_uniform"))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    adamax = Adamax() #Adamax(lr=0.001)
    model.compile(loss='mse', optimizer=adamax)
    model.summary()
    
    return model


class Model:
    def __init__(self, env):
        self.env = env
        self.model = create_nn((4, 96, 96))  # 4 consecutive steps, 111-element vector for each state

    def predict(self, s):
        # print("shape is ", np.reshape(s, (1, 4, 111)).shape)
        return self.model.predict(np.expand_dims(s, axis=0), verbose=0)[0]

    def update(self, s, G):
        self.model.fit(np.expand_dims(s, axis=0), np.array(G).reshape(-1, 11), epochs=1, verbose=0)

    def sample_action(self, s, eps):
        # print('THE SHAPE FOR PREDICTION: ', s.shape)
        qval = self.predict(s)
        if np.random.random() < eps:
            return random.randint(0, 10), qval
        else:
            return np.argmax(qval), qval


def convert_argmax_qval_to_env_action(output_value):
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

    return [steering, gaz, brake]

def play_one(env, model, eps, gamma):
    observation = env.reset()
    done = False
    full_reward_received = False
    totalreward = 0
    iters = 0
    # a, b, c = transform(observation)
    # state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
    state = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    # pprint(np.array([state,state,state, state]))
    stacked_state = np.array([state,state,state, state])

    while not done:
        # a, b, c = transform(observation)
        # state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        # print("STATE SHAPE: ", stacked_state.shape)
        argmax_qval, qval = model.sample_action(stacked_state, eps)
        prev_state = stacked_state
        action = convert_argmax_qval_to_env_action(argmax_qval)
        observation, reward, done, info = env.step(action)

        # a, b, c = transform(observation)        
        # curr_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        curr_state = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        stacked_state = np.append(stacked_state[1:], [curr_state], axis=0) # appending the lastest frame, pop the oldest
        # update the model
        # standard Q learning TD(0)
        next_qval = model.predict(stacked_state)
        G = reward + gamma*np.max(next_qval)
        y = qval[:]
        y[argmax_qval] = G
        model.update(prev_state, y)
        totalreward += reward
        iters += 1
        
        if iters > 1500:
            print("This episode is stuck")
            break
        
    return totalreward, iters

model = Model(env)
plot_model(model.model, '2D_cnn_model.png', show_shapes=True)
gamma = 0.99
trained_model = os.path.join(os.getcwd(),"cnn_train_car_test.h5")
N = 100
totalrewards = np.empty(N)
costs = np.empty(N)
for n in range(N):
    eps = 0.5/np.sqrt(n+1 + 900) 
    totalreward, iters = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward
    print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
    # if n % 10 == 0:
    #     model.model.save('race-car.h5')

print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
print("total steps:", totalrewards.sum())

plt.plot(totalrewards)
plt.title("Rewards")
plt.show()

plot_running_avg(totalrewards)

model.model.save(trained_model)
env.close()