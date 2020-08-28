import gym
import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym import wrappers
from gym.wrappers.monitoring import stats_recorder, video_recorder
from datetime import datetime
import random
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from keras.models import load_model
from keras import backend as K
from pprint import pprint
import cv2
import datetime
from collections import deque
# import ringbuffer
import pickle as pkl
import json
from keras_trainer import ringbuffer


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  fname = os.path.join(os.getcwd(), "train_logs/avg_dqn_10_seq_ra.png")
  plt.savefig(fname)

def reset_video_recorder_filename(filename,env):
    if env.video_recorder:
        env._close_video_recorder()
    print("FILENAME IN VR:{}/{} ".format(env.directory, filename))
    env.video_recorder = video_recorder.VideoRecorder(
                env=env,
                path=os.path.join(env.directory,filename),
                metadata={'episode_id': env.episode_id},
                enabled=env._video_enabled(),
            )
    env.video_recorder.capture_frame()


def transform(s):
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)
    
    upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
    upper_field_bw = upper_field_bw.astype('float')/255
        
    car_field = s[66:78, 43:53]
    #car_field = s[72:84, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
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


def create_nn(model_to_load, stack_len, freeze_hidden=False):
    try:
        m = load_model(model_to_load)
        print("Loaded pretrained model " + model_to_load)
        init_weights = m.get_weights()
        # only do this if loading a saved model. why would we freeze weights on a trained model?
        if freeze_hidden == True:
            m.layers[1].trainable = False # not sure if this is the right way to do this
            m.compile()
        return m, init_weights
    except Exception as e:
        print(e)
        print("Creating new network")
        model = Sequential()
        model.add(Dense(512, input_shape=(stack_len*111,), kernel_initializer="lecun_uniform"))# 7x7 + 3.  or 14x14 + 3 # a
        model.add(Activation('relu'))

        model.add(Dense(11, kernel_initializer="lecun_uniform"))
        # model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        adamax = Adamax(lr=0.01) #Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        if model_to_load is not None:
            model.save(model_to_load)   
        
        return model, model.get_weights()

class DQNAgent():
    def __init__(self, num_episodes, model_name=None, carConfig=None, replay_freq=20, freeze_hidden=False):
        K.clear_session()
        env = gym.make('CarRacingTrain-v1')
        env = wrappers.Monitor(env, 'flaskapp/static', force=False, resume = True, video_callable=None, mode='evaluation', write_upon_reset=False)
        self.carConfig = carConfig
        self.curr_pointer = 0
        self.env = env
        self.gamma = 0.99
        self.K = 10
        self.stack_len = 4  # number of continuous frames to stack
        self.model_name = model_name
        #model_name = "flask_model/avg_dqn_4_seq_model_every50_2_100.h5"
        self.model, self.init_weights = create_nn(model_name, self.stack_len, freeze_hidden)  # consecutive steps, 111-element vector for each state
        self.target_models = []
        for _ in range(self.K):
            target_model, _ = create_nn(model_name, self.stack_len)
            target_model.set_weights(self.init_weights)
            self.target_models.append(target_model)
        self.model.summary()
        self.replay_freq = replay_freq
        if not model_name:
            MEMORY_SIZE = 10000
        else:
            MEMORY_SIZE = 5000  # smaller memory for retraining
        self.memory = ringbuffer.RingBuffer(MEMORY_SIZE)
        self.num_episodes = num_episodes

    def predict(self, s):
        return self.model.predict(np.reshape(s, (1, self.stack_len*111)), verbose=0)[0]

    def target_predict(self, s):
        total_pred = self.target_models[0].predict(np.reshape(s, (1, self.stack_len*111)), verbose=0)[0]
        for i in range(1, self.K):
            pred = self.target_models[i].predict(np.reshape(s, (1, self.stack_len*111)), verbose=0)[0]
            total_pred += pred
        next_pred = total_pred/self.K
        return next_pred

    def update_targets(self):
        model_weights = self.model.get_weights()
        self.target_models[self.curr_pointer%self.K].set_weights(model_weights)

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
        
        return [steering, gaz, brake]

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        old_states = []
        old_state_preds = []
        for (old_state, argmax_qval, reward, next_state) in batch:
            next_state_pred = self.target_predict(next_state)
            max_next_pred = np.max(next_state_pred)
            old_state_pred = self.predict(old_state)
            target_q_value = reward + self.gamma * max_next_pred
            y = old_state_pred[:]
            y[argmax_qval] = target_q_value
            old_states.append(old_state)
            old_state_preds.append(y.reshape(1, 11))
        old_states = np.reshape(old_states, (batch_size, 111*self.stack_len))
        old_state_preds = np.array(old_state_preds).reshape(batch_size, 11)
        self.model.fit(old_states, old_state_preds, batch_size=batch_size, epochs=1, verbose=0, workers=10, use_multiprocessing=True)



    def play_one(self, eps,train=True,video_path=None, test_but_remember=False):
        if self.carConfig:
            # print("TRAINING WITH CAR CONFIG: ")
            # print(self.carConfig)
            observation = self.env.reset(self.carConfig)
        else: 
            observation = self.env.reset()
        if video_path is not None:
            print("Setting video path to: {}".format(video_path))
            reset_video_recorder_filename(video_path,self.env)
        done = False
        full_reward_received = False
        totalreward = 0
        iters = 0
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        stacked_state = np.array([state]*self.stack_len)
        while not done:
            argmax_qval, qval = self.sample_action(stacked_state, eps)
            prev_state = stacked_state
            action = self.convert_argmax_qval_to_env_action(argmax_qval)
            observation, reward, done, info = self.env.step(action)

            a, b, c = transform(observation)        
            curr_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            stacked_state = np.append(stacked_state[1:], [curr_state], axis=0) # appending the lastest frame, pop the oldest
            # print('state shape: ', stacked_state.shape)
            # add to memory
            if train == True or test_but_remember == True:
                self.memory.append((prev_state, argmax_qval, reward, stacked_state))
            if iters%100==0:
                self.curr_pointer += 1
                self.update_targets()
            # replay batch from memory every 20 steps
            if self.replay_freq!=0 and train==True:
                if iters % self.replay_freq==0 and iters>10:
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
        return totalreward, iters

    def train(self, retrain=False):
        totalrewards = np.empty(self.num_episodes)
        for n in range(self.num_episodes):
            print("training ", str(n))
            if not self.model_name:
                eps = 0.01 #0.5/np.sqrt(n + 100)
            else: # want to use a very small eps during retraining
                eps = 0.1
            totalreward, iters = self.play_one(eps)
            totalrewards[n] = totalreward
            print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
            if n>=0 and n%50==0 and not self.model_name:
                # save model (assuming this is NOT the flask app, which WILL pass a model name)
                trained_model = os.path.join(os.getcwd(),"train_logs/avg_dqn_{}.h5".format(str(n)))
                with open("train_logs/avg_dqn_total_rewards_{}.pkl".format(str(n)),'wb+') as outfile:
                    pkl.dump(totalrewards, outfile)
                self.model.save(trained_model)

        if self.model_name:
            # we assume that this IS the flask app; if you are trying to retrain FROM an h5, put it in the flask_model directory for now
            print('saving: ', self.model_name)
            self.model.save(self.model_name)
            
            plt.plot(totalrewards)
            model_name_no_extension = os.path.splitext(self.model_name)[0]
            rp_name = os.path.join(os.getcwd(), "{}.png".format(model_name_no_extension))
            plt.title("Rewards")
            plt.savefig(rp_name)
            plt.close()
            plot_running_avg(totalrewards)
            with open("{}_rewards_flask.pkl".format(model_name_no_extension),'wb+') as outfile:
                pkl.dump(totalrewards, outfile)
            with open("{}_car_config.json".format(model_name_no_extension),'w+') as outfile:
                json.dump(self.carConfig, outfile)


        if not self.model_name:
            plt.plot(totalrewards)
            rp_name = os.path.join(os.getcwd(), "train_logs/avg_dqn.png")
            plt.title("Rewards")
            plt.savefig(rp_name)
            plt.close()
            plot_running_avg(totalrewards)
            with open(os.path.join(os.getcwd(), "train_logs/avg_dqn_total_rewards_final.pkl"),'wb+') as outfile:
                pkl.dump(totalrewards, outfile)
        return totalrewards

if __name__ == "__main__":
    trainer = DQNAgent(1001, None, replay_freq=10)
    import pdb; pdb.set_trace()
    trainer.train()
