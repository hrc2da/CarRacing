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
import tensorflow as tf
from tensorflow import keras
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax
from tensorflow.keras.models import load_model
from pprint import pprint
import cv2
import datetime
from collections import deque
import pickle as pkl
import json
sys.path.append('/home/zhilong/Documents/HRC/CarRacing')
from keras_trainer import ringbuffer


def plot_running_avg(totalrewards, filename):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.savefig(filename)
    plt.close()

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
    # if speed>0:
    #     print("speed element: ", speed)
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255
    
        
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

vector_size = 10*10 + 7 + 4


def create_nn(model_to_load, stack_len, freeze_hidden=False, lr = 0.001):
    try:
        m = load_model(model_to_load)
        print("Loaded pretrained model " + model_to_load)
        init_weights = m.get_weights()
        #TODO: remove this if don't want to reset optimizer
        # m.compile(loss='mse', optimizer=Adamax(lr=lr))
        # only do this if loading a saved model. why would we freeze weights on a trained model?
        if freeze_hidden == True:
            # m.layers[2].trainable = False # not sure if this is the right way to do this
            m.compile(loss='mse', optimizer=Adamax(lr=lr))
            print("recompiled")
            
        return m, init_weights
    except Exception as e:
        model = Sequential()
        model.add(Dense(512, input_shape=(stack_len*111,), kernel_initializer="lecun_uniform"))# 7x7 + 3.  or 14x14 + 3 # a
        model.add(Activation('relu'))

        model.add(Dense(11, kernel_initializer="lecun_uniform"))

        adamax = Adamax(lr=lr) #Adamax(lr=0.001)
        model.compile(loss='mse', optimizer=adamax)
        if model_to_load:
            model.save(model_to_load)
        
        return model, model.get_weights()

class DQNAgent():
    def __init__(self, num_episodes, model_name=None, carConfig=None, replay_freq=20, memory_size=10000, freeze_hidden=False, lr=0.01):
        # K.clear_session()
        env = gym.make('CarRacingTrain-v1')
        env = wrappers.Monitor(env, 'flaskapp/static', force=False, resume = True, video_callable=None, mode='evaluation', write_upon_reset=False)
        tf.compat.v1.disable_eager_execution()
        self.carConfig = carConfig
        self.curr_pointer = 0
        self.env = env
        self.lr = lr
        self.gamma = 0.99
        self.K = 10
        self.stack_len = 4  # number of continuous frames to stack
        self.model_name = model_name
        self.model, self.init_weights = create_nn(model_name, self.stack_len, freeze_hidden, self.lr)  # consecutive steps, 111-element vector for each state
        self.target_models = []
        for _ in range(self.K):
            target_model, _ = create_nn(model_name, self.stack_len)
            target_model.set_weights(self.init_weights)
            self.target_models.append(target_model)
        self.orig_model, _ = create_nn(model_name, self.stack_len, freeze_hidden, lr)
        self.model.summary()
        self.replay_freq = replay_freq
        if not model_name:
            self.MEMORY_SIZE = memory_size
            self.memory = ringbuffer.RingBuffer(self.MEMORY_SIZE)
        else:
            self.MEMORY_SIZE = memory_size  # smaller memory for retraining
            self.memory = ringbuffer.RingBuffer(self.MEMORY_SIZE)
            # self.memory.load(os.path.join(os.getcwd(),"keras_trainer/dumped_memory_3eps.pkl")) #load memory
        self.num_episodes = num_episodes
        self.state_hist = []
        self.loss_hist = []
        self.retrained_q_hist = []
        self.orig_model_q_hist = []
        self.play_one_orig_best_q = []
        self.play_one_orig_taken_q = []
        self.play_one_retrained_best_q = []
        self.play_one_retrained_taken_q = []
        self.orig_peak_hist = []
        self.retrained_peak_hist = []
        self.actions = []

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
            return np.random.randint(0, 11), qval
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
            
        
        return [steering, gaz, brake]

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        old_states = []
        old_state_preds = []
        max_orig_predicts = []
        max_retrained_predicts = []
        for (old_state, argmax_qval, reward, next_state) in batch:
            next_state_pred = self.target_predict(next_state)
            max_next_pred = np.max(next_state_pred)
            ##### run pred with original model and retrained model to see q val divergence
            orig_pred = self.orig_model.predict(np.reshape(old_state, (1, self.stack_len*111)), verbose=0)[0]
            max_orig_pred = np.max(orig_pred)
            retrained_pred = self.predict(old_state)
            max_retrained_pred = np.max(retrained_pred)
            max_orig_predicts.append(max_orig_pred)
            max_retrained_predicts.append(max_retrained_pred)
            ##### done
            old_state_pred = self.predict(old_state)
            target_q_value = reward + self.gamma * max_next_pred
            y = old_state_pred[:]
            y[argmax_qval] = target_q_value
            old_states.append(old_state)
            old_state_preds.append(y.reshape(1, 11))
        old_states = np.reshape(old_states, (batch_size, 111*self.stack_len))
        old_state_preds = np.array(old_state_preds).reshape(batch_size, 11)
        hist = self.model.fit(old_states, old_state_preds, batch_size=batch_size, epochs=1, verbose=0)
        self.loss_hist += hist.history['loss']
        self.orig_model_q_hist.append(np.mean(max_orig_predicts))
        self.retrained_q_hist.append(np.mean(max_retrained_predicts))

    def play_one(self, eps, n, train=True,video_path=None):
        # env= gym.make('CarRacingTrain-v1')
        # self.env = wrappers.Monitor(env, 'flaskapp/static', force=False, resume = True, video_callable=None, mode='evaluation', write_upon_reset=False)

        if self.carConfig:
            print("TRAINING WITH CAR CONFIG: ")
            print(self.carConfig)
            observation = self.env.reset(self.carConfig)
        else: 
            observation = self.env.reset()
        if video_path is not None:
            print("Setting video path to: {}".format(video_path))
            reset_video_recorder_filename(video_path,self.env)
        done = False
        totalreward = 0
        iters = 0
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        stacked_state = np.array([state]*self.stack_len, dtype='float32')
        while not done:
            argmax_qval, qval = self.sample_action(stacked_state, eps)
            prev_state = stacked_state
            ##### Get the divergence in qval between orig and retrained network
            orig_pred = self.orig_model.predict(np.reshape(prev_state, (1, self.stack_len*111)), verbose=0)[0]
            orig_argmax = np.argmax(orig_pred)
            orig_max_q = np.max(orig_pred)
            orig_taken_q = orig_pred[argmax_qval]
            retrained_taken_q = qval[argmax_qval]
            retrained_max_q = qval[orig_argmax]
            orig_peak_diff = orig_max_q-orig_pred[orig_pred.argsort()[-2]]
            retrained_peak_diff = np.max(qval)-qval[qval.argsort()[-2]]
            self.play_one_orig_best_q.append(orig_max_q)
            self.play_one_orig_taken_q.append(orig_taken_q)
            self.play_one_retrained_best_q.append(retrained_max_q)
            self.play_one_retrained_taken_q.append(retrained_taken_q)
            self.orig_peak_hist.append(orig_peak_diff)
            self.retrained_peak_hist.append(retrained_peak_diff)
            self.state_hist.append(np.reshape(prev_state, (1, self.stack_len*111)))
            # if orig_peak_diff>5:
            #     self.actions.append(orig_argmax)
            #     print('ITERATION NUMBER: ', iters)
            # else:
            #     self.actions.append(0)
            ##### Done
            action = self.convert_argmax_qval_to_env_action(argmax_qval)
            observation, reward, done, info = self.env.step(action)

            a, b, c = transform(observation)        
            curr_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            curr_state.astype('float32')
            stacked_state = np.append(stacked_state[1:], [curr_state], axis=0) # appending the lastest frame, pop the oldest
            # add to memory
            if train == True:
                self.memory.append((prev_state, argmax_qval, reward, stacked_state))
            if iters%250==0:
                self.curr_pointer += 1
                self.update_targets()
            # replay batch from memory every 20 steps
            if self.replay_freq!=0 and train==True:
                if iters % self.replay_freq==0:# and n>0: #TODO: don't replay on small buffer
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
        # env.close()
        # self.env.close()
        return totalreward, iters

    def train(self, retrain=False):
        self.timestamp = time.strftime("%m%d_%H%M")
        if retrain:
            self.timestamp += "_retrain"
        #create directory for this training run
        dir_name = os.path.join(os.getcwd(), "train_logs/{}".format(self.timestamp))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        train_params = {"mem_size": self.MEMORY_SIZE, 
                        "num_episodes": self.num_episodes, 
                        "retrain": retrain,
                        "discount rate": self.gamma,
                        "replay_freq": self.replay_freq, 
                        "step size": self.lr,
                        "training model name": None or self.model_name,
                        "eps": 1/np.sqrt(101) if not self.model_name else 0.01} # remember to change eps for retrain as appropriate
        with open(os.path.join(os.getcwd(), "train_logs/{}/training_params.json".format(self.timestamp)),'w+') as outfile:
                json.dump(train_params, outfile)

        totalrewards = np.empty(self.num_episodes)
        for n in range(self.num_episodes):
            print("training ", str(n))
            if not self.model_name:
                eps = 1/np.sqrt(n + 100)
            else: # want to use a very small eps during retraining
                eps = 0.01
                # eps = 1/np.sqrt(n + 50)
                #testing with normal training eps
                # eps = 1/np.sqrt(n + 300)
            totalreward, iters = self.play_one(eps, n)
            totalrewards[n] = totalreward
            print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())        
            if n>=0 and n%50==0 and not self.model_name:
                # save model (assuming this is NOT the flask app, which WILL pass a model name)
                trained_model = os.path.join(os.getcwd(),"train_logs/{}/avg_dqn_ep_{}.h5".format(self.timestamp, str(n)))
                with open(os.path.join(os.getcwd(), "train_logs/{}/avg_dqn_ep_{}.pkl".format(self.timestamp, str(n))),'wb+') as outfile:
                    pkl.dump(totalrewards, outfile)
                self.model.save(trained_model)

        if self.model_name:
            # we assume that this IS the flask app; if you are trying to retrain FROM an h5, put it in the flask_model directory for now
            state_hist_name = os.path.join(os.getcwd(), "train_logs/{}/{}".format(self.timestamp, 'state history'))
            self.state_hist = np.reshape(self.state_hist, (len(self.state_hist), 444))
            unique_state = np.unique(self.state_hist, axis=0) # remove duplicate states
            print("state hist after filtering: ", unique_state.shape)
            with open(state_hist_name, 'wb+') as f:
                pkl.dump(unique_state, f)
            model_name_no_extension = os.path.basename(self.model_name)
            new_model_name = os.path.join(os.getcwd(), "train_logs/{}/{}".format(self.timestamp, model_name_no_extension))
            print('saving: ', new_model_name)
            self.model.save(new_model_name)
            rp_name = os.path.join(os.getcwd(), "train_logs/{}/rewards_plot_{}.png".format(self.timestamp, model_name_no_extension))
            plt.plot(totalrewards, label='retraining reward')
            plt.title("Rewards")
            plt.legend()
            plt.savefig(rp_name)
            plt.close()
            rap_name = os.path.join(os.getcwd(), "train_logs/{}/ra_plot_{}.png".format(self.timestamp, model_name_no_extension))
            plot_running_avg(totalrewards, rap_name)
            with open(os.path.join(os.getcwd(), "train_logs/{}/{}_rewards_flask.pkl".format(self.timestamp, model_name_no_extension)),'wb+') as outfile:
                pkl.dump(totalrewards, outfile)
            with open(os.path.join(os.getcwd(), "train_logs/{}/{}_car_config.json".format(self.timestamp, model_name_no_extension)),'w+') as outfile:
                json.dump(self.carConfig, outfile)
            # plot the training loss
        plt.plot(self.loss_hist, label='training loss')
        plt.legend()
        tl_name = os.path.join(os.getcwd(), "train_logs/{}/training_loss.png".format(self.timestamp))
        plt.savefig(tl_name)
        plt.close()
        with open(os.path.join(os.getcwd(), "train_logs/{}/training_loss.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.loss_hist, outfile)
        
        ## plotting of training q val hist
        plt.plot(self.orig_model_q_hist, label='max q of original')
        plt.plot(self.retrained_q_hist, label='max q of retrained')
        mq_name = os.path.join(os.getcwd(), "train_logs/{}/training_max_q_vals.png".format(self.timestamp))
        plt.legend()
        plt.savefig(mq_name)
        plt.close()
        with open(os.path.join(os.getcwd(), "train_logs/{}/orig_q_hist.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.orig_model_q_hist, outfile)
        with open(os.path.join(os.getcwd(), "train_logs/{}/retrained_q_hist.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.retrained_q_hist, outfile)
        ##

        ## plotting of sampled action q val
        plt.plot(self.play_one_retrained_taken_q, label='retrain max qval taken')
        plt.plot(self.play_one_orig_taken_q, label='orig max qval taken')
        stq_name = os.path.join(os.getcwd(), "train_logs/{}/sampled_taken_maxq.png".format(self.timestamp))
        plt.legend()
        plt.savefig(stq_name)
        plt.close()
        with open(os.path.join(os.getcwd(), "train_logs/{}/retrain_sampled_taken_q.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.play_one_retrained_taken_q, outfile)
        with open(os.path.join(os.getcwd(), "train_logs/{}/orig_sampled_taken_q.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.play_one_orig_taken_q, outfile)

        plt.plot(self.play_one_retrained_best_q, label='retrain max qval best')
        plt.plot(self.play_one_orig_best_q, label='orig max qval best')
        plt.legend()
        stq_name = os.path.join(os.getcwd(), "train_logs/{}/sampled_best_maxq.png".format(self.timestamp))
        plt.savefig(stq_name)
        plt.close()
        with open(os.path.join(os.getcwd(), "train_logs/{}/retrain_sampled_best_q.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.play_one_retrained_best_q, outfile)
        with open(os.path.join(os.getcwd(), "train_logs/{}/orig_sampled_best_q.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.play_one_orig_best_q, outfile)

        plt.plot(self.retrained_peak_hist, label='retrained diff')
        plt.plot(self.orig_peak_hist, label='orig diff')
        # plt.plot(self.actions, '*', label='actions')
        plt.legend()
        diff_plot_name = os.path.join(os.getcwd(), "train_logs/{}/peaky_diff.png".format(self.timestamp))
        plt.savefig(diff_plot_name)
        plt.close()
        with open(os.path.join(os.getcwd(), "train_logs/{}/retrain_peak_diff.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.retrained_peak_hist, outfile)
        with open(os.path.join(os.getcwd(), "train_logs/{}/orig_peak_diff.pkl".format(self.timestamp)),'wb+') as outfile:
            pkl.dump(self.orig_peak_hist, outfile)
        ##

        if not self.model_name:
            plt.plot(totalrewards)
            # rp_name = os.path.join(os.getcwd(), "train_logs/avg_dqn_lr001_replay20_cpweights250.png")
            rp_name = os.path.join(os.getcwd(), "train_logs/{}/rewards_plot.png".format(self.timestamp))
            plt.title("Rewards")
            plt.savefig(rp_name)
            plt.close()
            rap_name = os.path.join(os.getcwd(), "train_logs/{}/ra_plot.png".format(self.timestamp))
            plot_running_avg(totalrewards, rap_name)
            # with open(os.path.join(os.getcwd(), "train_logs/avg_dqn_total_rewards_final_lr001_replay20_cpweights250.pkl"),'wb+') as outfile:
            with open(os.path.join(os.getcwd(), "train_logs/{}/total_rewards.pkl".format(self.timestamp)),'wb+') as outfile:
                pkl.dump(totalrewards, outfile)
        self.env.close()

if __name__ == "__main__":
    # Load carconfig to train with
    # with open(os.path.join(os.getcwd(),"keras_trainer/best_rand_car_design.json")) as f:
    #     car_config = json.load(f)

    trainer = DQNAgent(301, None, replay_freq=10, lr=0.001)#, carConfig=car_config)
    trainer.train()
