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
import torch
import torch.nn as nn
from pprint import pprint
import cv2
import datetime
from collections import deque
import ringbuffer
import torch_ae
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  fname = os.path.join(os.getcwd(), "train_logs/ae_dqn_10000_3ra.png")
  plt.savefig(fname)




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
        model = torch.nn.Sequential(
            torch.nn.Linear(input_shape, h1_shape),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_shape, output_shape)
        )
        model.load_state_dict(torch.load(model_to_load))
        print("Loaded pretrained model " + model_to_load)
        init_weights = m.state_dict()
        return m, init_weights
    except:
        print("Creating new network")
        input_shape = 4*111
        h1_shape = 512
        output_shape = 11
        
        model = torch.nn.Sequential(
            torch.nn.Linear(input_shape, h1_shape),
            torch.nn.ReLU(),
            torch.nn.Linear(h1_shape, output_shape)
        )
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4)
        
        return model.to(device), model.state_dict(), loss_fn, optimizer

class DenseQNetwork(nn.Module):
    h1_size = 512
    h2_size = 64
    def __init__(self, state_size, action_size, stack_size = 1):
        super(DenseQNetwork, self).__init__()
        self.input_layer = torch.nn.Linear(stack_size * state_size, self.h1_size)
        #self.h1_layer = torch.nn.Linear(self.h1_size, self.h2_size)
        #self.h2_layer = torch.nn.Linear(self.h2_size, action_size)
        self.h1_layer = torch.nn.Linear(self.h1_size, action_size)
        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.ReLU()
        

    def forward(self, state):
        input_out = self.input_layer(state)
        input_activated = self.relu1(input_out)
        q_vals = self.h1_layer(input_activated)
        #h1_out = self.h1_layer(input_activated)
        #h1_activated = self.relu2(h1_out)
        #q_vals = self.h2_layer(h1_activated)
        return q_vals


class DQNAgent():
    
    action_size = 11

    def __init__(self, num_episodes, model_name=None, carConfig=None, replay_freq=20):
        env = gym.make('CarRacingTrain-v1')
        env = wrappers.Monitor(env, 'monitor-folder', force=True)
        self.carConfig = carConfig
        self.env = env
        self.gamma = 0.99
        self.K = 10
        self.model_name = model_name
        self.stack_size = 4

        # setup the autoencoder and q networks
        self.ae = torch_ae.AutoEncoder().to(device)
        self.ae.load_state_dict(torch.load('keras_trainer/conv_autoencoder.pth'))
        self.latent_size = self.ae.latent_size
        self.state_size = self.latent_size #+ 7 # extracted bottom bar stuff
        self.model = DenseQNetwork(self.state_size, self.action_size, self.stack_size).to(device)
        self.target_model = DenseQNetwork(self.state_size, self.action_size, self.stack_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adamax(self.model.parameters())
        
        #self.model, self.init_weights, self.loss_fn, self.optimizer = create_nn(model_name)  # 4 consecutive steps, 111-element vector for each state
        #self.target_model, _, _, _ = create_nn(model_name)
        #self.past_weights = deque(maxlen=self.K)
        #self.past_weights.append(self.init_weights)
        #self.target_model.load_state_dict(self.init_weights)
        
        self.replay_freq = replay_freq
        if not model_name:
            MEMORY_SIZE = 10000
        else:
            MEMORY_SIZE = 5000  # smaller memory for retraining
        self.memory = ringbuffer.RingBuffer(MEMORY_SIZE)
        self.num_episodes = num_episodes


    def predict(self, state):
        #return self.model.forward(torch.from_numpy(np.reshape(s, (1, 4*111))).float().to(device))
        return self.model(state)

    def target_predict(self, state):
        #return self.target_model.forward(torch.from_numpy(np.reshape(s, (1, 4*111))).float().to(device))
        return self.target_model(state)



    # def update(self, s, G, B):
    #     self.model.fit(s, np.array(G).reshape(-1, 11), batch_size=B, epochs=1, use_multiprocessing=True, verbose=0)

    def sample_action(self, state, eps):
        # print('THE SHAPE FOR PREDICTION: ', s.shape)
        qvals = self.predict(torch.from_numpy(np.reshape(state, (1, self.stack_size*self.state_size))).float().to(device))
        if np.random.random() < eps:
            return torch.randint(low=0,high=11,size=(1,)), qvals #pytorch randint upper bound is one above sample range
        else:
            return torch.argmax(qvals), qvals

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

    def transform(self, obs):        
        bottom_black_bar = obs[84:, 12:]
        # convert raw obs to z, mu, logvar
        obs = obs[0:84, :, :]
        
        obs = np.array(Image.fromarray(obs.astype(np.uint8)).resize((64,64)))
        #obs = ((1.0 - obs) * 255).round().astype(np.uint8)
        #result = np.copy(obs).astype(np.float)/255.0
        obs = obs.astype(np.float)/255.0
        obs = torch.Tensor(obs.reshape(1, 3, 64, 64)).to(device)

        with torch.no_grad():
            embedding = self.ae.encoder(obs)

        img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
        bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
        bottom_black_bar_bw = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation = cv2.INTER_NEAREST)
        
        return embedding.cpu().numpy(), bottom_black_bar_bw

    def replay(self, batch_size):
        batch = self.memory.sample(batch_size)
        targets = None
        predictions = None
        for (old_state, argmax_qval, reward, next_state) in batch:
            old_state = torch.from_numpy(np.reshape(old_state, (1, self.stack_size*self.state_size))).float().to(device).requires_grad_()
            next_state = torch.from_numpy(np.reshape(next_state, (1, self.stack_size*self.state_size))).float().to(device)
            
            with torch.no_grad(): # I think we can turn off gradient tracking since we are not back-prop'ing these
                next_state_pred = self.target_predict(next_state) # returns tensor
                max_next_pred = torch.max(next_state_pred)
            old_state_pred = self.predict(old_state)
            target_q_value = reward + self.gamma * max_next_pred
            target = old_state_pred.clone().detach() #is this the same as detach.clone but worse performance?
            target[0, argmax_qval] = target_q_value #in-place is ok b/c no gradient needed (I think)
            if targets is None:
                targets = target
                predictions = old_state_pred
            else:
                targets = torch.cat((targets,target))
                predictions = torch.cat((predictions,old_state_pred))
        loss = self.criterion(predictions,targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



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
        embedded_obs, black_bar = self.transform(observation)
        
        #state = np.concatenate((embedded_obs[0], np.array([compute_steering_speed_gyro_abs(black_bar)]).reshape(1,-1).flatten()), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        state = embedded_obs
        stacked_state = np.array([state]*self.stack_size) #I think this makes copies of the ref, but it might be ok here
        # print('state shape: ', stacked_state.shape)
        while not done:
            argmax_qval, qval = self.sample_action(stacked_state, eps)
            prev_state = stacked_state
            action = self.convert_argmax_qval_to_env_action(argmax_qval.detach().cpu().item())
            observation, reward, done, info = self.env.step(action)

            embedded_obs, black_bar = self.transform(observation)        
            #curr_state = np.concatenate((embedded_obs[0], np.array([compute_steering_speed_gyro_abs(black_bar)]).reshape(1,-1).flatten()), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            curr_state = embedded_obs
            stacked_state = np.append(stacked_state[1:], [curr_state], axis=0) # appending the lastest frame, pop the oldest
            # print('state shape: ', stacked_state.shape)
            # add to memory
            self.memory.append((prev_state, argmax_qval, reward, stacked_state)) # these are all np arrays, except for argmax_qval, which is a tensor
            # replay batch from memory every 20 steps
            if self.replay_freq!=0:
                if iters % self.replay_freq==0 and iters>10:
                    try:
                        batch_size = min(256,len(self.memory))
                        self.replay(batch_size)
                    except Exception as e: # will error if the memory size not enough for minibatch yet
                        print("error when replaying: ", e)
                        raise e
            totalreward += reward
            iters += 1
            
            if iters > 1500:
                print("This episode is stuck")
                break
        
        # update the target model at the end of each episode
        self.target_model.load_state_dict(self.model.state_dict())
        return totalreward, iters

    def train(self, retrain=False):
        totalrewards = np.empty(self.num_episodes)
        for n in range(self.num_episodes):
            print("training ", str(n))
            if not self.model_name:
                eps = 0.75/np.sqrt(n + 100)
            else: # want to use a very small eps during retraining
                eps = 0.1
            totalreward, iters = self.play_one(eps)
            totalrewards[n] = totalreward
            print("episode:", n, "iters", iters, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())     
            if n>0 and n%500==0 and not self.model_name:
                # save model
                trained_model = os.path.join(os.getcwd(),"train_logs/ae_dqn_trained_model_torch_3_{}.h5".format(str(n)))
                torch.save(self.model.state_dict(), trained_model)

        if self.model_name:
            print('saving: ', self.model_name)
            torch.save(self.model.state_dict(), self.model_name)

        if not self.model_name:
            plt.plot(totalrewards)
            rp_name = os.path.join(os.getcwd(), "train_logs/ae_dqn_10000_rewards_torch_3.png")
            plt.title("Rewards")
            plt.savefig(rp_name)
            plt.close()
            plot_running_avg(totalrewards)
        self.env.close()

if __name__ == "__main__":
    # ae = torch_ae.AutoEncoder()
    # import pdb; pdb.set_trace()
    trainer = DQNAgent(5001, None, replay_freq=20)
    trainer.train()
