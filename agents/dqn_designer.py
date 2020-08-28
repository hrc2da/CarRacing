import time
import os
import sys
import pickle as pkl

sys.path.append('/home/hrc2/hrcd/cars/carracing')
sys.path.append('/home/zhilong/Documents/HRC/CarRacing')
sys.path.append('/home/dev/scratch/cars/carracing_clean')

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from keras_trainer import ringbuffer
from keras_trainer import avg_dqn

def calculate_flattened_dim(layer_to_flatten, episode_length, prior_conv_layers):
    channels = layer_to_flatten.out_channels
    start_size = episode_length
    for layer in prior_conv_layers:
        n0 = start_size
        k0 = layer.kernel_size[0]
        p0 = layer.padding[0]
        s0 = layer.stride[0]
        # see https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
        
        #output_shape = np.floor((n0 - k0 + p0 + s0)/s0)
        output_shape = int(np.floor(n0/s0)) # I'm just going to assume we've padded it 
        start_size = output_shape
    return start_size

class DeepQNetwork(nn.Module):
    h1_size = 512
    h2_size = 256
    def __init__(self, frame_size, episode_length, action_size):
        super(DeepQNetwork, self).__init__()
        # self.input_layer = torch.nn.Linear(state_size, self.h1_size)
        # self.h1_layer = torch.nn.Linear(self.h1_size, self.h2_size)
        # self.h2_layer = torch.nn.Linear(self.h2_size, action_size)
        # self.relu = torch.nn.ReLU()
        # padding should be: floor(len/stride)*stride + (kernel_size - stride) - len, e.g. floor(500/15)*15 + (30-15) - 500 = 33*15 + 15 -500 = 510 - 500 = 10
        self.input_layer = torch.nn.Conv1d(frame_size, 32, kernel_size = 30, stride = 15, padding = 10) # output is len = 33
        self.h1_layer = torch.nn.Conv1d(32, 64, kernel_size = 5, stride = 2, padding = 0) # output is len = 15
        self.h2_layer = torch.nn.Conv1d(64, 128, kernel_size = 5, stride = 1, padding = 0) # output is len = 11
        #h2_flattened_dimension = self.h2_layer.out_channels*(episode_length - (self.input_layer.kernel_size[0] - 1) - (self.h1_layer.kernel_size[0] - 1) - (self.h2_layer.kernel_size[0] - 1))
        # h2_flattened_dimension = calculate_flattened_dim(self.h2_layer, episode_length, [self.input_layer, self.h1_layer, self.h2_layer])
        h2_flattened_dimension = 128 * 11
        print("H2 Flattened Size: {}".format(h2_flattened_dimension))
        self.dense_layer = torch.nn.Linear(h2_flattened_dimension, action_size)
        self.flatten_layer = nn.Flatten()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        

    def forward(self, state):
        h1_in = self.input_layer(state)
        h1_in = self.relu1(h1_in)
        h1_out = self.h1_layer(h1_in)
        h2_in = self.relu2(h1_out)
        h2_out = self.h2_layer(h2_in)
        h2_flattened = self.flatten_layer(h2_out)
        dense_in = self.relu3(h2_flattened)
        dense_out = self.dense_layer(dense_in)
        q_vals = self.sigmoid(dense_out)
        return q_vals



class DQN_Designer:
    '''

        The "state" of this task is the current ability of the agent to drive the current car. This is obviously hidden. However, we can observe how well the car is driving
        in a sequence of episodes.

        The action that the agent can take is modifying the design. For now, to keep it simple, the agent can only change the horsepower.

        The "step" involves retraining the agent on the new car for a set number of episodes.

        The "next state" is observed by test-driving (deterministic) the retrained agent on the new car.

        The reward is calculated as the change in mean reward across the test drive from the previous "state".
    '''
    MEMORY_SIZE = 10000
    HP_STEP_SIZE = 5000
    HP_MIN = 5000

    def __init__(self, initial_driver=None, initial_car=None, gamma=0.9, replay_frequency=1):
        '''
            
        '''
        self.driver = initial_driver
        self.car_config = initial_car
        self.gamma = gamma
        self.replay_frequency = replay_frequency
        self.memory = ringbuffer.RingBuffer(self.MEMORY_SIZE)
        # self.episode_length = self.driver.env.get_episode_lengths()[0] ## this is problematic if the episode ends early
        self.episode_length = 500 #hard code it for safety for now
        self.state = self.setup_state(driver)
        self.state_size = self.state.squeeze().shape[0]
        
        print("EPISODE LENGTH: {}".format(self.episode_length))
        self.action_size = 3 # actions = {hp_same, hp_up, hp_down}
        self.dqn = DeepQNetwork(self.state_size, self.episode_length, self.action_size).to(device)
        self.target_dqn = DeepQNetwork(self.state_size, self.episode_length, self.action_size).to(device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.criterion = torch.nn.MSELoss() # so loss between the q pred and target
        self.optimizer = torch.optim.SGD(self.dqn.parameters(), lr=0.01)

    def setup_driver(self, initial_driver):
        if initial_driver is None:
            initial_driver = None #TODO: Implement this
        return initial_driver

    def setup_car(self, initial_car):
        if initial_car is None:
            initial_car = None #TODO: Implement this
        return initial_car

    def setup_state(self, driver):
        driver.play_one(eps = 0, test_but_remember=True) # we want to fill the replay buffer, so we will let it train one episode before redesigning
        state = self.extract_state_from_driver(driver)
        return state

    def modify_car(self, action):
        prior_hp = self.car_config['eng_power']
        if action == 0:
            return
        if action == 1:
            self.car_config['eng_power'] += self.HP_STEP_SIZE
        else:
            if self.car_config['eng_power'] - self.HP_MIN > 0:
                self.car_config['eng_power'] -= self.HP_STEP_SIZE
            else:
                self.car_config['eng_power'] = self.HP_MIN
        print("CHANGING HP FROM {} TO {}".format(prior_hp, self.car_config['eng_power']))
        
    def extract_state_from_history(self, driving_history, padding=0):
        # take an episode of a driver driving and extract the trajectory. pad with zeros if early termination
        trajectory = []
        for frame in driving_history:
            old_state, argmax_qval, reward, next_state = frame
            trajectory.append(old_state[-1]) # add the most recent frame to the trajectory
        if len(trajectory) == 0:
            raise ValueError("Got an empty episode!!!")
        for _ in range(padding):
            trajectory.append(np.zeros(trajectory[-1].shape))
        assert len(trajectory) == self.episode_length
        assert len(trajectory[-1]) == 111
        return torch.Tensor(trajectory).transpose(0,1).unsqueeze(0).to(device) # returns a [1, n_channels, n_steps] tensor

    def extract_state_from_driver(self, driver):
        # get the state from the last episode the driver played
        episode_length = driver.env.get_episode_lengths()[-1]
        history = driver.memory[-episode_length:]
        padding = self.episode_length - episode_length
        return self.extract_state_from_history(history, padding)

    def play_one(self, eps):
        # sample the next action (car modification), based on the current state
        with torch.no_grad():
            q_vals = self.dqn(self.state)
            action = torch.argmax(q_vals)
            if torch.rand(1) < eps:
                action = torch.randint(0, len(q_vals), (1,))
        # take the action (modify the car) and run M retraining episodes
        print("Modifying Car")
        self.modify_car(action)
        self.driver.env.reset(self.car_config)
        print("training")
        training_rewards = self.driver.train(retrain=True)
        next_state = self.extract_state_from_driver(driver)
        # test drive for N episodes (as set by the dqndriver class)
        #self.driver.play_one() # we won't test drive for now, will just use the last training episode as the state
        # calculate reward based on aggregate stats; another option would be to test drive and just use that reward, but seems it would be noisy?
        halfway_point = len(training_rewards)//2
        reward = np.mean(training_rewards[halfway_point:]) - np.mean(training_rewards[:halfway_point])
        # add experience to the replay buffer
        print("Designer's reward: {}".format(reward))
        self.memory.append((self.state, action, reward, next_state))
        # update the current state
        self.state = next_state
        return action, reward, training_rewards

    def train(self, n_episodes, min_memory = 20):
        # run n iterations with the same car (where each iteration is a test drive with one design change)
        actions = []
        rewards = []
        training_rewards = []
        eps = 0.2
        for t in range(n_episodes):
            print("Episode {}***********".format(t))
            action, reward, training_reward = self.play_one(eps)
            eps = eps*0.5
            actions.append(action)
            rewards.append(reward)
            training_rewards.append(training_reward)
            if t % self.replay_frequency == 0 and t > min_memory:
                bs = min(len(self.memory), 128)
                print("replaying wtih batch size {}...".format(bs))
                self.replay(batch_size = bs )
                print("Average Designer Reward: {}".format(np.mean(rewards)))
                t_rewards = [np.mean(tr) for tr in training_rewards]
                print("Average Driver Reward: {}".format(t_rewards))
                
        return actions, rewards, training_rewards

    def replay(self, batch_size = 32, epochs=10):
        
        # sample batch_size samples from the memory
        batch = self.memory.sample(batch_size)
        # update the target network (hopefully will help learn)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        for e in range(epochs): 
            targets = None
            predictions = None
            for (old_state, argmax_qval, reward, next_state) in batch:
                # note to self: the targets don't change and aren't connected to the graph, so you can compute these once for all the epochs
                # note to self: actually, that's not true, because the argmax q-val in the target is based on the next state's q_vals, according to the network
                # so if the network changes, the target should also change.
                
                # get the q value of the best action from next state for the td-update
                with torch.no_grad(): # I think we can turn off gradient tracking since we are not back-prop'ing these
                    q_vals_from_next_state = self.target_dqn(next_state) # returns tensor
                    max_q_from_next_state = torch.max(q_vals_from_next_state)
                    action_q_value = reward + self.gamma * max_q_from_next_state
                
                predicted_q_vals = self.dqn(old_state.clone()) # this one should have gradients, as the loss is backprop'ed wrt the prediction
                
                target_q_vals = predicted_q_vals.clone().detach() #is this the same as detach.clone but worse performance? 
                target_q_vals[0, argmax_qval] = action_q_value #in-place is ok b/c no gradient needed (I think)
                
                assert predicted_q_vals.requires_grad is True
                assert target_q_vals.requires_grad is False
                
                if targets is None:
                    targets = target_q_vals
                    predictions = predicted_q_vals
                else:
                    targets = torch.cat((targets,target_q_vals))
                    predictions = torch.cat((predictions,predicted_q_vals))
        
            # clear the optimizer
            self.optimizer.zero_grad()
            loss = self.criterion(predictions,targets)
            print("Epoch {} loss is {}".format(e,loss))
            loss.backward()
            self.optimizer.step()

if __name__ == '__main__':
    import time

    start_time_sec = time.time()

    driver = avg_dqn.DQNAgent(4) # driver trains for 4 episodes per rollout
    driver.env.reset() # get the car config (default for now)
    # have to unwrap the car from the gym monitor and time limit wrappers
    designer = DQN_Designer(driver,driver.env.env.env.car.config, replay_frequency=20)
    episodes = []
    results = designer.train(41) # have the designer train with this car for 10 design episodes (so a total of 40 episodes)
    episodes.append(results)
    for i in range(50):
        driver = avg_dqn.DQNAgent(4)
        driver.env.reset()
        designer.driver = driver
        designer.car_config = driver.env.env.env.car.config
        results = designer.train(41)
        episodes.append(results)
        model_path = os.path.join(os.getcwd(),"train_logs/dqn_designer_{}.pt".format(i))
        torch.save(designer.dqn.state_dict(),model_path)

    with open("train_logs/dqn_designer_results.pkl",'wb+') as outfile:
        pkl.dump(episodes, outfile)

    end_time_sec = time.time()

    print("Total time in sec: {}".format(end_time_sec - start_time_sec))
    print("TOtal time in hrs: {}".format((end_time_sec - start_time_sec)/3600.0))