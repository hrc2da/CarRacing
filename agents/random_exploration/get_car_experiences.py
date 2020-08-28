from ga_generate_random_car_15 import nsgaii_agent, np_arr2car
import numpy as np
import gym
import cv2
import pickle as pkl
import os
from tqdm import tqdm
import time

agent = nsgaii_agent()
num_cars = 500
num_eps_per_car = 10
experiences = {'cars':[], 'experience':[]} # NOTE this data structure!

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
        print("error: random action invalid")
        
    
    return [steering, gaz, brake]

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

env = gym.make('CarRacingRandomStart-v1')
env = gym.wrappers.Monitor(env, 'flaskapp/static', force=False, resume = True, video_callable=None, mode='evaluation', write_upon_reset=False)

for i in tqdm(range(num_cars)):
    arr = np.random.rand(15)
    car_config = np_arr2car(arr,agent)
    print(car_config)
    experiences['cars'].append(car_config)
    for j in range(num_eps_per_car):
        curr_car_exp = []
        # rand_start = np.random.randint(0, 114) # recall the track is of length 113
        observation = env.reset(car_config)
        done = False
        a, b, c = transform(observation)
        state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
        stacked_state = np.array([state]*4, dtype='float32')

        # run a random episode
        while not done:
            action = np.random.randint(0, 11)
            action = convert_argmax_qval_to_env_action(action)
            prev_state = stacked_state
            observation, reward, done, _ = env.step(action)
            a, b, c = transform(observation)        
            if b.all():
                break
            # print("*"*30)
            # print(reward)
            # print(b)
            # print("*"*30)
            curr_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(), b.reshape(1,-1).flatten(), c), axis=0) # this is 3 + 7*7 size vector.  all scaled in range 0..1      
            curr_state.astype('float32')
            stacked_state = np.append(stacked_state[1:], [curr_state], axis=0)
            experience = (prev_state, action, reward, stacked_state)
            curr_car_exp.append(experience)
    experiences['experience'].append(curr_car_exp)
out_name = os.path.join(os.getcwd(), f'rand_car_experience_dump_{time.time()}.pkl')
with open(out_name, 'wb+') as f:
    pkl.dump(experiences, f)
print('done')
