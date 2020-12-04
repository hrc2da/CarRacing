import glob
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
# train_adapt_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_50_ep_warmup" 
train_adapt_root = "/home/dev/scratch/cars/carracing_clean/joe_dqn_only_eps_fixed" 
train_adapt_50rf_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_25_50_40_50rf" 
train_adapt_10rf_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_25_50_40_10rf" 
train_adapt_20_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track" 
train_MAB_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_MAB"
train_non_adapt_root = "/home/dev/scratch/cars/carracing_clean/train_logs_non_adaptive_random_track"
train_root = "/home/dev/scratch/cars/carracing_clean/train_logs_random_track"
human_root = "/home/dev/scratch/cars/carracing_clean/train_logs"
takeover_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_scratch_takeover"
anna_root = "/home/dev/scratch/cars/carracing_clean/train_logs_video_dump_50rf_testing2"
jihyun_root = "/home/dev/scratch/cars/carracing_clean/train_logs_video_dump_50rf_testing_jihyun"
design_input_root = "/home/dev/scratch/cars/carracing_clean/joe_dqn_only_1"


# train_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_50_ep_warmup"

# files23 = glob.glob(f'{train_root}/0723*/total_rewards.pkl')

files = glob.glob(f'{train_root}/*/total_rewards.pkl')
files_adapt = glob.glob(f'{train_adapt_root}/*/total_rewards.pkl')
files_adapt_20 = glob.glob(f'{train_adapt_20_root}/*/total_rewards.pkl')
files_adapt_50rf = glob.glob(f'{train_adapt_50rf_root}/*/total_rewards.pkl')
files_adapt_10rf = glob.glob(f'{train_adapt_10rf_root}/*/total_rewards.pkl')
files_MAB = glob.glob(f'{train_MAB_root}/*/total_rewards.pkl')
files_non_adapt = glob.glob(f'{train_non_adapt_root}/*/total_rewards.pkl')
files_human = glob.glob(f'{human_root}/0730*/avg_dqn_scratch_driver0.h5_rewards_flask.pkl')
files_takeover = glob.glob(f'{takeover_root}/*/avg_dqn_scratch_driver0.h5_rewards_flask.pkl')
files_anna = glob.glob(f'{anna_root}/*/total_rewards.pkl')
files_jihyun = glob.glob(f'{jihyun_root}/*/total_rewards.pkl')
files_design_input = glob.glob(f'{design_input_root}/*/total_rewards.pkl')

files.sort()
files_adapt.sort()
files_adapt_20.sort()
files_MAB.sort()
files_non_adapt.sort()
files_human.sort()
files_takeover.sort()
files_adapt_50rf.sort()
files_adapt_10rf.sort()
files_anna.sort()
files_jihyun.sort()
files_design_input.sort()

def running_avg(totalrewards):
  # return totalrewards
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
      running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
  return running_avg





rewards = []
for f in files:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards.append(d)

rewards_adapt = []
for f in files_adapt:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_adapt.append(d)

rewards_adapt_20 = []
for f in files_adapt_20:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_adapt_20.append(d)

rewards_adapt_MAB = []
for f in files_MAB:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_adapt_MAB.append(d)

rewards_non_adapt = []
for f in files_non_adapt:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_non_adapt.append(d)

rewards_human = []
for f in files_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_human.append(d)

rewards_takeover = []
for f in files_takeover:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_takeover.append(d)

rewards_adapt_50rf = []
for f in files_adapt_50rf:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_adapt_50rf.append(d)

rewards_adapt_10rf = []
for f in files_adapt_10rf:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_adapt_10rf.append(d)

rewards_anna = []
for f in files_anna:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_anna.append(d)
rewards_jihyun = []
for f in files_jihyun:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_jihyun.append(d)

rewards_design_input = []
for f in files_design_input:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_design_input.append(d)


# plt.plot(running_avg(rewards_non_adapt),label='non-adaptive joint opt')
# plt.plot(running_avg(rewards_adapt), label='Driver Performance While Training')
plt.plot(running_avg(rewards_design_input), label='Driver Performance While Training')
# plt.plot(running_avg(rewards_anna), label='human supervisor no warmup')
# plt.plot(running_avg(rewards_jihyun), label='human supervisor 50 ep warmup')
# plt.plot(running_avg(rewards_human), label='human designer')
# plt.plot(range(len(rewards_human),len(rewards_human)+len(rewards_takeover)),running_avg(rewards_takeover), label='bo takeover from human')
# plt.plot(running_avg(rewards_adapt_50rf), label='adaptive 25_50_40 Replay Freq 50')
# plt.plot(running_avg(rewards_adapt_10rf), label='adaptive 25_50_40 Replay Freq 10')
# plt.plot(running_avg(rewards),label='policy-only opt', color='0.5')
# plt.plot(running_avg(rewards_adapt_20), label='adaptive joint opt 20 ep warmup')
# plt.plot(running_avg(rewards_adapt_MAB), label='adaptive joint opt MAB feature select')
plt.text(20,640,"Driver's Performance While Learning to Drive", fontsize=15)
plt.xlabel('Training Episode')
plt.ylabel('Episode Score (Higher is Better)')
# plt.axvline(len(rewards_human),color='0.5',linestyle='--')
for i in (250,500,750,1000):
  plt.axvline(i,color='0.5',linestyle='--')
plt.xlim(0,1000)
plt.ylim(-100,600)
plt.text(150,610,'First Redesign')
plt.text(380,610,'Second Redesign')
plt.text(650,610,'Third Redesign')
# plt.legend()
plt.show()
