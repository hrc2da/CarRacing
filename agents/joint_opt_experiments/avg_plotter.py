import glob
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np

dqn_only_root = "/home/dev/scratch/cars/carracing_clean/joe_dqn_only"
b_dqn_baseline_root = "/home/dev/scratch/cars/carracing_clean/joe_b_dqn_baseline"
b_dqn_clear_root = "/home/dev/scratch/cars/carracing_clean/joe_b_dqn_clear_design"

# train_root = "/home/dev/scratch/cars/carracing_clean/train_logs_adaptive_random_track_50_ep_warmup"

# files23 = glob.glob(f'{train_root}/0723*/total_rewards.pkl')

files_dqn_only = glob.glob(f'{dqn_only_root}_*/*/total_rewards.pkl')
files_b_dqn_baseline = glob.glob(f'{b_dqn_baseline_root}_*/*/total_rewards.pkl')
files_b_dqn_clear = glob.glob(f'{b_dqn_clear_root}_*/*/total_rewards.pkl')

def get_traces(file_list,root,trim=750):
  file_list.sort()
  print(file_list)
  rewards = dict()
  for f in file_list:
    trace_num,segment,path = f.replace(f'{root}_','').split('/')
    with open(f, 'rb') as infile:
      data = pkl.load(infile)
      if trace_num in rewards.keys():
        rewards[trace_num] = np.concatenate((rewards[trace_num],data))
      else:
        rewards[trace_num] = data
  reward_list = list(rewards.values())
  rows_to_drop = []
  for i,row in enumerate(reward_list):
    if len(row) > trim:
      reward_list[i] = row[:trim]
    if len(row) < trim:
      # import pdb; pdb.set_trace()
      rows_to_drop.append(i)
  reward_list = [r for i,r in enumerate(reward_list) if i not in rows_to_drop]
  return np.array(reward_list)

dqn_only_traces = get_traces(files_dqn_only,dqn_only_root,trim=1500)
b_dqn_baseline_traces = get_traces(files_b_dqn_baseline,b_dqn_baseline_root,trim=1250)
b_dqn_clear_traces = get_traces(files_b_dqn_clear,b_dqn_clear_root)
import pdb; pdb.set_trace()

def running_avg(totalrewards):
  # return totalrewards
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
      running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
  return running_avg


def plot_avg_std(traces, label):
  mu = np.mean(traces,0)
  std = np.std(traces,0)
  ra_mu = running_avg(mu)
  ra_std = running_avg(std)

  plt.plot(ra_mu,label=label)
  plt.fill_between(range(len(ra_mu)),ra_mu-ra_std,ra_mu+ra_std, alpha=0.2)

plot_avg_std(dqn_only_traces, 'Policy Only Optimization (n=3)')
plot_avg_std(b_dqn_baseline_traces,'Joint Optimization (n=3)')
# plt.plot(running_avg(np.mean(dqn_only_traces,0)),label='Policy Only Optimization (n=3)')



# plt.plot(running_avg(np.mean(b_dqn_baseline_traces,0)),label='Joint Optimization (n=3)')
# plt.plot(running_avg(np.mean(b_dqn_clear_traces,0)),label='Joint Optimization w/ Design Resets')

# plt.plot(running_avg(rewards_non_adapt),label='non-adaptive joint opt')
# plt.plot(running_avg(rewards_adapt), label='adaptive joint opt')
# plt.plot(running_avg(rewards_design_input), label='design_input joint opt')
# plt.plot(running_avg(rewards_anna), label='human supervisor no warmup')
# plt.plot(running_avg(rewards_jihyun), label='human supervisor 50 ep warmup')
# plt.plot(running_avg(rewards_human), label='human designer')
# plt.plot(range(len(rewards_human),len(rewards_human)+len(rewards_takeover)),running_avg(rewards_takeover), label='bo takeover from human')
# plt.plot(running_avg(rewards_adapt_50rf), label='adaptive 25_50_40 Replay Freq 50')
# plt.plot(running_avg(rewards_adapt_10rf), label='adaptive 25_50_40 Replay Freq 10')
# plt.plot(running_avg(rewards),label='policy-only opt', color='0.5')
# plt.plot(running_avg(rewards_adapt_20), label='adaptive joint opt 20 ep warmup')
# plt.plot(running_avg(rewards_adapt_MAB), label='adaptive joint opt MAB feature select')

plt.xlabel('policy training episode')
plt.ylabel('episode reward')
# plt.axvline(len(rewards_human),color='0.5',linestyle='--')
for i in (250,500,750,1000):
  plt.axvline(i,color='0.5',linestyle='--')
plt.xlim(0,1250)
plt.ylim(-100,600)
plt.legend()
plt.show()
