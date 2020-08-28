import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

with open('/home/dev/scratch/cars/carracing_clean/codesigntrials/set1/avg_dqn_4_seq_rewards_every50_total_rewards_flask.pkl', 'rb') as infile:
    set1 = pkl.load(infile)

with open('/home/dev/scratch/cars/carracing_clean/codesigntrials/set2/avg_dqn_4_seq_rewards_every50_total_rewards_flask.pkl', 'rb') as infile:
    set2 = pkl.load(infile)

total = np.concatenate((set1,set2))

with open('/home/dev/scratch/cars/carracing_clean/codesigntrials/set3/200_250.txt', 'r') as infile:
    for line in infile:
        total = np.append(total,float(line))

#with open('/home/dev/scratch/cars/carracing_clean/codesigntrials/no_redesign_same_eps_starting_from_100.pkl', 'rb') as infile:
with open('/home/dev/scratch/cars/carracing_clean/codesigntrials/baseline.txt', 'r') as infile:
    baseline = np.array([])
    for line in infile:
        baseline = np.append(baseline, float(line))

print(baseline.shape)
print(total.shape)

def plot_running_avg(totalrewards, baseline):
    N = len(totalrewards)
    bN = len(baseline)
    running_baseline = np.empty(bN)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    for t in range(bN):
        running_baseline[t] = baseline[max(0, t-100):(t+1)].mean()
    t = np.arange(100,250)
    plt.plot(t,running_baseline, label="Baseline Car")
    plt.plot(t[:-3],running_avg, label="Redesigned Car")
    plt.title("Running Average Score")
    plt.legend(loc=0)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    fname = os.path.join(os.getcwd(), "running_avg_reward.png")
    plt.savefig(fname)

plot_running_avg(total,baseline)