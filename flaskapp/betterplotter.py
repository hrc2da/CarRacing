import numpy as np
import glob
import pickle as pkl
from matplotlib import pyplot as plt
from data_paths import human_roots, hybrid_roots
import itertools


def get_filenames(root_path,pickle_glob='*rewards_flask.pkl'):
    files = glob.glob(f'{root_path}/{pickle_glob}')
    files.sort()
    return files

def get_rewards(files):
    rewards = []
    for f in files:
        with open(f, 'rb') as infile:
            data = pkl.load(infile)
            for d in data:
                rewards.append(d)
    return rewards

def get_distribution(files):
    # treat each file as a separate run, don't append them
    rewards = []
    for f in files:
        with open(f, 'rb') as infile:
            data = pkl.load(infile)
            rewards.append(data)
    mean = np.mean(rewards,0)
    std = np.std(rewards,0)
    return mean, std

def get_matrix(files):
    rewards = []
    for f in files:
        with open(f, 'rb') as infile:
            data = pkl.load(infile)
            rewards.append(data)
    return np.array(rewards)

def running_avg(totalrewards):
  # return totalrewards
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
      running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
  return running_avg


# file_roots = ["/home/dev/scratch/cars/carracing_clean/joe_dqn_only_0","/home/dev/scratch/cars/carracing_clean/joe_dqn_only_1","/home/dev/scratch/cars/carracing_clean/joe_dqn_only_2"]

if __name__ == '__main__':

    human_files = list(itertools.chain(*[get_filenames(f,"*rewards_flask.pkl") for f in list(itertools.chain(*list(human_roots.values())))]))
    hum_mean,hum_std = get_distribution(human_files)
    hum_upper_bound = running_avg(hum_mean+hum_std)
    hum_lower_bound = running_avg(hum_mean-hum_std)

    hybrid_files = list(itertools.chain(*[get_filenames(f,"*rewards_flask.pkl") for f in list(itertools.chain(*list(hybrid_roots.values())))]))
    hyb_mean,hyb_std = get_distribution(hybrid_files)
    hyb_upper_bound = running_avg(hyb_mean+hyb_std)
    hyb_lower_bound = running_avg(hyb_mean-hyb_std)


    rewards_dqn_0 = get_rewards(get_filenames("/home/dev/scratch/cars/carracing_clean/joe_dqn_only_0","*/total_rewards.pkl"))
    ## 1 is the one given to users
    rewards_dqn_1 = get_rewards(get_filenames("/home/dev/scratch/cars/carracing_clean/joe_dqn_only_1","*/total_rewards.pkl"))
    rewards_dqn_2 = get_rewards(get_filenames("/home/dev/scratch/cars/carracing_clean/joe_dqn_only_2","*/total_rewards.pkl"))
    plt.plot(running_avg(rewards_dqn_1), label='DQN Only')
    upper_bound = running_avg(np.mean([rewards_dqn_0,rewards_dqn_1,rewards_dqn_2],0)+np.std([rewards_dqn_0,rewards_dqn_1,rewards_dqn_2],0))
    lower_bound = running_avg(np.mean([rewards_dqn_0,rewards_dqn_1,rewards_dqn_2],0)-np.std([rewards_dqn_0,rewards_dqn_1,rewards_dqn_2],0))
    plt.fill_between(range(len(upper_bound)), lower_bound,upper_bound, alpha=0.2)

    plt.plot(range(250,500),running_avg(rewards_dqn_1[:250]+list(hum_mean))[250:], label='Human Designs')
    plt.fill_between(range(250,250+len(hum_upper_bound)), hum_lower_bound, hum_upper_bound, alpha=0.2)


    plt.plot(range(250,500),running_avg(rewards_dqn_1[:250]+list(hyb_mean))[250:], label='Hybrid Designs')
    plt.fill_between(range(250,250+len(hyb_upper_bound)), hyb_lower_bound, hyb_upper_bound, alpha=0.2)

    # plt.plot(running_avg(rewards_design_input[:250]+rewards_full_bo), label='BayesOpt Only')
    # plt.plot(running_avg(rewards_design_input[:250]+rewards_anna_hybrid), label='BayesOpt Hybrid')
    # plt.plot(running_avg(rewards_design_input[:250]+rewards_anna_human), label='Human Only')
    # plt.plot(running_avg(rewards_anna), label='human supervisor no warmup')
    # plt.plot(running_avg(rewards_jihyun), label='human supervisor 50 ep warmup')
    # plt.plot(running_avg(rewards_human), label='human designer')
    # plt.plot(range(len(rewards_human),len(rewards_human)+len(rewards_takeover)),running_avg(rewards_takeover), label='bo takeover from human')
    # plt.plot(running_avg(rewards_adapt_50rf), label='adaptive 25_50_40 Replay Freq 50')
    # plt.plot(running_avg(rewards_adapt_10rf), label='adaptive 25_50_40 Replay Freq 10')
    # plt.plot(running_avg(rewards),label='policy-only opt', color='0.5')
    # plt.plot(running_avg(rewards_adapt_20), label='adaptive joint opt 20 ep warmup')
    # plt.plot(running_avg(rewards_adapt_MAB), label='adaptive joint opt MAB feature select')
    # plt.text(20,430,"Driver's Performance While Learning to Drive", fontsize=15)
    plt.xlabel('Training Episode')
    plt.ylabel('Episode Score (Higher is Better)')
    # plt.axvline(len(rewards_human),color='0.5',linestyle='--')
    for i in (250,500,750,1000):
        plt.axvline(i,color='0.5',linestyle='--')
    plt.xlim(0,500)
    plt.ylim(-100,800)
    # plt.text(150,410,'First Redesign')
    # plt.text(380,410,'Second Redesign')
    # plt.text(650,410,'Third Redesign')
    plt.legend()
    plt.show()