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
alap_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b15a47a3828c43d79fa74ca0cffdeb53/bo_sessions/5f875228b9d252ffa7e54be9/policy_training/1014_2025"
alap_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b15a47a3828c43d79fa74ca0cffdeb53/5f8765c68a4b3890d9075c25/training/1014_1655"
full_bo_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions_all_features/5f6d02c2673446edf0d88f1c/policy_training/1014_0656"
full_bo_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/9a5f5d937d79438daa2b52cb4ce26216/bo_sessions_all_features/5f87bcceb9d252ffa7e54beb/policy_training/1021_1511"

amit_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions/5f6d02c2673446edf0d88f1c/policy_training/1001_0528"
amit_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/5f6d6ef5e0159310a95af84f/training/0925_0015"

dan_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/e6900ed30d77497a97b8b9800d3becdf/bo_sessions/5f6cd5e2d8a6d9430d007bf3/policy_training/1014_2032"
dan_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/e6900ed30d77497a97b8b9800d3becdf/5f8765648e0a6804d8915cbf/training/1014_1653"

swati_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b0e35b9e8db847d992fa81afa8851753/bo_sessions/5f87a26bb9d252ffa7e54bea/policy_training/1015_0145"
swati_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b0e35b9e8db847d992fa81afa8851753/5f87cdcab5adc0c1f20c498f/training/1015_0019"

yuhan_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/9a5f5d937d79438daa2b52cb4ce26216/bo_sessions/5f87bcceb9d252ffa7e54beb/policy_training/1015_0252"
yuhan_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/9a5f5d937d79438daa2b52cb4ce26216/5f87ce34c50d4de5498fed50/training/1015_0021"

jihyun_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/bo_sessions/5f8df42ad5c68de5e8185f59/policy_training/1020_0233"
jihyun_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/5f8e652f9609b767c12fed5c/training/1020_0018"
jihyun_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/session_5f8df42ad5c68de5e8185f59_human_design_retrain/training/1021_0111"
jihyun_human_root = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/session_5f8df42ad5c68de5e8185f59_human_design_retrain/training_0/1021_1230'

anna_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/2a532da3d761421890cc5de28b3ff2f3/bo_sessions/5f90983b8dfbb43775149641/policy_training/1021_1946"
anna_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/2a532da3d761421890cc5de28b3ff2f3/5f90a6c53bb7fa72ca1bd413/training/1021_1723"

nikhil_hybrid_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/bo_sessions/5f90fdee8dfbb43775149642/policy_training/1022_0356"
nikhil_human_root = "/home/dev/scratch/cars/carracing_clean/flaskapp/static/b00a73908f3147b5b35e90936134a77f/5f911ce7769413210aa2397d/training/1022_0147"

user_id = "b00a73908f3147b5b35e90936134a77f" #nikhil
session_override_id = "5f90fdee8dfbb43775149642" #nikhil

##^^caution, this has densities

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
files_alap_hybrid = glob.glob(f'{alap_hybrid_root}/*rewards_flask.pkl')
files_alap_human = glob.glob(f'{alap_human_root}/*rewards_flask.pkl')
files_dan_hybrid = glob.glob(f'{dan_hybrid_root}/*rewards_flask.pkl')
files_dan_human = glob.glob(f'{dan_human_root}/*rewards_flask.pkl')
files_amit_hybrid = glob.glob(f'{amit_hybrid_root}/*rewards_flask.pkl')
files_amit_human = glob.glob(f'{amit_human_root}/*rewards_flask.pkl')
files_swati_hybrid = glob.glob(f'{swati_hybrid_root}/*rewards_flask.pkl')
files_swati_human = glob.glob(f'{swati_human_root}/*rewards_flask.pkl')
files_yuhan_hybrid = glob.glob(f'{yuhan_hybrid_root}/*rewards_flask.pkl')
files_yuhan_human = glob.glob(f'{yuhan_human_root}/*rewards_flask.pkl')
files_jihyun_hybrid = glob.glob(f'{jihyun_hybrid_root}/*rewards_flask.pkl')
files_jihyun_human = glob.glob(f'{jihyun_human_root}/*rewards_flask.pkl')
files_anna_hybrid = glob.glob(f'{anna_hybrid_root}/*rewards_flask.pkl')
files_anna_human = glob.glob(f'{anna_human_root}/*rewards_flask.pkl')
files_nikhil_hybrid = glob.glob(f'{nikhil_hybrid_root}/*rewards_flask.pkl')
files_nikhil_human = glob.glob(f'{nikhil_human_root}/*rewards_flask.pkl')

files_full_bo = glob.glob(f'{full_bo_root}/*rewards_flask.pkl')



import pdb; pdb.set_trace()
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
files_amit_hybrid.sort()
files_amit_human.sort()
files_dan_hybrid.sort()
files_dan_human.sort()
files_alap_hybrid.sort()
files_alap_human.sort()
files_swati_hybrid.sort()
files_swati_human.sort()
files_yuhan_hybrid.sort()
files_yuhan_human.sort()
files_jihyun_hybrid.sort()
files_jihyun_human.sort()
files_anna_hybrid.sort()
files_anna_human.sort()
files_nikhil_hybrid.sort()
files_nikhil_human.sort()
files_full_bo.sort()


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

rewards_alap_hybrid = []
for f in files_alap_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_alap_hybrid.append(d)

rewards_alap_human = []
for f in files_alap_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_alap_human.append(d)

rewards_amit_hybrid = []
for f in files_amit_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_amit_hybrid.append(d)

rewards_amit_human = []
for f in files_amit_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_amit_human.append(d)

rewards_dan_hybrid = []
for f in files_dan_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_dan_hybrid.append(d)

rewards_dan_human = []
for f in files_dan_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_dan_human.append(d)

rewards_swati_hybrid = []
for f in files_swati_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_swati_hybrid.append(d)

rewards_swati_human = []
for f in files_swati_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_swati_human.append(d)

rewards_yuhan_hybrid = []
for f in files_yuhan_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_yuhan_hybrid.append(d)

rewards_yuhan_human = []
for f in files_yuhan_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_yuhan_human.append(d)

rewards_jihyun_hybrid = []
for f in files_jihyun_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_jihyun_hybrid.append(d)

rewards_jihyun_human = []
for f in files_jihyun_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_jihyun_human.append(d)

rewards_anna_hybrid = []
for f in files_anna_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_anna_hybrid.append(d)

rewards_anna_human = []
for f in files_anna_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_anna_human.append(d)

rewards_nikhil_hybrid = []
for f in files_nikhil_hybrid:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_nikhil_hybrid.append(d)

rewards_nikhil_human = []
for f in files_nikhil_human:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_nikhil_human.append(d)

rewards_full_bo = []
for f in files_full_bo:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards_full_bo.append(d)


# plt.plot(running_avg(rewards_non_adapt),label='non-adaptive joint opt')
# plt.plot(running_avg(rewards_adapt), label='Driver Performance While Training')
plt.plot(running_avg(rewards_design_input), label='DQN Only')
plt.plot(running_avg(rewards_design_input[:250]+rewards_full_bo), label='BayesOpt Only')
plt.plot(running_avg(rewards_design_input[:250]+rewards_anna_hybrid), label='BayesOpt Hybrid')
plt.plot(running_avg(rewards_design_input[:250]+rewards_anna_human), label='Human Only')
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
