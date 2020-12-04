from config import Config
import glob
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
import os

user_id = 'a3476ba7c278432db0315eda9546b7a4'

reward_files = glob.glob(f'{Config.FILE_PATH}/{user_id}/*/training/*/*_rewards*.pkl')
reward_files.sort()
print(reward_files)
rewards = []
for f in reward_files:
  with open(f, 'rb') as infile:
    data = pkl.load(infile)
    for d in data:
      rewards.append(d)

def running_avg(totalrewards):
  # return totalrewards
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
      running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
  return running_avg


plt.plot(running_avg(rewards))
# plt.text(20,430,"Driver's Cumulative Performance While Learning to Drive", fontsize=15)
plt.xlabel('Training Episode')
plt.ylabel('Episode Score (Higher is Better)')
plt.ylim(-100,600)
for i in (250,500,750,1000):
  plt.axvline(i,color='0.5',linestyle='--')
# plt.text(150,410,'First Redesign')
# plt.text(380,410,'Second Redesign')
# plt.text(650,410,'Third Redesign')
new_session_dir = f'{Config.FILE_PATH}/{user_id}/'
plt.savefig(os.path.join(new_session_dir,'reward_plot2.png'), bbox_inches='tight')

