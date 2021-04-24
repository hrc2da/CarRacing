import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
from matplotlib import pyplot as plt
from config import Config
import numpy as np

client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
db = client[Config.DB_NAME]
sessions_collection = db.sessions

def running_avg(totalrewards, window=100):
  # return totalrewards
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
      running_avg[t] = np.mean(totalrewards[max(0, t-window):(t+1)])
  return running_avg

experiments = db.experiments
control_test_drives_h2 = list(experiments.find({"experiment_type":"h2control"}))
human_test_drives_h2 = list(experiments.find({"experiment_type":"h2human"}))

control_rewards = [running_avg(e['training_rewards']) for e in control_test_drives_h2]
human_rewards = [running_avg(e['training_rewards']) for e in human_test_drives_h2]

for c in control_rewards:
    plt.plot(c, color='orange')

for h in human_rewards:
    plt.plot(h, color='blue')

plt.show()

