from data_paths import dqn_roots, human_roots, human_designs, hybrid_roots, hybrid_designs, bo_roots, bo_designs, anonymous_keys, default_design, default_roots
from betterplotter import get_filenames, get_matrix, get_rewards
from pilotsessions import sessions,users
import json
import pickle as pkl
import itertools
import glob
import pymongo
import datetime
from pymongo import MongoClient, ReturnDocument
from config import Config
from bson.objectid import ObjectId
import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


output_dict = dict()
output_list = list()

dqn_reward_files = [get_filenames(f,"*/total_rewards.pkl") for f in dqn_roots]
# default_rewards = get_matrix(default_reward_files)
dqn_rewards = [get_rewards(f[:3]) for f in dqn_reward_files]

# get some friendly names for keys
with open('flaskapp/nfl_names.csv') as infile:
    friendlynames = list(csv.reader(infile))

# get the db'ed results, then get the ones stored in files
client = MongoClient('localhost', 27017)
db = client[Config.DB_NAME]
experiments = db.experiments
all_sessions = db.sessions
pilots = dict()
for s in sessions:
    sess = all_sessions.find_one({"_id":ObjectId(s)})
    pilots[sess["user_id"]] = sess

curated_pilots = []
with open('flaskapp/output.csv') as infile:
    pilot_csv = csv.reader(infile)
    for p in pilot_csv:
        curated_pilots.append(p)

low_hum = ['b0e35b9e8db847d992fa81afa8851753',
'540d7a5f797745c3beeababa9048d930',
'e6900ed30d77497a97b8b9800d3becdf',
'2a532da3d761421890cc5de28b3ff2f3',
'a3476ba7c278432db0315eda9546b7a4',
'9a5f5d937d79438daa2b52cb4ce26216',
'c79f3969753e4d91a9cb88d4382c9ca8']

high_hum = ['b11c5f8063d647b1aa73cb00eab176e0',
'1b7db0cc7094434b85c284547269f99c',
'786f2281540e4e468fca9d1a74df5c38',
'b00a73908f3147b5b35e90936134a77f',
'b15a47a3828c43d79fa74ca0cffdeb53']

for p in curated_pilots:
    pilots[p[0]]['avg100'] = float(p[1])


test_drive_low = [[p[0] for p in pilots[u]['tested_results']] for u in low_hum]
test_drive_high = [[p[0] for p in pilots[u]['tested_results']] for u in high_hum]
training_reward_low = np.array([pilots[u]['avg100'] for u in low_hum])
training_reward_high = np.array([pilots[u]['avg100'] for u in high_hum])
avg_test_drive_low = np.array([np.mean(t) for t in test_drive_low])
avg_test_drive_high = np.array([np.mean(t) for t in test_drive_high])
avg_dropoff_low = -(training_reward_low-avg_test_drive_low)
avg_dropoff_high = -(training_reward_high-avg_test_drive_high)
n_test_drive_low = np.array([len(t) for t in test_drive_low])
n_test_drive_high = np.array([len(t) for t in test_drive_high])
# plt.boxplot([avg_test_drive_low,avg_test_drive_high])
# plt.title('Test Drive Scores Split Across Baseline Performance')
# plt.ylabel('Average Test Drive Score')
# plt.xticks([1,2],['Worse than Baseline Cars', 'Better than Baseline Cars'])
# plt.show()

# plt.boxplot([avg_dropoff_low,avg_dropoff_high])
# plt.title('Dropoff from Test Drive to Training')
# plt.ylabel('Avg Test Drive - Training Reward')
# plt.xticks([1,2],['Worse than Baseline Cars', 'Better than Baseline Cars'])
# plt.show()


test_drives = [[p[0] for p in pilots[u]['tested_results']] for u in users]
n_test_drives = np.array([len(t) for t in test_drives])
avg_test_drive = np.array([np.mean(t) for t in test_drives])
last_test_drive = np.array([t[-1] for t in test_drives])
training_reward = np.array([pilots[u]['avg100'] for u in users])
flattened_test_drives = []
for t in test_drives:
    for i in t:
        flattened_test_drives.append(i)
# avg_test_drive = np.mean(i)

# plt.boxplot([n_test_drives, n_test_drive_low, n_test_drive_high])
# plt.title('Number of Designs Tested During Study')
# plt.xticks([1,2,3], ['Overall','Worse than Baseline', "Better than Baseline"])
# plt.show()


# slope, intercept, r_value, p_value, std_err = stats.linregress(avg_test_drive, training_reward)
# plt.plot(avg_test_drive, training_reward, 'o')
# plt.plot(avg_test_drive, intercept + slope*avg_test_drive, 'r')
# plt.xlabel("Average Test Drive Score")
# plt.ylabel("Average Training Reward over last 100 Episodes")
# plt.text(400,200, f'R^2={r_value**2:.3f}, p={p_value:.3f}')
# plt.show()

# x = avg_test_drive-training_reward
# slope, intercept, r_value, p_value, std_err = stats.linregress(training_reward, x)
# plt.plot(training_reward, x,'o')
# plt.plot(training_reward, intercept + slope*training_reward, 'r')
# plt.ylabel("Dropoff from Average Test Drive to Training Reward")
# plt.xlabel("Average Training Reward over last 100 Episodes")
# plt.text(400,200, f'R^2={r_value**2:.3f}, p={p_value:.3f}')
# plt.show()

# x = avg_test_drive-training_reward
# slope, intercept, r_value, p_value, std_err = stats.linregress(avg_test_drive, x)
# plt.plot(avg_test_drive, x,'o')
# plt.plot(avg_test_drive, intercept + slope*avg_test_drive, 'r')
# plt.ylabel("Dropoff from Average Test Drive to Training Reward")
# plt.xlabel("Average Training Reward over last 100 Episodes")
# plt.text(400,200, f'R^2={r_value**2:.3f}, p={p_value:.3f}')
# plt.show()

# import pdb; pdb.set_trace()



finished = list(experiments.find({"finished_training": {"$exists": True}}))


def get_type_key(experiment_type):
    if experiment_type== "human":
        return "human"
    if experiment_type == "bo_hybrid_select_features" or experiment_type == "hybrid_hum_select":
        return "hybrid_hum_select"    # human selects features to optimize
    if experiment_type == "full_bo_on_redesign":
        return "hybrid_hum_init"    # full bo on human's design
    if experiment_type == "full_bo":
        return "bo"
    if experiment_type == "bo_hybrid_select_features_start_from_redesign":
        return "hybrid_hum_init_select"    # human selects features to optimize but start from human's design
    if experiment_type == "bo_hybrid_select_features_by_confidence_start_from_user_design":
        return "hybrid_hum_confidence"    # human selects features NOT to optimize by confidence
    if experiment_type == "benchmark":
        return "benchmark"
    else:
        raise ValueError(f"Unexpected Experiment Type {experiment_type}")




with open('flaskapp/d3plots/car_list.json') as infile:
    all_cars = json.load(infile)

hum = []
bo = []
hybrid_hum_select = []
hybrid_hum_init = []
hybrid_hum_init_select = []
hybrid_hum_confidence = []
hybrid  = []
benchmarks = []
# this is smart--all the cars are in the carlist
# for c in all_cars:
#     if c["type"] == "human":
#         hum.append(c)
#     if c["type"] == "bo":
#         bo.append(c)
#     if c["type"] == "hybrid_hum_select":
#         hybrid_hum_select.append(c)
#     if c["type"] == "hybrid_hum_init":
#         hybrid_hum_init.append(c)
#     if c["type"] == "hybrid_hum_init_select":
#         hybrid_hum_init_select.append(c)
#     if c["type"] == "hybrid_hum_confidence":
#         hybrid_hum_confidence.append(c)
#     if "hybrid" in c["type"]:
#         hybrid.append(c)
# import pdb; pdb.set_trace()

typekeys = ["human", "bo", "hybrid_hum_select", "hybrid_hum_init", "hybrid_hum_init_select", "hybrid_hum_confidence","benchmark"]
test_drive_dict = {k:{u:[] for u in users+["bayesopt_only", "benchmark"]} for k in typekeys}
MAX_FREQ = 5 * 10 # accepting 10 test drives for 5 agents each
PRETRAIN = False
def build_test_drive_dict(exp_type,user_id,experiment,pre_train=False):
    # if exp_type == 'bo':
    #     if 'bo_designs' in experiment.keys() and len(experiment['bo_designs']) > 50:
    #         return
    #     if 'tested_design' in experiment.keys() and len(experiment['tested_design']) > 50:
    #         return
    cur_samples = test_drive_dict[exp_type][user_id]
    if pre_train:
        if 'pre_train_test_drive_results' not in experiment.keys():
            return
        new_samples = experiment['pre_train_test_drive_results']
    else:
        new_samples = experiment['final_test_drive_results']
    for trained_agent in new_samples:
        for sample in trained_agent:
            if len(cur_samples) < MAX_FREQ or exp_type in ['bo','benchmark']:
                cur_samples.append(sample)
            else:
                test_drive_dict[exp_type][user_id] = cur_samples
                return
    test_drive_dict[exp_type][user_id] = cur_samples
    return

def print_test_drive_dict():
    for k,v in test_drive_dict.items():
        print(k)
        for u,samples in v.items():
            print(f'\t{u}: {len(samples)}')

def get_flattened_test_drive_dict():
    flattened_dict = dict()
    for k,v in test_drive_dict.items():
        flattened_samples = []
        for samples in v.values():
            flattened_samples += samples
        flattened_dict[k] = flattened_samples
    return flattened_dict


for f in finished:
    if get_type_key(f["experiment_type"]) == "human":
        # hum.append(f)
        build_test_drive_dict("human",f["user_id"],f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "bo":
        # bo.append(f)
        build_test_drive_dict("bo","bayesopt_only",f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "hybrid_hum_select":
        # hybrid_hum_select.append(f)
        build_test_drive_dict("hybrid_hum_select",f["user_id"],f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "hybrid_hum_init":
        # hybrid_hum_init.append(f)
        build_test_drive_dict("hybrid_hum_init",f["user_id"],f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "hybrid_hum_init_select":
        # hybrid_hum_init_select.append(f)
        build_test_drive_dict("hybrid_hum_init_select",f["user_id"],f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "hybrid_hum_confidence":
        # hybrid_hum_confidence.append(f)
        build_test_drive_dict("hybrid_hum_confidence",f["user_id"],f,PRETRAIN)
    if get_type_key(f["experiment_type"]) == "benchmark":
        # benchmarks.append(f)
        build_test_drive_dict("benchmark","benchmark",f,PRETRAIN)
    # if "hybrid" in get_type_key(f["experiment_type"]):
        # hybrid.append(f)

print_test_drive_dict()
test_drives = get_flattened_test_drive_dict()
import pdb; pdb.set_trace()

# avg_hum = np.array([np.mean(c['results'][0][-100:]) for c in hum])
# avg_bo = np.array([np.mean(c['results'][0][-100:]) for c in bo])
# avg_hybrid = np.array([np.mean(c['results'][0][-100:]) for c in hybrid])
# avg_hybrid_hum_select = np.array([np.mean(c['results'][0][-100:]) for c in hybrid_hum_select])
# avg_hybrid_hum_init = np.array([np.mean(c['results'][0][-100:]) for c in hybrid_hum_init])
# avg_hybrid_hum_init_select = np.array([np.mean(c['results'][0][-100:]) for c in hybrid_hum_init_select])
# avg_hybrid_hum_confidence = np.array([np.mean(c['results'][0][-100:]) for c in hybrid_hum_confidence])

# avg_hum = np.array([np.mean(c['final_test_drive_results']) for c in hum])
# avg_bo = np.array([np.mean(c['final_test_drive_results']) for c in bo])
# avg_hybrid = np.array([np.mean(c['final_test_drive_results']) for c in hybrid])
# avg_hybrid_hum_select = np.array([np.mean(c['final_test_drive_results']) for c in hybrid_hum_select])
# avg_hybrid_hum_init = np.array([np.mean(c['final_test_drive_results']) for c in hybrid_hum_init])
# avg_hybrid_hum_init_select = np.array([np.mean(c['final_test_drive_results']) for c in hybrid_hum_init_select])
# avg_hybrid_hum_confidence = np.array([np.mean(c['final_test_drive_results']) for c in hybrid_hum_confidence])

# avg_hum = np.array([np.mean(c['pre_train_test_drive_results']) for c in hum])
# avg_bo = np.array([np.mean(c['pre_train_test_drive_results']) for c in bo])
# avg_hybrid = np.array([np.mean(c['pre_train_test_drive_results']) for c in hybrid])
# avg_hybrid_hum_select = np.array([np.mean(c['pre_train_test_drive_results']) for c in hybrid_hum_select])
# avg_hybrid_hum_init = np.array([np.mean(c['pre_train_test_drive_results']) for c in hybrid_hum_init])
# avg_hybrid_hum_init_select = np.array([np.mean(c['pre_train_test_drive_results']) for c in hybrid_hum_init_select])
# avg_hybrid_hum_confidence = np.array([np.mean(c['pre_train_test_drive_results']) for c in hybrid_hum_confidence])
# benchmark = np.mean([np.mean(c['pre_train_test_drive_results']) for c in hum])

# avg_hum = np.concatenate([np.array(c['final_test_drive_results']) for c in hum]).flatten()
# avg_bo = np.concatenate([np.array(c['final_test_drive_results']) for c in bo]).flatten()
# avg_hybrid = np.concatenate([np.array(c['final_test_drive_results']) for c in hybrid]).flatten()
# avg_hybrid_hum_select = np.concatenate([np.array(c['final_test_drive_results']) for c in hybrid_hum_select]).flatten()
# avg_hybrid_hum_init = np.concatenate([np.array(c['final_test_drive_results']) for c in hybrid_hum_init]).flatten()
# avg_hybrid_hum_init_select = np.concatenate([np.array(c['final_test_drive_results']) for c in hybrid_hum_init_select]).flatten()
# avg_hybrid_hum_confidence = np.concatenate([np.array(c['final_test_drive_results']) for c in hybrid_hum_confidence]).flatten()
# benchmark = np.concatenate([np.array(c['final_test_drive_results']) for c in benchmarks]).flatten()
benchmark = test_drives['benchmark']
avg_hum = test_drives['human']
avg_bo = test_drives['bo']
avg_hybrid_hum_select = test_drives['hybrid_hum_select']
avg_hybrid_hum_init = test_drives['hybrid_hum_init']
avg_hybrid_hum_init_select = test_drives['hybrid_hum_init_select']
avg_hybrid_hum_confidence = test_drives['hybrid_hum_confidence']
avg_hybrid = avg_hybrid_hum_select + avg_hybrid_hum_init + avg_hybrid_hum_init_select + avg_hybrid_hum_confidence
print(np.mean(benchmark))
fig,ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
plt.violinplot([benchmark, avg_hum, avg_bo, avg_hybrid])
# plt.axhline(y=np.mean(benchmark), linestyle='--')
# plt.text(3.4,397,f'Benchmark')
plt.xticks([1,2,3,4],[f'No Redesign n={len(benchmark)}', f"Human n={len(avg_hum)}", f"BayesOpt n={len(avg_bo)}", f"Hybrid n={len(avg_hybrid)}"], fontsize=18)
plt.ylabel("Test Drive Episode Reward",fontsize=18)
plt.title("Driver Performance after Training on New Designs", fontsize=20)
plt.show()

# plt.violinplot([avg_hum, avg_bo, avg_hybrid_hum_select, avg_hybrid_hum_init, avg_hybrid_hum_init_select, avg_hybrid_hum_confidence])
# plt.xticks([1,2,3,4,5,6],["Human", "BayesOpt", "Human-Selected Features", "Human Initial Design", "Human-Selected Features and Design", "Features by Confidence"])
fig,ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
plt.violinplot([benchmark, avg_hum, avg_bo, avg_hybrid_hum_select, avg_hybrid_hum_init, avg_hybrid_hum_init_select, avg_hybrid_hum_confidence])
plt.xticks([1,2,3,4,5,6,7,8],["No Redesign","Human", "BayesOpt", "Hum-Select", "Hum-Init", "Human-Select + Init", "Hum-Confidence"])
# plt.axhline(y=np.mean(benchmark), linestyle='--')
# plt.text(6.4,397,f'Benchmark')
plt.ylabel("Test Drive Episode Reward",fontsize=18)
plt.title("Driver Performance after Training on New Designs", fontsize=20)
plt.show()
import pdb; pdb.set_trace()



import csv
outlist = []
for r in avg_hum:
    outlist.append(("hum",r))
for r in avg_bo:
    outlist.append(("bo",r))
for r in avg_hybrid:
    outlist.append(("hybrid",r))
for r in avg_hybrid_hum_select:
    outlist.append(("hybrid_hum_select",r))
for r in avg_hybrid_hum_init:
    outlist.append(("hybrid_hum_init",r))
for r in avg_hybrid_hum_init_select:
    outlist.append(("hybrid_hum_init_select",r))
for r in avg_hybrid_hum_confidence:
    outlist.append(("hybrid_hum_confidence",r))
for r in benchmark:
    outlist.append(("benchmark",r))

with open('chopshopjasp_pretrain.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(("design_type","test_drive"))
    writer.writerows(outlist)