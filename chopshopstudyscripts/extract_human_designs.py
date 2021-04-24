import csv
import numpy as np
import os

# extract the final human_designs and dump to a csv

import pymongo
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
from pilotsessions import sessions, session2indexdict
from config import Config
from utils import feature_labels, feature_ranges, car2config, config2car, densities, wheelmoment, blacklist, chopshopfeatures2indexlist, feature_labels, featuremask2names

out_dir = 'data'
questions = ["driver_approach","driver_strengths","driver_struggles","driver_design","design_description","design_rationale"]

client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
db = client[Config.DB_NAME]
sessions_collection = db.sessions

session_ids = [ObjectId(s) for s in sessions]
pilot_sessions = list(sessions_collection.find({"_id":{"$in":session_ids}}))

# get the number of designs attempted
num_tested_arr = [0]*len(pilot_sessions)
for s in pilot_sessions:
    sid = str(s['_id'])
    num_tested = len(s['tested_results'])
    num_tested_arr[session2indexdict[sid]] = num_tested
with open(os.path.join(out_dir,'num_designs_attempted.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    for i,nt in enumerate(num_tested_arr):
        writer.writerow([i,nt])


designs = [[str(s['_id'])] + car2config(s['final_design'], correct_negative_widths=True) for s in pilot_sessions]

design_dict = {}
for d in designs:
    design_dict[d[0]] = d[1:]

with open(os.path.join(out_dir,'human_designs.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    culled_feature_labels = [l for i,l in enumerate(feature_labels) if i not in blacklist]
    writer.writerow(culled_feature_labels)
    for d in designs:
        culled_design = [feature for i,feature in enumerate(d) if i not in blacklist]
        writer.writerow(culled_design[1:]) # strip the session id


experiments = db.experiments
control_test_drives_h1 = list(experiments.find({"experiment_type":"h1control"}))
import pdb; pdb.set_trace()
def get_features_modified(design, default_design, labels):
    culled_design = [feature for i,feature in enumerate(design) if i not in blacklist]
    assert len(culled_design) == len(default_design)
    assert len(culled_design) == len(labels)
    modified = [labels[i] for i,feature_value in enumerate(culled_design) if feature_value != default_design[i]]
    return modified



with open(os.path.join(out_dir,'human_modified_features.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    culled_feature_labels = [l for i,l in enumerate(feature_labels) if i not in blacklist]
    modified_features = [[]]*len(designs)
    default_design = car2config(control_test_drives_h1[0]['final_design'])
    culled_default_design = [feature for i,feature in enumerate(default_design) if i not in blacklist]
    for d in designs:
        modified_features[session2indexdict[d[0]]] = get_features_modified(d[1:],culled_default_design,culled_feature_labels)
    for m in modified_features:
        writer.writerow(m)


pilot_test_drives = list(experiments.find({"experiment_type":"h1human","session_id":{"$in":session_ids}}))

pilot_h2_test_drives = list(experiments.find({"experiment_type":"h2human","session_id":{"$in":session_ids}}))


with open(os.path.join(out_dir,'human_design_test_drives.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    culled_feature_labels = ['reward'] + [l for i,l in enumerate(feature_labels) if i not in blacklist] + ['session']
    writer.writerow(culled_feature_labels)
    for exp in pilot_test_drives:
        for sample in exp['test_drive_results']:
            for result in sample:
                d = design_dict[str(exp['session_id'])]
                culled_design = [feature for i,feature in enumerate(d) if i not in blacklist]
                writer.writerow([result] + culled_design + [str(exp['session_id'])])



baselines_arr = np.array([control_run['test_drive_results'][0] for control_run in control_test_drives_h1])
baseline_boxes = [baselines_arr[:,i] for i in range(baselines_arr.shape[1])]
from matplotlib import pyplot as plt
plt.boxplot(baseline_boxes)
plt.show()
baselines = np.mean(baselines_arr, axis=0) # average across runs for each map
print(f"Before: {np.mean(baselines)}")
with open(os.path.join(out_dir,'h1_delta_test_drives.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    culled_feature_labels = ['reward'] + [l for i,l in enumerate(feature_labels) if i not in blacklist] + ['session'] + ['map_num']
    writer.writerow(culled_feature_labels)
    for sess_num, exp in enumerate(pilot_test_drives):
        for sample in exp['test_drive_results']:
            for map_num,result in enumerate(sample):
                d = design_dict[str(exp['session_id'])]
                culled_design = [feature for i,feature in enumerate(d) if i not in blacklist]
                # writer.writerow([result-baselines[map_num]] + culled_design + [str(exp['session_id'])] + [map_num])
                writer.writerow([result-baselines[map_num]] + culled_design + [session2indexdict[str(exp['session_id'])]] + [map_num])


control_test_drives_h2 = experiments.find({"experiment_type":"h2control"})
after_baselines_arr = np.array([control_run['test_drive_results'][0] for control_run in control_test_drives_h2])

after_baseline_boxes = [after_baselines_arr[:,i] for i in range(after_baselines_arr.shape[1])]
from matplotlib import pyplot as plt
plt.boxplot(after_baseline_boxes)
plt.show()
after_baselines = np.mean(after_baselines_arr, axis=0)
import pdb; pdb.set_trace()
print(f"After: {np.mean(after_baselines)}")
with open(os.path.join(out_dir,'h2_delta_test_drives.csv'), 'w+') as outfile:
    writer = csv.writer(outfile)
    culled_feature_labels = ['reward'] + [l for i,l in enumerate(feature_labels) if i not in blacklist] + ['session'] + ['map_num']
    writer.writerow(culled_feature_labels)
    for sess_num, exp in enumerate(pilot_h2_test_drives):
        for sample in exp['test_drive_results']:
            for map_num,result in enumerate(sample):
                d = design_dict[str(exp['session_id'])]
                culled_design = [feature for i,feature in enumerate(d) if i not in blacklist]
                # writer.writerow([result-baselines[map_num]] + culled_design + [str(exp['session_id'])] + [map_num])
                writer.writerow([result-after_baselines[map_num]] + culled_design + [session2indexdict[str(exp['session_id'])]] + [map_num])