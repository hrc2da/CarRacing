import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from utils import feature_labels, feature_ranges, car2config, config2car, densities, wheelmoment, blacklist, chopshopfeatures2indexlist, feature_labels, featuremask2names
from pilotsessions import sessions, session2indexdict
from config import Config

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

client = MongoClient(Config.MONGO_HOST, Config.MONGO_PORT)
db = client[Config.DB_NAME]
sessions_collection = db.sessions
session_ids = [ObjectId(s) for s in sessions]
pilot_sessions = list(sessions_collection.find({"_id":{"$in":session_ids}}))



body_features = ['bumper_width1','bumper_width2','hull2_width1','hull2_width2','hull3_width1','hull3_width2','hull3_width3','hull3_width4','spoiler_width1','spoiler_width2']
body_features_indices = [feature_labels.index(f) for f in body_features]
all_features = [f for i,f in enumerate(feature_labels) if i not in blacklist and i not in body_features_indices]

designs = [[f"P{session_ids.index(s['_id'])}"] + [feature_val for i,feature_val in enumerate(car2config(s['final_design'], correct_negative_widths=True)) 
                if i not in blacklist and i not in body_features_indices] for s in pilot_sessions]

default_design = [feature_val for i,feature_val in enumerate(car2config(pilot_sessions[0]['initial_design'], correct_negative_widths=True))
                    if i not in blacklist and i not in body_features_indices]

assert len(default_design) == 9

default_design_without_color = default_design[:-1]

feature_matrix = np.zeros((len(designs)+1,8)) # ignore color
labels = ['default'] + [None]*len(designs)
feature_matrix[0,:] = np.array(default_design_without_color)
for d in designs:
    idx = int(d[0][1:]) + 1
    feature_matrix[idx,:] = d[1:-1]
    labels[idx] = d[0]

# design_dict = {}
# for d in designs:
#     design_dict[d[0]] = d[1:]
# data = {
#     'P0':{
#         'features': ('engine_power','tire_tread','wheel_radius','body_shape','steering_sensitivity','brake_sensitivity'),
#         'n_designs': 36
#     },
#     'P1':{
#         'features': ('engine_power','tire_tread','wheel_radius','wheel_width','body_shape','steering_sensitivity','rear_steering','brake_sensitivity','max_speed','color'),
#         'n_designs': 9
#     },
#     'P2':{
#         'features':('engine_power','tire_tread','wheel_radius','wheel_width','body_shape','steering_sensitivity','brake_sensitivity','max_speed'),
#         'n_designs':10
#     }
# }

# all_features = ['engine_power','tire_tread','wheel_radius','wheel_width','steering_sensitivity','brake_sensitivity','rear_steering','max_speed','color','body_shape']

def features2row(features: list) -> list:
    global all_features
    return [1 if feature in features else 0 for feature in all_features ]
    # return[np.abs(design[f]-default[f] for f in features)]

# print(features2row(data['P0']['features']))

# todo: get the actual designs from the db so we can plot values AND the body shape!! on the plot
# feature_matrix = np.array([features2row(d['features']) for d in data.values()])

print(feature_matrix)
binary_feature_matrix = np.zeros(feature_matrix.shape)
for i,row in enumerate(feature_matrix):
    for j,feature_val in enumerate(row):
        binary_feature_matrix[i,j] = default_design[j] != feature_val

normalized_divergence_feature_matrix = np.zeros(feature_matrix.shape)
ranges = np.ptp(feature_matrix,0) # you should actually get ranges from the js app TODO
for i,row in enumerate(feature_matrix):
    for j,feature_val in enumerate(row):
        normalized_divergence_feature_matrix[i,j] =  (feature_val - default_design[j])/ranges[j]


# fig,ax = plt.subplots()
# ax.imshow(feature_matrix, cmap='binary')
# ax.set_xticks(np.arange(len(all_features))+0.5, minor=False)
# ax.set_yticks(np.arange(len(data))+0.5, minor=False)
# ax.xaxis.tick_top()
# ax.xaxis.set_label_position('bottom')
# ax.set_xticklabels(all_features, rotation=90)
# plt.grid()
# plt.show()




# ax = sns.heatmap(normalized_divergence_feature_matrix, annot=feature_matrix, fmt='.2g', linewidth=1, linecolor='black', square=False, xticklabels=all_features[:-1], yticklabels=labels,cmap="vlag_r")
ax = sns.heatmap(binary_feature_matrix, annot=False, linewidth=1, linecolor='black', square=False, xticklabels=all_features[:-1], yticklabels=labels,cmap="rocket")
ax.tick_params(bottom=False, labeltop=True, labelbottom=False)
ax.tick_params(axis='x', rotation=90)
ax.tick_params(axis='y', rotation=0)
plt.show()
