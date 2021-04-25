import csv 

participant_data = {
    0: (["engine power", "tire tread", "wheel radius", "body shape", "steering sensitivity", "brake sensitivity"], 36),
    1: (["engine power", "tire tread", "wheel radius", "wheel width" ,"body shape", "steering sensitivity", "rear steering", "brake sensitivity", "max speed", "color"], 9),
    2: (["engine power", "tire tread", "wheel radius", "wheel width" ,"body shape", "steering sensitivity", "brake sensitivity"], 10),
    3: (["engine power", "wheel width" ,"body shape", "steering sensitivity", "brake sensitivity", "max speed"], 6),
    4: (["engine power", "wheel radius", "wheel width" ,"body shape", "steering sensitivity", "rear steering", "brake sensitivity", "max speed", "color"], 12),
    5: (["engine power", "body shape"], 3),
    6: (["tire tread", "wheel radius", "body shape", "steering sensitivity", "brake sensitivity", "color"], 82),
    7: (["engine power", "tire tread","body shape", "steering sensitivity", "max speed", "color"], 17),
    8: (["wheel width", "steering sensitivity", "max speed", "color"], 4),
    9: (["engine power", "tire tread", "wheel width" ,"body shape", "steering sensitivity", "rear steering", "max speed", "color"], 50),
    10: (["tire tread", "wheel width" ,"body shape", "brake sensitivity", "max speed", "color"], 5),
    11: (["tire tread", "wheel radius", "wheel width" ,"body shape", "steering sensitivity", "rear steering", "brake sensitivity", "max speed", "color"], 14),
}

all_features = set(["engine power", "tire tread", "wheel radius", "wheel width" ,"body shape", "steering sensitivity", "rear steering", "brake sensitivity", "max speed", "color"])

rows = []
hist_rows = []
for participant_num, participant_info in participant_data.items():
    participant_num_fmt = f"p{participant_num}"
    features, num_designs = participant_info
    features_in_final_design = features
    hist_rows.append([participant_num_fmt, num_designs])
    features_not_in_final_design = all_features - set(features_in_final_design)
    for feature in features_in_final_design:
        rows.append([participant_num_fmt, feature, 1])
    for feature in features_not_in_final_design:
        rows.append([participant_num_fmt, feature, 0])

with open('heatmap_data.csv','w+') as outfile:
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['participant','feature','value'])
    for row in rows:
        writer.writerow(row)

with open('hist_data.csv','w+') as outfile:
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['participant','value'])
    for row in hist_rows:
        writer.writerow(row)