import glob
import pickle as pkl
import json
import numpy as np


filename = 'pilot/hybrid_amit250'

def pack(config):

    '''
    "eng_power": eng_power,
    "wheel_moment": wheel_moment,
    "friction_lim": friction_lim,
    "wheel_rad": wheel_rad,
    "wheel_width": wheel_width,
    "wheel_pos": wheel_pos,
    "hull_poly1": hull_poly1,
    "hull_poly2": hull_poly2,
    "hull_poly3": hull_poly3,
    "hull_poly4": hull_poly4,
    "drive_train": drive_train,
    "hull_densities": hull_densities
    '''
    packed_config = {}
    packed_config['eng_power'] = config[0]
    packed_config['wheel_moment'] = config[1]
    packed_config['friction_lim'] = config[2]
    packed_config['wheel_rad'] = config[3]
    packed_config['wheel_width'] = config[4]
    packed_config['drive_train'] = [0,0,1,1]
    packed_config['bumper'] = {'w1':config[5], 'w2':config[6], 'd':config[7]}
    packed_config['hull_poly2'] = {'w1':config[8],'w2':config[9],'d':config[10]}
    packed_config['hull_poly3'] = {'w1':config[11],'w2':config[12],'w3':config[13],'w4':config[14],'d':config[15]}
    packed_config['spoiler'] = {'w1':config[16], 'w2':config[17], 'd':config[18]}
    packed_config['steering_scalar'] = config[19]
    packed_config['rear_steering_scalar'] = config[20]
    packed_config['brake_scalar'] = config[21]
    packed_config['max_speed'] = config[22]
    return packed_config

def parse_config(config):
    l1 = 0
    l2 = 0
    spoiler_d = 35 #TODO: check that these match the poly's on the baseline car
    bumper_d = 25
    densities =[0,0,0,0]
    # for reference, from the baseline car: 
    # bumper is hull_poly1, spoiler is hull_poly4
    # hull_poly1 = [(-60,+130), (+60,+130), (+60,+110), (-60,+110)]
    # hull_poly2 = [(-15,+120), (+15,+120),(+20, +20), (-20,  20)]
    # hull_poly3 = [  (+25, +20),(+50, -10),(+50, -40),(+20, -90),(-20, -90),(-50, -40),(-50, -10),(-25, +20)]
    # hull_poly4 = [(-50,-120), (+50,-120),(+50,-90),  (-50,-90)]
    bumper_y1 = 130
    bumper_y2 = 110
    h2_y1 = 120
    h2_y2 = 20
    h3_y1 = 20
    h3_y2 = -10
    h3_y3 = -40
    h3_y4 = -90
    spoiler_y1 = -120
    spoiler_y2 = -90
    if(config == {}):
        return config
    else:
        if("bumper" in config.keys()):
            bumper = config["bumper"]
            config["hull_poly1"] = [(-bumper['w1']/2, bumper_y1),(bumper["w1"]/2, bumper_y1),( bumper["w2"]/2, bumper_y2),(-bumper["w2"]/2, bumper_y2)]
            densities[0] = bumper["d"]
            del config["bumper"]

        if("hull_poly2" in config.keys()):
            hull2 = config["hull_poly2"]
            config["hull_poly2"] = [(-hull2['w1']/2, h2_y1),(hull2['w1']/2, h2_y1),(hull2['w2']/2, h2_y2),(-hull2['w2']/2, h2_y2)]
            densities[1] = hull2['d']

        if("hull_poly3" in config.keys()):
            hull3 = config["hull_poly3"]
            config["hull_poly3"] = [(hull3['w1']/2, h3_y1),(hull3['w2']/2, h3_y2),(hull3['w3']/2, h3_y3),(hull3['w4']/2, h3_y4), \
                                    (-hull3['w4']/2, h3_y4),(-hull3['w3']/2, h3_y3),(-hull3['w2']/2, h3_y2),(-hull3['w1']/2, h3_y1)]
            densities[2] = hull3['d']

        if("spoiler" in config.keys()):
            spoiler = config["spoiler"]
            config["hull_poly4"] = [(-spoiler['w1']/2, spoiler_y1),(spoiler['w1']/2, spoiler_y1),(spoiler['w2']/2, spoiler_y2),(-spoiler['w2']/2, spoiler_y2)]
            densities[3] = spoiler['d']
            del config["spoiler"]

        config["hull_densities"] = densities
        config["wheel_pos"] = [(-55,+80), (+55,+80),(-55,-82), (+55,-82)]
    return config

def config2car(config):
    # converts a vector car config into a json
    config = [float(c) for c in config]
    return parse_config(pack(config))


with open(filename+'.pkl','rb') as infile:
    data = pkl.load(infile)

# min because reward is inverted
# best_car = config2car(data[0][np.argmin(data[1])])
best_car = config2car(data[-1])

with open(filename+'.json','w+') as outfile:
    json.dump(best_car,outfile)
import pdb; pdb.set_trace()
