from skopt.space import Real, Integer
import itertools
from copy import deepcopy
import numpy as np


# feature_ranges = [Integer(low=10000,high=600000), #eng power
#                     Real(low=0.01,high=5.0), # wheel moment
#                     Integer(low=100,high=10000), # friction_lim
#                     Integer(low=10,high=80), # wheel_rad
#                     Integer(low=5,high=80), #wheel_width
#                     Real(low=5,high=300), #bumper_width1
#                     Real(low=5,high=300), #bumper_width2
#                     Real(low=0.1,high=2), #bumper_density
#                     Real(low=10,high=250), #hull2_width1
#                     Real(low=10,high=250), #hull2_width2
#                     Real(low=0.1,high=2), #hull2_density
#                     Real(low=10,high=250), #hull3_width1
#                     Real(low=10,high=250), #hull3_width2
#                     Real(low=10,high=250), #hull3_width3
#                     Real(low=10,high=250), #hull3_width3
#                     Real(low=0.1,high=2), #hull3_density
#                     Real(low=5,high=300), #spoiler_width1     16
#                     Real(low=5,high=300), #spoiler_width2     17
#                     Real(low=0.1,high=2), #spoiler_density   18
#                     Real(low=0.0,high=2), #steering_scalar  19
#                     Real(low=0.0,high=2), #rear_steering_scalar 20
#                     Real(low=0.0,high=2), #brake_scalar 21
#                     Integer(low=5,high=200), #max_speed 22
#                     Integer(low=0,high=16777215)] #color -- you need to set this up!!!!!!



feature_ranges = [(10000,600000), #eng_power          0
                (0.01,5), #wheel_moment        1 # mask this out!!!
                (100,10000), #friction_lim      2
                (10,80), #wheel_rad         3
                (5,80), #wheel_width       4
                (0,300), #bumper_width1      5
                (1,300), #bumper_width2      6
                (0.1,2), #bumper_density    7
                (10,250), #hull2_width1      8
                (10,250), #hull2_width2      9
                (0.1,2), #hull2_density     10
                (10,250), #hull3_width1      11
                (10,250), #hull3_width2      12
                (10,250), #hull3_width3      13
                (10,250), #hull3_width4      14
                (0.1,2), #hull3_density     15
                (0,300), #spoiler_width1     16
                (1,300), #spoiler_width2     17
                (0.1,2), #spoiler_density   18
                (0.0,2), #steering_scalar  19
                (0.0,2), #rear_steering_scalar 20
                (0.0,2), #brake_scalar 21
                (5,200), #max_speed 22
                (0,16777215)] #color -- you need to set this up!!!!!!




# feature_types = [float, float, float, int, int, int, int, float, int, int, float, int, int, int, int, float, int, int, float, float, float, float, int]
feature_labels = ['eng_power','wheel_moment','friction_lim','wheel_rad','wheel_width',
'bumper_width1','bumper_width2','bumper_density','hull2_width1','hull2_width2','hull2_density',
'hull3_width1','hull3_width2','hull3_width3','hull3_width4','hull3_density',
'spoiler_width1','spoiler_width2','spoiler_density','steering_scalar','rear_steering_scalar','brake_scalar','max_speed','color']

noise_scalars = [500, 0.5, 100, 2, 2, 5, 5, 1, 5,5,1, 5,5,5,5,1, 5,5,1, 0.2,0.2,0.2, 5, 1000]

assert len(feature_labels) == len(noise_scalars)
assert len(noise_scalars) == len(feature_ranges)
def perturb(base, mask_indices):
    ''' add random noise to the masked features and return
    '''
    perturbed_design = []
    for i in range(len(feature_ranges)):
        if i in mask_indices:
            noise = np.random.normal(loc=0,scale=noise_scalars[i])
            perturbed_val = np.clip(base[i]+noise,feature_ranges[i][0],feature_ranges[i][1])
            perturbed_design.append(perturbed_val)
        else:
            perturbed_design.append(base[i])
    assert len(perturbed_design) == len(base)
    return perturbed_design

def chopshop2index(label):
    # returns a list
    try:
        return [feature_labels.index(label)]
    except ValueError as e:
        # if the label isn't in the list
        if label == 'bumper':
            return [5,6]
        elif label == 'spoiler':
            return [16,17]
        elif label == 'front_body':
            return [8,9]
        elif label == 'rear_body':
            return [11,12,13,14]
        else: raise(e)

def chopshopfeatures2indexlist(feature_dict, invert=False):
    if invert==True:
        features_selected = [label.lower() for label, value in feature_dict.items() if value == False]
    else:
        features_selected = [label.lower() for label, value in feature_dict.items() if value == True]
    unflattened_indices = [chopshop2index(feature) for feature in features_selected]
    indices = list(itertools.chain(*unflattened_indices))
    indices.sort()
    return indices

# def indexlist2chopshopfeatures(list_of_indices):

def featuremask2names(list_of_indices):
    return [feature_labels[i] for i in list_of_indices]

densities = [7,10,15,18]
wheelmoment = [1]

blacklist = wheelmoment + densities


# assert len(feature_types) == len(feature_labels)

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
    packed_config['color'] = f'{int(config[23]):0{6}x}' #CONVERT int to hex
    return packed_config

def unpack(config):
    '''
        return an array version of an unparsed config
    '''
    unpacked_config = [0 for i in range(24)]
    unpacked_config[0] = config['eng_power']    
    unpacked_config[1] = config['wheel_moment']
    unpacked_config[2] = config['friction_lim']
    unpacked_config[3] = config['wheel_rad']
    unpacked_config[4] = config['wheel_width']
    # unpacked_config[5] = config['drive_train']
    unpacked_config[5] = config['bumper']['w1']
    unpacked_config[6] = config['bumper']['w2']
    unpacked_config[7] = config['bumper']['d']
    unpacked_config[8] = config['hull_poly2']['w1']
    unpacked_config[9] = config['hull_poly2']['w2']
    unpacked_config[10] = config['hull_poly2']['d']
    unpacked_config[11] = config['hull_poly3']['w1']
    unpacked_config[12] = config['hull_poly3']['w2']
    unpacked_config[13] = config['hull_poly3']['w3']
    unpacked_config[14] = config['hull_poly3']['w4']
    unpacked_config[15] = config['hull_poly3']['d']
    unpacked_config[16] = config['spoiler']['w1']
    unpacked_config[17] = config['spoiler']['w2']
    unpacked_config[18] = config['spoiler']['d']
    unpacked_config[19] = config['steering_scalar']
    unpacked_config[20] = config['rear_steering_scalar']
    unpacked_config[21] = config['brake_scalar']
    unpacked_config[22] = config['max_speed']
    unpacked_config[23] = int(config['color'],16) #convert hex to int
    return unpacked_config
    # return [self.problem.types[i].encode(unpacked_config[i]) for i in range(self.problem.nvars)]

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

def unparse_config(config,correct_negative_widths=False):
    '''
        translate a config in the carracing format back into the ga format (packed)
        This returns a dict, for the GA we need an array.
        to get all the way to the ga format, call unpack(unparse_config(config))
        This is reverse-engineered from parse_config
    '''
    config = deepcopy(config)
    if(config == {}):
        return config
    else:
        if correct_negative_widths==True:
            if("hull_poly1" in config.keys()):
                coords = config["hull_poly1"]
                config['bumper'] = {'w1':abs(coords[1][0]*2), 'w2':abs(coords[2][0]*2), 'd':config['hull_densities'][0]}
            if("hull_poly2" in config.keys()):
                coords = config["hull_poly2"]
                config['hull_poly2'] = {'w1':abs(coords[1][0]*2),'w2':abs(coords[2][0]*2),'d':config['hull_densities'][1]}
            if("hull_poly3" in config.keys()):
                coords = config["hull_poly3"]
                config['hull_poly3'] = {'w1':abs(coords[0][0]*2),'w2':abs(coords[1][0]*2),'w3':abs(coords[2][0]*2),'w4':abs(coords[3][0]*2),'d':config['hull_densities'][2]}
            if("hull_poly4" in config.keys()):
                coords = config["hull_poly4"]
                config["spoiler"] = {'w1':abs(coords[1][0]*2),'w2':abs(coords[2][0]*2), 'd':config['hull_densities'][3]}
        else:
            if("hull_poly1" in config.keys()):
                coords = config["hull_poly1"]
                config['bumper'] = {'w1':coords[1][0]*2, 'w2':coords[2][0]*2, 'd':config['hull_densities'][0]}
            if("hull_poly2" in config.keys()):
                coords = config["hull_poly2"]
                config['hull_poly2'] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2,'d':config['hull_densities'][1]}
            if("hull_poly3" in config.keys()):
                coords = config["hull_poly3"]
                config['hull_poly3'] = {'w1':coords[0][0]*2,'w2':coords[1][0]*2,'w3':coords[2][0]*2,'w4':coords[3][0]*2,'d':config['hull_densities'][2]}
            if("hull_poly4" in config.keys()):
                coords = config["hull_poly4"]
                config["spoiler"] = {'w1':coords[1][0]*2,'w2':coords[2][0]*2, 'd':config['hull_densities'][3]}
        del config['hull_poly1']
        del config['hull_poly4']
    return config

def config2car(config):
    # converts a vector car config into a json
    config = [float(c) for c in config]
    return parse_config(pack(config))

def car2config(car, correct_negative_widths=False):
    # global feature_types
    # converts a json car config into a vector
    return unpack(unparse_config(car, correct_negative_widths))
    # return [feature_types[i](val) for i, val in enumerate(unpack(unparse_config(car)))]



# def _create_track(self, save_path=None):
#         CHECKPOINTS = 12

#         # Create checkpoints
#         checkpoints = []
#         for c in range(CHECKPOINTS):
#             noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
#             alpha = 2 * math.pi * c / CHECKPOINTS + noise
#             rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
#             if c == 0:
#                 alpha = 0
#                 rad = 1.5*TRACK_RAD
#             if c == CHECKPOINTS-1:
#                 alpha = 2*math.pi*c/CHECKPOINTS
#                 self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
#                 rad = 1.5*TRACK_RAD
#             checkpoints.append((alpha, rad*math.cos(alpha), rad*math.sin(alpha)))
#         self.road = []

#         # Go from one checkpoint to another to create track
#         x, y, beta = 1.5*TRACK_RAD, 0, 0
#         dest_i = 0
#         laps = 0
#         track = []
#         no_freeze = 2500
#         visited_other_side = False
#         while True:
#             alpha = math.atan2(y, x)
#             if visited_other_side and alpha > 0:
#                 laps += 1
#                 visited_other_side = False
#             if alpha < 0:
#                 visited_other_side = True
#                 alpha += 2*math.pi
#             while True: # Find destination from checkpoints
#                 failed = True
#                 while True:
#                     dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
#                     if alpha <= dest_alpha:
#                         failed = False
#                         break
#                     dest_i += 1
#                     if dest_i % len(checkpoints) == 0:
#                         break
#                 if not failed:
#                     break
#                 alpha -= 2*math.pi
#                 continue
#             r1x = math.cos(beta)
#             r1y = math.sin(beta)
#             p1x = -r1y
#             p1y = r1x
#             dest_dx = dest_x - x  # vector towards destination
#             dest_dy = dest_y - y
#             proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
#             while beta - alpha > 1.5*math.pi:
#                  beta -= 2*math.pi
#             while beta - alpha < -1.5*math.pi:
#                  beta += 2*math.pi
#             prev_beta = beta
#             proj *= SCALE
#             if proj > 0.3:
#                  beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
#             if proj < -0.3:
#                  beta += min(TRACK_TURN_RATE, abs(0.001*proj))
#             x += p1x*TRACK_DETAIL_STEP
#             y += p1y*TRACK_DETAIL_STEP
#             track.append((alpha,prev_beta*0.5 + beta*0.5,x,y))
#             if laps > 4:
#                  break
#             no_freeze -= 1
#             if no_freeze == 0:
#                  break

#         # Find closed loop range i1..i2, first loop should be ignored, second is OK
#         i1, i2 = -1, -1
#         i = len(track)
#         while True:
#             i -= 1
#             if i == 0:
#                 return False  # Failed
#             pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
#             if pass_through_start and i2 == -1:
#                 i2 = i
#             elif pass_through_start and i1 == -1:
#                 i1 = i
#                 break
#         if self.verbose == 1:
#             print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
#         assert i1 != -1
#         assert i2 != -1

#         track = track[i1:i2-1]

#         first_beta = track[0][1]
#         first_perp_x = math.cos(first_beta)
#         first_perp_y = math.sin(first_beta)
#         # Length of perpendicular jump to put together head and tail
#         well_glued_together = np.sqrt(
#             np.square(first_perp_x*(track[0][2] - track[-1][2])) +
#             np.square(first_perp_y*(track[0][3] - track[-1][3])))
#         if well_glued_together > TRACK_DETAIL_STEP:
#             return False

#         # Red-white border on hard turns
#         border = [False]*len(track)
#         for i in range(len(track)):
#             good = True
#             oneside = 0
#             for neg in range(BORDER_MIN_COUNT):
#                 beta1 = track[i-neg-0][1]
#                 beta2 = track[i-neg-1][1]
#                 good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
#                 oneside += np.sign(beta1 - beta2)
#             good &= abs(oneside) == BORDER_MIN_COUNT
#             border[i] = good
#         for i in range(len(track)):
#             for neg in range(BORDER_MIN_COUNT):
#                 border[i-neg] |= border[i]
#         if save_path is not None:
#             road_tiles = []
#             borders = []
#         # Create tiles
#         for i in range(len(track)):
#             alpha1, beta1, x1, y1 = track[i]
#             alpha2, beta2, x2, y2 = track[i-1]
#             road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
#             road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
#             road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
#             road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
#             vertices = [road1_l, road1_r, road2_r, road2_l]
#             road_tiles.append(vertices)
#             self.fd_tile.shape.vertices = vertices
#             t = self.world.CreateStaticBody(fixtures=self.fd_tile)
#             t.userData = t
#             c = 0.01*(i%3)
#             t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
#             t.road_visited = False
#             t.road_friction = 1.0
#             t.fixtures[0].sensor = True
#             self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
#             self.road.append(t)
#             if border[i]:
#                 side = np.sign(beta2 - beta1)
#                 b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
#                 b1_r = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1),
#                         y1 + side * (TRACK_WIDTH+BORDER)*math.sin(beta1))
#                 b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
#                 b2_r = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2),
#                         y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
#                 self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0)))
#                 if save_path is not None:
#                     borders.append(([b1_l, b1_r, b2_r, b2_l],beta1,beta2))
#             else:
#                 if save_paths is not None:
#                     borders.append([])
#         self.track = track
#         if save_paths is not None:
#             with open(save_path, 'wb+') as trackfile:
#                 pkl.dump((track, road_tiles, borders), trackfile)
#         return True
