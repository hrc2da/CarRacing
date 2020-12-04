from utils import config2car, car2config
import numpy as np
import pickle as pkl

path = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/540d7a5f797745c3beeababa9048d930/bo_sessions/5f8df42ad5c68de5e8185f59/policy_training_0/designs.pkl'

with open(path, 'rb') as infile:
    x,y,design = pkl.load(infile)

best_design_idx = np.argmin(y)
best_reward = y[best_design_idx]
best_design = x[best_design_idx]

assert best_design == design

print(f'Best design reward: {best_reward}')

print(config2car(design))
import pdb; pdb.set_trace()


