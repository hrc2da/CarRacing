import pickle as pkl
import pdb

path = '/home/dev/scratch/cars/carracing_clean/flaskapp/static/a3476ba7c278432db0315eda9546b7a4/bo_sessions_all_features/5f6d02c2673446edf0d88f1c/policy_training/designs.pkl'

with open(path, 'rb') as infile:
    data = pkl.load(infile)

print(len(data))
print(f'number of bo steps: {len(data[0])}')
pdb.set_trace()
