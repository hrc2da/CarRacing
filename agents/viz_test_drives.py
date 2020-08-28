import json
import numpy as np
import matplotlib.pyplot as plt

with open('/home/dev/scratch/cars/carracing_clean/agents/test_drives_normal_200_car.json', 'r') as infile:
    benchmark = json.load(infile)

with open('/home/dev/scratch/cars/carracing_clean/agents/test_drives_nice_200_car.json', 'r') as infile:
    redesigned = json.load(infile)

print(f'Benchmark Car: {np.mean(benchmark)}, {np.std(benchmark)}')
print(f'Test Car: {np.mean(redesigned)}, {np.std(redesigned)}')
plt.boxplot([benchmark,redesigned])
plt.title("Test Drive Performance (n=100) with the Same Driver (200 episodes of experience)")
plt.xticks([1,2],["Benchmark Car", "Select GA-Designed Car"])
plt.ylabel("Episode Reward (>900 Considered Successful)")
plt.show()

import pdb; pdb.set_trace()