from keras.models import load_model

from keras.utils import plot_model
import numpy as np

model_path = "/home/dev/scratch/cars/carracing_clean/train_logs/avg_dqn_10_seq_model_2_2000.h5"

model = load_model(model_path)
model.summary()
#plot_model(model, to_file="utils/model.png")

from vis.utils import utils
from vis.visualization import visualize_saliency, visualize_activation

layer_idx = utils.find_layer_idx(model, 'activation_2')

fake_input = np.ones(111*10)

grad_top1 = visualize_saliency(model, layer_idx, filter_indices = None, seed_input = fake_input)
import pdb; pdb.set_trace()
visualization = visualize_activation(model, layer_idx, filter_indices= [1], seed_input = fake_input)
