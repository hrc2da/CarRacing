import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.transforms import functional
# torchvision.transforms.functional.to_tensor() https://github.com/pytorch/vision/issues/1419

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)


def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, start_idx, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  data = torch.zeros((M*N, 3, 64, 64)) #np.zeros((M*N, 3, 64, 64), dtype=np.uint8)
  idx = 0
  for i in range(start_idx, start_idx+N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    first_sample = raw_data[0]
    
    l = len(raw_data)
    #raw_data = raw_data.reshape((l,3,64,64)) # not sure if this is ok or not
    #assert np.array_equal(raw_data[0].reshape((64,64,3)),first_sample)
    raw_data = torch.stack([functional.to_tensor(frame) for frame in raw_data]) # stupid ugly way to do this
    
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data


# class CarFramesDataset(Dataset):

#     def __init__(self, filelist, transform=None):
#         '''
#         filelist should be a list of filenames with paths
#         transform (callable, optional): optional transform to be applied on a sample
#         '''


class UnFlatten(nn.Module):
    # def forward(self, input, size=256):
    #     return input.view(input.size(0), size, 2, 2)
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class AutoEncoder(nn.Module):
    '''
    Vanilla Autoencoder
    '''

    input_shape = (64,64,3) # carracing is 96x96; worldmodels is cropping out the bottom and reshaping to 64x64 (see env.py)
    latent_size = 32
    

    def __init__(self):
        super(AutoEncoder, self).__init__()
        # h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
        # h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        # h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        # h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
        # h = tf.reshape(h, [-1, 2*2*256])
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=4, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2), #outputs [batch_size,256,2,2]
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(256*4, self.latent_size)
        )
        # h = tf.layers.dense(self.z, 4*256, name="dec_fc")
        # h = tf.reshape(h, [-1, 1, 1, 4*256])
        # h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
        # h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
        # h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
        # self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, 256*4),
            UnFlatten(), # unflatten will give us [batch_size, 1024, 1, 1] copying the network above (which differs from the conv2d output of [batch,256,2,2], but I guess the kernels are differently sized here.)
            torch.nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    
    AE = AutoEncoder().to(device)
    print(AE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(AE.parameters())

    # load dataset from record/*. only use first 10K, sorted by filename.
    filelist = os.listdir(DATA_DIR)
    filelist.sort()
    N_step = 100
    max_file = 10000
    filelist_len = len(filelist) - len(filelist) % N_step # or len(filelist) / N_step * N_step
    print("Num Files: {}".format(filelist_len))
    # probably not ideal, but I am not sure how else to train on all this data
    for N in range(N_step,min(max_file,filelist_len),N_step):
        filelist_batch = filelist[N-N_step:N]
        #print("check total number of images:", count_length_of_filelist(filelist))
        print("Training on files {} to {}".format(N-N_step,N))
        dataset = create_dataset(filelist_batch,start_idx=0, N=N_step)
        reference_image_batch = dataset[0:100].clone().to(device)
        reference_image_path = os.path.join(model_save_path,"reference_file_batch_{}.png".format(N-N_step))
        reference_image = make_grid(reference_image_batch, nrow=10, padding=2)
        print("Saving reference image, shape {} at: {}".format(reference_image.shape,reference_image_path))
        save_image(reference_image, reference_image_path)
        # split into batches:
        total_length = len(dataset)
        num_batches = int(np.floor(total_length/batch_size))
        print("num_batches", num_batches)
        # train loop:
        print("train", "step", "loss", "recon_loss", "kl_loss")
        for epoch in range(NUM_EPOCH):
            #np.random.shuffle(dataset)
            dataset = dataset[torch.randperm(len(dataset))] # random shuffle the torch way
            for idx in range(num_batches):
                batch_input = dataset[idx*batch_size:(idx+1)*batch_size].to(device)

                #obs = batch.astype(np.float)/255.0 # already scaled in to_tensor transform, so no need to scale again

                #batch_input = batch.to(device) # note that you don't really have to reassign here: batch.self is updated AND returned according to https://discuss.pytorch.org/t/questions-about-to-method-for-object-and-tensor-and-how-to-make-code-transparent-between-cpu-and-gpu/25365
                batch_output = AE(batch_input)
                loss = criterion(batch_output,batch_input)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # pass the reference images and save the reconstructions
            with torch.no_grad():
                reference_reconstruction_batch = AE(reference_image_batch)
                reference_reconstruction_path = os.path.join(model_save_path,"reference_file_batch_{}_epoch_{}.png".format(N-N_step,epoch))
                reference_reconstruction_grid = make_grid(reference_reconstruction_batch, nrow=10, padding=2)
                save_image(reference_reconstruction_grid, reference_reconstruction_path)
            print('epoch [{}/{}], loss:{:.4f}'
                    .format(epoch+1, NUM_EPOCH, loss.item()))
            # if epoch % 10 == 0:
            #     pic = to_img(output.cpu().data)
            #     save_image(pic, './dc_img/image_{}.png'.format(epoch))

        # finished, final model:
        torch.save(AE.state_dict(), os.path.join(model_save_path,'conv_autoencoder.pth'))



# ## borrowed from https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = Variable(img).cuda()
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch+1, num_epochs, loss.data[0]))
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, './dc_img/image_{}.png'.format(epoch))

# torch.save(model.state_dict(), './conv_autoencoder.pth')