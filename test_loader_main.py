import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from bsp_ae_loader import BSP_AE_Loader
from modelSVR import BSP_SVR

import argparse
import h5py

import trimesh

parser = argparse.ArgumentParser()
parser.add_argument("--phase", action="store", dest="phase", default=1, type=int, help="phase 0 = continuous, phase 1 = hard discrete, phase 2 = hard discrete with L_overlap, phase 3 = soft discrete [1]")
#phase 0 continuous for better convergence
#phase 1 hard discrete for bsp
#phase 2 hard discrete for bsp with L_overlap
#phase 3 soft discrete for bsp
#use [phase 0 -> phase 1] or [phase 0 -> phase 2] or [phase 0 -> phase 3]
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.0001, type=float, help="Learning rate for adam [0.0001]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")

parser.add_argument("--shape_batch_size", type=int, default=24, help="")

FLAGS = parser.parse_args()



if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)


bsp_ae = BSP_AE_Loader(FLAGS)


item_index = 16

points_ori = bsp_ae.data_points_ori[item_index]
in_points_idx = (bsp_ae.data_values[item_index] == 1.)
out_points_idx = (bsp_ae.data_values[item_index] == 0.)

voxel_raw = bsp_ae.data_voxels[item_index,0]
voxel = trimesh.voxel.VoxelGrid(voxel_raw)
mesh = voxel.marching_cubes


print('[HERE: In test_loader_main] Samples inside original mesh = %d/%d' % ((in_points_idx==True).sum(), in_points_idx.shape[0]))
in_points_idx_by_mcmesh = mesh.contains((points_ori.astype(np.float32) + .5)/4. - .5)
print('[HERE: In test_loader_main] Samples inside MCubes mesh = %d/%d' % ((in_points_idx_by_mcmesh==True).sum(), points_ori.shape[0]))
print('[HERE: In test_loader_main] Common samples inside mesh = %d/%d' % ((in_points_idx_by_mcmesh==in_points_idx[:,0]).sum(), in_points_idx_by_mcmesh.shape[0]))

x_in = points_ori[:,0][in_points_idx[:,0]]
y_in = points_ori[:,1][in_points_idx[:,0]]
z_in = points_ori[:,2][in_points_idx[:,0]]

x_out = points_ori[:,0][out_points_idx[:,0]]
y_out = points_ori[:,1][out_points_idx[:,0]]
z_out = points_ori[:,2][out_points_idx[:,0]]
#out_points = points_ori[~in_points_idx]


skip = 1   # Skip every n points

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#point_range = range(0, points.shape[0], skip) # skip points to prevent crash
ax.scatter(x_in,   # x
           y_in,   # y
           z_in,   # z
           #c=points[point_range, 2], # height data for color
           )
#plt.show()

fig.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_out,   # x
           y_out,   # y
           z_out,   # z
           #c=points[point_range, 2], # height data for color
           )
#plt.show()
#voxel.marching_cubes.show()
