import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import mcubes
from bspt import digest_bsp, get_mesh, get_mesh_watertight
#from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from utils import *

#pytorch 1.2.0 implementation


class BSP_AE_Loader(object):
    def __init__(self, config):
        """
        Args:
            too lazy to explain
        """

        print('here: BSP_AE init called.')
        self.phase = config.phase

        #progressive training
        #1-- (16, 16*16*16)
        #2-- (32, 16*16*16)
        #3-- (64, 16*16*16*4)
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size==16:
            self.load_point_batch_size = 16*16*16
        elif self.sample_vox_size==32:
            self.load_point_batch_size = 16*16*16
        elif self.sample_vox_size==64:
            self.load_point_batch_size = 16*16*16*4
        self.shape_batch_size = config.shape_batch_size
        self.point_batch_size = 16*16*16
        self.input_size = 64 #input voxel grid size

        self.ef_dim = 32
        self.p_dim = 4096
        self.c_dim = 256

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        
        data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'

        print('[HERE: In modelAE/BSP_AE] -----dataset checkpoint config display----')
        print('[HERE: In modelAE/BSP_AE] dataset_name: %s'%self.dataset_name)
        print('[HERE: In modelAE/BSP_AE] dataset_load: %s'%self.dataset_load)
        print('[HERE: In modelAE/BSP_AE] checkpoint_dir: %s'%self.checkpoint_dir)
        print('[HERE: In modelAE/BSP_AE] data_dir: %s'%self.data_dir)
        print('[HERE: In modelAE/BSP_AE] data_hdf5_name: %s'%data_hdf5_name)

        print('[HERE: In modelAE/BSP_AE] -----dataset loading process----')
        if os.path.exists(data_hdf5_name):
            print('[HERE: In modelAE/BSP_AE] data loading starts')
            data_dict = h5py.File(data_hdf5_name, 'r')
            print('[HERE: In modelAE/BSP_AE] data loading done')
            print('[HERE: In modelAE/BSP_AE] data_dict keys:', data_dict.keys())

            print('[HERE: In modelAE.BSP_AR] data_dict[\'pixels\']:', data_dict['pixels'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_16\']:', data_dict['points_16'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_32\']:', data_dict['points_32'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_64\']:', data_dict['points_64'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_16\']:', data_dict['values_16'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_32\']:', data_dict['values_32'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_64\']:', data_dict['values_64'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'voxels\']:', data_dict['voxels'])

            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_16\']:', data_dict['points_16'][:])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_16\']:', np.array(data_dict['values_16']))
            #print('[HERE: In modelAE.BSP_AR] data_dict[\'voxels\']:', np.array(data_dict['voxels']))

            print('[HERE: In modelAE/BSP_AE] data preprocessing starts')
            print('[HERE: In modelAE/BSP_AE] data_points normalization starts')
            self.data_points_ori = data_dict['points_'+str(self.sample_vox_size)][:]

            self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
            print('[HERE: In modelAE/BSP_AE] data_points normalization done')
            print('[HERE: In modelAE/BSP_AE] data_dict[\'points_%s\'] info:' % str(self.sample_vox_size))
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_points))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_points.shape)
            print('[HERE: In modelAE/BSP_AE] | content', self.data_points)

            
            print('[HERE: In modelAE/BSP_AE] data_points concatenation starts. This turns to homogenous coordinates.')
            self.data_points = np.concatenate([self.data_points, np.ones([len(self.data_points),self.load_point_batch_size,1],np.float32) ],axis=2)
            print('[HERE: In modelAE/BSP_AE] data_points concatenation done')
            print('[HERE: In modelAE/BSP_AE] data_points concatenated info:')
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_points))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_points.shape)
            
            print('[HERE: In modelAE/BSP_AE] data_values retyping starts')
            self.data_values_ori = data_dict['values_'+str(self.sample_vox_size)][:]
            self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
            print('[HERE: In modelAE/BSP_AE] data_values retyping done')
            print('[HERE: In modelAE/BSP_AE] data_values info:')
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_values))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_values.shape)
            print('[HERE: In modelAE.BSP_AE] | content:', self.data_values)

            print('[HERE: In modelAE/BSP_AE] data_voxels load starts')
            print('[HERE: In modelAE/BSP_AE] data_dict[\'voxels\'] info:')
            print('[HERE: In modelAE/BSP_AE] | type: ', type(data_dict['voxels']))
            print('[HERE: In modelAE/BSP_AE] | shape: ', data_dict['voxels'].shape)
            print('[HERE: In modelAE/BSP_AE] changing dataset voxels from h5py dataset to numpy array. this could take a while.')
            self.data_voxels = np.array(data_dict['voxels']) # [:] at the end means to turn the hdf5 data format to numpy arrays.
            print('[HERE: In modelAE/BSP_AE] dataset voxels are now numpy array.')
            print('[HERE: In modelAE/BSP_AE] self.data_voxels info:')
            print('[HERE: In modelAE/BSP_AE] | type: ', type(self.data_voxels))
            print('[HERE: In modelAE/BSP_AE] | shape: ', self.data_voxels.shape)
            print('[HERE: In modelAE/BSP_AE] data_voxels load done')
            #reshape to NCHW
            print('[HERE: In modelAE/BSP_AE] data_voxels reshaping starts')
            self.data_voxels = np.reshape(self.data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
            print('[HERE: In modelAE/BSP_AE] data_voxels reshaping done')
            print('[HERE: In modelAE/BSP_AE] self.data_voxels reshaped info:')
            print('[HERE: In modelAE/BSP_AE] | type: ', type(self.data_voxels))
            print('[HERE: In modelAE/BSP_AE] | shape: ', self.data_voxels.shape)
            print('[HERE: In modelAE/BSP_AE] data preprocessing done')
        else:
            print("error: cannot load " + data_hdf5_name)
            exit(0)
        
        self.real_size = 64 #output point-value voxel grid size in testing
        self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
        test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change
        
        #get coords
        print('[HERE: In modelAE/BSP_AE] coords building starts')
        

        dima = self.test_size
        dim = self.real_size
        self.aux_x = np.zeros([dima,dima,dima],np.uint8)
        self.aux_y = np.zeros([dima,dima,dima],np.uint8)
        self.aux_z = np.zeros([dima,dima,dima],np.uint8)
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier

        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i,j,k] = i*multiplier
                    self.aux_y[i,j,k] = j*multiplier
                    self.aux_z[i,j,k] = k*multiplier
        self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
        self.coords = (self.coords+0.5)/dim-0.5
        self.coords = np.reshape(self.coords,[multiplier3,test_point_batch_size,3])
        self.coords = np.concatenate([self.coords, np.ones([multiplier3,test_point_batch_size,1],np.float32) ],axis=2)
        self.coords = torch.from_numpy(self.coords)

        print('[HERE: In modelAE/BSP_AE] some coords related parameters:')
        print('[HERE: In modelAE/BSP_AE] | dima = %d, dim = %d' % (dima, dim))
        print('[HERE: In modelAE/BSP_AE] | self.coords shape =', self.coords.shape)
        
        print('[HERE: In modelAE/BSP_AE] coords tensor build complete.')

       
    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

   
