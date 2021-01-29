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



class encoder(nn.Module):
    def __init__(self, ef_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.constant_(self.conv_1.bias,0)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.constant_(self.conv_2.bias,0)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.constant_(self.conv_3.bias,0)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.constant_(self.conv_4.bias,0)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)

        #print('[HERE" In modelAE/encoder.init] ef_dim:', self.ef_dim)
        
    def forward(self, inputs, is_training=False):
        #print('[HERE" In modelAE/encoder] input shape:', inputs.shape)
        d_1 = self.conv_1(inputs)
        d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

        d_2 = self.conv_2(d_1)
        d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)
        
        d_3 = self.conv_3(d_2)
        d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)

        d_4 = self.conv_4(d_3)
        d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.ef_dim*8)
        d_5 = torch.sigmoid(d_5)

        #print('[HERE" In modelAE/encoder] output shape:', d_5.shape)
        return d_5

class decoder(nn.Module):
    def __init__(self, ef_dim, p_dim):
        super(decoder, self).__init__()
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
        self.linear_2 = nn.Linear(self.ef_dim*16, self.ef_dim*32, bias=True)
        self.linear_3 = nn.Linear(self.ef_dim*32, self.ef_dim*64, bias=True)
        self.linear_4 = nn.Linear(self.ef_dim*64, self.p_dim*4, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.xavier_uniform_(self.linear_4.weight)
        nn.init.constant_(self.linear_4.bias,0)
        print('[HERE" In modelAE/decoder.init] ef_dim = %d' % self.ef_dim)
        print('[HERE" In modelAE/decoder.init] p_dim = %d' % self.p_dim)

    def forward(self, inputs, is_training=False):
        #print('[HERE" In modelAE/decoder] input shape:', inputs.shape)
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)

        l4 = self.linear_4(l3)
        l4 = l4.view(-1, 4, self.p_dim)

        #print('[HERE" In modelAE/decoder] output shape:', l4.shape)

        return l4

class generator(nn.Module):
    def __init__(self, phase, p_dim, c_dim):
        super(generator, self).__init__()
        self.phase = phase
        self.p_dim = p_dim
        self.c_dim = c_dim
        convex_layer_weights = torch.zeros((self.p_dim, self.c_dim))
        concave_layer_weights = torch.zeros((self.c_dim, 1))
        self.convex_layer_weights = nn.Parameter(convex_layer_weights)
        self.concave_layer_weights = nn.Parameter(concave_layer_weights)
        nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
        nn.init.normal_(self.concave_layer_weights, mean=1e-5, std=0.02)
        #print('[HERE" In modelAE/generator.init] p_dim:', self.p_dim)
        #print('[HERE" In modelAE/generator.init] c_dim:', self.c_dim)
        #print('[HERE" In modelAE/generator.init] convex_layer_weights:', self.convex_layer_weights.shape)
        #print('[HERE" In modelAE/generator.init] concave_layer_weights:', self.concave_layer_weights.shape)

    def forward(self, points, plane_m, is_training=False):
        if self.phase==0:
            #print('[HERE" In modelAE/generator] phase = %d' % self.phase)
            #print('[HERE" In modelAE/generator] points.shape =', points.shape)
            #print('[HERE" In modelAE/generator] plane_m.shape =', plane_m.shape)
            #level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)
            #print('[HERE" In modelAE/generator] h1.shape =', h1.shape)

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)
            h2 = torch.clamp(1-h2, min=0, max=1)

            #level 3
            h3 = torch.matmul(h2, self.concave_layer_weights)
            h3 = torch.clamp(h3, min=0, max=1)

            #print('[HERE" In modelAE/generator] h2.shape =', h2.shape)
            #print('[HERE" In modelAE/generator] h3.shape =', h3.shape)

            return h2,h3
        elif self.phase==1 or self.phase==2:
            #print('[HERE" In modelAE/generator] phase = %d' % self.phase)
            #print('[HERE" In modelAE/generator] points.shape =', points.shape)
            #print('[HERE" In modelAE/generator] plane_m.shape =', plane_m.shape)
            #level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)
            #print('[HERE" In modelAE/generator] h1.shape =', h1.shape)

            #level 2
            h2 = torch.matmul(h1, (self.convex_layer_weights>0.01).float())

            #level 3
            h3 = torch.min(h2, dim=2, keepdim=True)[0]
            #print('[HERE" In modelAE/generator] h2.shape =', h2.shape)
            #print('[HERE" In modelAE/generator] h3.shape =', h3.shape)

            return h2,h3
        elif self.phase==3:
            #print('[HERE" In modelAE/generator] phase = %d' % self.phase)
            #print('[HERE" In modelAE/generator] points.shape =', points.shape)
            #print('[HERE" In modelAE/generator] plane_m.shape =', plane_m.shape)
            #level 1
            h1 = torch.matmul(points, plane_m)
            h1 = torch.clamp(h1, min=0)
            #print('[HERE" In modelAE/generator] h1.shape =', h1.shape)

            #level 2
            h2 = torch.matmul(h1, self.convex_layer_weights)

            #level 3
            h3 = torch.min(h2, dim=2, keepdim=True)[0]
            #print('[HERE" In modelAE/generator] h2.shape =', h2.shape)
            #print('[HERE" In modelAE/generator] h3.shape =', h3.shape)

            return h2,h3
        else:
            #print("Congrats you got an error!")
            #print("generator.phase should be in [0,1,2,3], got", self.phase)
            exit(0)

class bsp_network(nn.Module):
    def __init__(self, phase, ef_dim, p_dim, c_dim):
        super(bsp_network, self).__init__()
        self.phase = phase
        self.ef_dim = ef_dim
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.encoder = encoder(self.ef_dim)
        self.decoder = decoder(self.ef_dim, self.p_dim)
        self.generator = generator(self.phase, self.p_dim, self.c_dim)

    def forward(self, inputs, z_vector, plane_m, point_coord, is_training=False):
        if is_training:
            z_vector = self.encoder(inputs, is_training=is_training)
            plane_m = self.decoder(z_vector, is_training=is_training)
            net_out_convexes, net_out = self.generator(point_coord, plane_m, is_training=is_training)
        else:
            if inputs is not None:
                z_vector = self.encoder(inputs, is_training=is_training)
            if z_vector is not None:
                plane_m = self.decoder(z_vector, is_training=is_training)
            if point_coord is not None:
                net_out_convexes, net_out = self.generator(point_coord, plane_m, is_training=is_training)
            else:
                net_out_convexes = None
                net_out = None

        return z_vector, plane_m, net_out_convexes, net_out


class BSP_AE(object):
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
        self.dataset_id = self.dataset_name.split('_')[0]
        self.dataset_load = self.dataset_id + '_vox256_img_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_id + '_vox256_img_test'
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

            #print('[HERE: In modelAE.BSP_AR] data_dict[\'pixels\']:', data_dict['pixels'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_16\']:', data_dict['points_16'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_32\']:', data_dict['points_32'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'points_64\']:', data_dict['points_64'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_16\']:', data_dict['values_16'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_32\']:', data_dict['values_32'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'values_64\']:', data_dict['values_64'])
            print('[HERE: In modelAE.BSP_AR] data_dict[\'voxels\']:', data_dict['voxels'])

            #print('[HERE: In modelAE.BSP_AR] data_dict[\'points_16\']:', np.array(data_dict['pixels']))
            #print('[HERE: In modelAE.BSP_AR] data_dict[\'values_16\']:', np.array(data_dict['pixels']))
            #print('[HERE: In modelAE.BSP_AR] data_dict[\'voxels\']:', np.array(data_dict['voxels']))

            print('[HERE: In modelAE/BSP_AE] data preprocessing starts')
            print('[HERE: In modelAE/BSP_AE] data_points normalization starts')
            self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
            print('[HERE: In modelAE/BSP_AE] data_points normalization done')
            print('[HERE: In modelAE/BSP_AE] data_dict[\'points_%s\'] info:' % str(self.sample_vox_size))
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_points))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_points.shape)
            #print('[HERE: In modelAE/BSP_AE] | content', self.data_points)

            
            print('[HERE: In modelAE/BSP_AE] data_points concatenation starts. This turns to homogenous coordinates.')
            self.data_points = np.concatenate([self.data_points, np.ones([len(self.data_points),self.load_point_batch_size,1],np.float32) ],axis=2)
            print('[HERE: In modelAE/BSP_AE] data_points concatenation done')
            print('[HERE: In modelAE/BSP_AE] data_points concatenated info:')
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_points))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_points.shape)
            
            print('[HERE: In modelAE/BSP_AE] data_values retyping starts')
            self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
            print('[HERE: In modelAE/BSP_AE] data_values retyping done')
            print('[HERE: In modelAE/BSP_AE] data_values info:')
            print('[HERE: In modelAE/BSP_AE] | type:', type(self.data_values))
            print('[HERE: In modelAE/BSP_AE] | shape:', self.data_values.shape)
            #print('[HERE: In modelAE.BSP_AE] | content:', self.data_values)

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

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.coords = self.coords.to(self.device)

        #build model
        self.bsp_network = bsp_network(self.phase, self.ef_dim, self.p_dim, self.c_dim)
        self.bsp_network.to(self.device)
        #print params
        #for param_tensor in self.bsp_network.state_dict():
        #	print(param_tensor, "\t", self.bsp_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.bsp_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='BSP_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        print('[HERE: In modelAE/BSP_AE] model building done.')
        
        #loss
        if config.phase==0:
            #phase 0 continuous for better convergence
            #L_recon + L_W + L_T
            #G2 - network output (convex layer), the last dim is the number of convexes
            #G - network output (final output)
            #point_value - ground truth inside-outside value for each point
            #cw2 - connections T
            #cw3 - auxiliary weights W
            def network_loss(G2,G,point_value,cw2,cw3):
                loss_sp = torch.mean((point_value - G)**2)
                loss_convex = (torch.sum(torch.clamp(cw2-1, min=0) - torch.clamp(cw2, max=0)))
                loss_concave = torch.sum(torch.abs(cw3-1))
                loss = loss_sp + loss_convex + loss_concave
                return loss_sp, loss_convex, loss_concave, loss
            self.loss = network_loss
        elif config.phase==1:
            #phase 1 hard discrete for bsp
            #L_recon
            def network_loss(G2,G,point_value,cw2,cw3):
                loss_sp = torch.mean((1-point_value)*(1-torch.clamp(G, max=1)) + point_value*(torch.clamp(G, min=0)))
                loss = loss_sp
                return loss_sp,loss
            self.loss = network_loss
        elif config.phase==2:
            #phase 2 hard discrete for bsp with L_overlap
            #L_recon + L_overlap
            def network_loss(G2,G,point_value,cw2,cw3):
                loss_sp = torch.mean((1-point_value)*(1-torch.clamp(G, max=1)) + point_value*(torch.clamp(G, min=0)))
                G2_inside = (G2<0.01).float()
                bmask = G2_inside * (torch.sum(G2_inside, dim=2, keepdim=True)>1).float()
                loss = loss_sp - torch.mean(G2*point_value*bmask)
                return loss_sp,loss
            self.loss = network_loss
        elif config.phase==3:
            #phase 3 soft discrete for bsp
            #L_recon + L_T
            #soft cut with loss L_T: gradually move the values in T (cw2) to either 0 or 1
            def network_loss(G2,G,point_value,cw2,cw3):
                loss_sp = torch.mean((1-point_value)*(1-torch.clamp(G, max=1)) + point_value*(torch.clamp(G, min=0)))
                loss = loss_sp + (torch.sum(torch.min(torch.abs(cw2)*100,torch.abs(cw2-1))[0]))
                return loss_sp,loss
            self.loss = network_loss
        
        print('[HERE: In modelAE/BSP_AE] BSP_AE init end.')

    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

    def train(self, config):
        print('[HERE: In modelAE/BSP_AE.train] BSP_AE.train is called.')
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Checkpoint path does not exist. Load failed...")
            
        shape_num = len(self.data_voxels)
        batch_index_list = np.arange(shape_num)
        
        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("batch size         ", self.shape_batch_size)
        print("batch num          ", int(shape_num/self.shape_batch_size))
        print("load_point_bs      ", self.load_point_batch_size)
        print("point_batch_size   ", self.point_batch_size)
        print("point batch num    ", int(self.load_point_batch_size/self.point_batch_size))
        print("config epoch       ", config.epoch)
        print("training_epoch     ", config.epoch + int(config.iteration/shape_num))
        print("-------------------------------\n\n")
        
        start_time = time.time()
        assert config.epoch==0 or config.iteration==0
        training_epoch = config.epoch + int(config.iteration/shape_num)
        batch_num = int(shape_num/self.shape_batch_size)
        point_batch_num = int(self.load_point_batch_size/self.point_batch_size)

        self.bsp_network.train()
        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_loss_tt = 0
            avg_num = 0
            for idx in range(batch_num):
                dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
                #print('[HERE: In modelAE/BSP_AE.train] dxb shape =', dxb.shape)
                
                batch_voxels = self.data_voxels[dxb].astype(np.float32)
                #print('[HERE: In modelAE/BSP_AE.train] batch_voxels shape =', batch_voxels.shape)
                #print(batch_voxels[0])

                voxels = batch_voxels[0,0]

                colors = np.empty(voxels.shape, dtype=object)
                colors = 'blue'
                
                #fig = plt.figure()
                #ax = fig.gca(projection='3d')
                #ax.voxels(voxels, facecolors=colors, edgecolor='k')

                #plt.show()
                #fig.clear()

                if point_batch_num==1:
                    point_coord = self.data_points[dxb]
                    point_value = self.data_values[dxb]
                    
                else:
                    which_batch = np.random.randint(point_batch_num)
                    point_coord = self.data_points[dxb, which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
                    point_value = self.data_values[dxb, which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]

                #print(point_coord)
                #print('[HERE: In modelAE/BSP_AE.train] point_batch_num = %d' % point_batch_num)
                #print('[HERE: In modelAE/BSP_AE.train] point_coord shape =', point_coord.shape)
                #print('[HERE: In modelAE/BSP_AE.train] point_value shape =', point_value.shape)

                # Reading data and putting them to device
                batch_voxels = torch.from_numpy(batch_voxels).to(self.device)
                point_coord = torch.from_numpy(point_coord).to(self.device)
                point_value = torch.from_numpy(point_value).to(self.device)

                self.bsp_network.zero_grad()
                _, _, net_out_convexes, net_out = self.bsp_network(batch_voxels, None, None, point_coord, is_training=True)
                losses = self.loss(net_out_convexes, net_out, point_value, self.bsp_network.generator.convex_layer_weights, self.bsp_network.generator.concave_layer_weights)

                # errSP: space partitioning loss
                # errT: convex connection layer loss T
                # errW: concave connection layer loss T
                # errTT: total loss
                if len(losses) == 4:
                    errSP, errT, errW, errTT = losses
                else:
                    errSP, errTT = losses
                errTT.backward()
                self.optimizer.step()

                avg_loss_sp += errSP.item()
                avg_loss_tt += errTT.item()
                avg_num += 1

                if len(losses) == 4:
                    print(str(self.sample_vox_size) + " Epoch: [%2d/%2d], step %d/%d: loss_sp: %.6f, loss_T: %.6g, loss_W: %.6g, loss_total: %.6f"%(epoch, training_epoch, idx, batch_num, errSP.item(), errT.item(), errW.item(), errTT.item()))
                else:
                    print(str(self.sample_vox_size) + " Epoch: [%2d/%2d], step %d/%d: loss_sp: %.6f, loss_total: %.6f"%(epoch, training_epoch, idx, batch_num, errSP.item(), errTT.item()))
            print(str(self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f, loss_total: %.6f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num, avg_loss_tt/avg_num))
            if epoch%10==9:
                self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
            if epoch%20==19:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(self.phase)+"-"+str(epoch)+".pth")
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
                #delete checkpoint
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                #save checkpoint
                torch.save(self.bsp_network.state_dict(), save_dir)
                #update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                #write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                fout = open(checkpoint_txt, 'w')
                for i in range(self.max_to_keep):
                    pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
                    if self.checkpoint_manager_list[pointer] is not None:
                        fout.write(self.checkpoint_manager_list[pointer]+"\n")
                fout.close()

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(self.phase)+"-"+str(epoch)+".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
        #delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        #save checkpoint
        torch.save(self.bsp_network.state_dict(), save_dir)
        #update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        #write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer]+"\n")
        fout.close()

        print('HERE: BSP_AE.train is done.')

    def test_1(self, config, name):
        multiplier = int(self.real_size/self.test_size)
        multiplier2 = multiplier*multiplier

        if config.phase==0:
            thres = 0.5
        else:
            thres = 0.99
        
        t = np.random.randint(len(self.data_voxels))
        model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
        batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
        batch_voxels = torch.from_numpy(batch_voxels)
        batch_voxels = batch_voxels.to(self.device)
        _, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i*multiplier2+j*multiplier+k
                    point_coord = self.coords[minib:minib+1]
                    _,_,_, net_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                    if config.phase!=0:
                        net_out = torch.clamp(1-net_out, min=0, max=1)
                    model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
        
        vertices, triangles = mcubes.marching_cubes(model_float, thres)
        vertices = (vertices-0.5)/self.real_size-0.5
        #output ply sum
        write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
        print("[sample: %s]"%(config.sample_dir+"/"+name+".ply"))


    #output bsp shape as ply
    def test_bsp(self, config):
        print('HERE: BSP_AE.test_bsp is called.')
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        
        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels),config.end)):
            model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        point_coord = self.coords[minib:minib+1]
                        _,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
            
            out_m = out_m.detach().cpu().numpy()
            
            bsp_convex_list = []
            model_float = model_float<0.01
            model_float_sum = np.sum(model_float,axis=3)
            for i in range(self.c_dim):
                slice_i = model_float[:,:,:,i]
                if np.max(slice_i)>0: #if one voxel is inside a convex
                    if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
                        model_float_sum = model_float_sum-slice_i
                    else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j,i]>0.01:
                                a = -out_m[0,0,j]
                                b = -out_m[0,1,j]
                                c = -out_m[0,2,j]
                                d = -out_m[0,3,j]
                                box.append([a,b,c,d])
                        if len(box)>0:
                            bsp_convex_list.append(np.array(box,np.float32))

            #print(bsp_convex_list)
            print(len(bsp_convex_list))
            
            #convert bspt to mesh
            vertices, polygons = get_mesh(bsp_convex_list)
            #use the following alternative to merge nearby vertices to get watertight meshes
            #vertices, polygons = get_mesh_watertight(bsp_convex_list)

            #output ply
            write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
        
        print('HERE: BSP_AE.test_bsp is done.')
    
    #output bsp shape as ply and point cloud as ply
    def test_mesh_point(self, config):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels),config.end)):
            print(t)
            model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
            model_float_combined = np.ones([self.real_size,self.real_size,self.real_size],np.float32)
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        point_coord = self.coords[minib:minib+1]
                        _,_, model_out, model_out_combined = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
                        model_float_combined[self.aux_x+i,self.aux_y+j,self.aux_z+k] = np.reshape(model_out_combined.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
            
            out_m_ = out_m.detach().cpu().numpy()

            bsp_convex_list = []
            model_float = model_float<0.01
            model_float_sum = np.sum(model_float,axis=3)
            for i in range(self.c_dim):
                slice_i = model_float[:,:,:,i]
                if np.max(slice_i)>0: #if one voxel is inside a convex
                    #if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
                    #	model_float_sum = model_float_sum-slice_i
                    #else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j,i]>0.01:
                                a = -out_m_[0,0,j]
                                b = -out_m_[0,1,j]
                                c = -out_m_[0,2,j]
                                d = -out_m_[0,3,j]
                                box.append([a,b,c,d])
                        if len(box)>0:
                            bsp_convex_list.append(np.array(box,np.float32))
                            
            #convert bspt to mesh
            vertices, polygons = get_mesh(bsp_convex_list)
            #use the following alternative to merge nearby vertices to get watertight meshes
            #vertices, polygons = get_mesh_watertight(bsp_convex_list)

            #output ply
            write_ply_polygon(config.sample_dir+"/"+str(t)+"_bsp.ply", vertices, polygons)
            
            #sample surface points
            sampled_points_normals = sample_points_polygon_vox64(vertices, polygons, model_float_combined, 16000)
            #check point inside shape or not
            point_coord = np.reshape(sampled_points_normals[:,:3]+sampled_points_normals[:,3:]*1e-4, [1,-1,3])
            point_coord = np.concatenate([point_coord, np.ones([1,point_coord.shape[1],1],np.float32) ],axis=2)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)
            _,_,_, sample_points_value = self.bsp_network(None, None, out_m, point_coord, is_training=False)
            sample_points_value = sample_points_value.detach().cpu().numpy()
            sampled_points_normals = sampled_points_normals[sample_points_value[0,:,0]>1e-4]
            print(len(bsp_convex_list), len(sampled_points_normals))
            np.random.shuffle(sampled_points_normals)
            write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals[:4096])


    #output bsp shape as obj with color
    def test_mesh_obj_material(self, config):
        print('HERE BSP_AE.test_mesh_obj_material is called.')
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        
        w2 = self.bsp_network.generator.convex_layer_weights.detach().cpu().numpy()

        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier

        #write material
        #all output shapes share the same material
        #which means the same convex always has the same color for different shapes
        #change the colors in default.mtl to visualize correspondences between shapes
        fout2 = open(config.sample_dir+"/default.mtl", 'w')
        for i in range(self.c_dim):
            fout2.write("newmtl m"+str(i+1)+"\n") #material id
            fout2.write("Kd 0.80 0.80 0.80\n") #color (diffuse) RGB 0.00-1.00
            fout2.write("Ka 0 0 0\n") #color (ambient) leave 0s
        fout2.close()

        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels),config.end)):
            model_float = np.ones([self.real_size,self.real_size,self.real_size,self.c_dim],np.float32)
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        point_coord = self.coords[minib:minib+1]
                        _,_, model_out, _ = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x+i,self.aux_y+j,self.aux_z+k,:] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size,self.c_dim])
            
            out_m = out_m.detach().cpu().numpy()
            
            bsp_convex_list = []
            color_idx_list = []
            model_float = model_float<0.01
            model_float_sum = np.sum(model_float,axis=3)
            for i in range(self.c_dim):
                slice_i = model_float[:,:,:,i]
                if np.max(slice_i)>0: #if one voxel is inside a convex
                    if np.min(model_float_sum-slice_i*2)>=0: #if this convex is redundant, i.e. the convex is inside the shape
                        model_float_sum = model_float_sum-slice_i
                    else:
                        box = []
                        for j in range(self.p_dim):
                            if w2[j,i]>0.01:
                                a = -out_m[0,0,j]
                                b = -out_m[0,1,j]
                                c = -out_m[0,2,j]
                                d = -out_m[0,3,j]
                                box.append([a,b,c,d])
                        if len(box)>0:
                            bsp_convex_list.append(np.array(box,np.float32))
                            color_idx_list.append(i)

            #print(bsp_convex_list)
            print(len(bsp_convex_list))
            
            #convert bspt to mesh
            vertices = []

            #write obj
            fout2 = open(config.sample_dir+"/"+str(t)+"_bsp.obj", 'w')
            fout2.write("mtllib default.mtl\n")

            for i in range(len(bsp_convex_list)):
                vg, tg = get_mesh([bsp_convex_list[i]])
                vbias=len(vertices)+1
                vertices = vertices+vg

                fout2.write("usemtl m"+str(color_idx_list[i]+1)+"\n")
                for ii in range(len(vg)):
                    fout2.write("v "+str(vg[ii][0])+" "+str(vg[ii][1])+" "+str(vg[ii][2])+"\n")
                for ii in range(len(tg)):
                    fout2.write("f")
                    for jj in range(len(tg[ii])):
                        fout2.write(" "+str(tg[ii][jj]+vbias))
                    fout2.write("\n")

            fout2.close()
        print('HERE BSP_AE.test_mesh_obj_material is called.')


    #output h3
    def test_dae3(self, config):
        print('HERE BSP_AE.test_dae3 is called.')
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
        
        dima = self.test_size
        dim = self.real_size
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier
        
        self.bsp_network.eval()
        for t in range(config.start, min(len(self.data_voxels),config.end)):
            model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
            batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels_)
            batch_voxels = batch_voxels.to(self.device)
            _, out_m, _,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            for i in range(multiplier):
                for j in range(multiplier):
                    for k in range(multiplier):
                        minib = i*multiplier2+j*multiplier+k
                        point_coord = self.coords[minib:minib+1]
                        _,_,_, model_out = self.bsp_network(None, None, out_m, point_coord, is_training=False)
                        model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(model_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
            
            vertices, triangles = mcubes.marching_cubes(model_float, 0.5)
            vertices = (vertices-0.5)/self.real_size-0.5
            #output prediction
            write_ply_triangle(config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)

            vertices, triangles = mcubes.marching_cubes(batch_voxels_[0,0,:,:,:], 0.5)
            vertices = (vertices-0.5)/self.real_size-0.5
            #output ground truth
            write_ply_triangle(config.sample_dir+"/"+str(t)+"_gt.ply", vertices, triangles)
            
            print("[sample %s wrote to file]"%(config.sample_dir+"/"+str(t)+"_gt.ply"))
        print('HERE BSP_AE.test_dae3 is done.')
    
    def get_z(self, config):
        print('HERE BSP_AE.test_dae3 is called.')
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.bsp_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num,self.ef_dim*8], np.float32)

        self.bsp_network.eval()
        print(shape_num)
        for t in range(shape_num):
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            out_z, _,_,_ = self.bsp_network(batch_voxels, None, None, None, is_training=False)
            hdf5_file["zs"][t:t+1,:] = out_z.detach().cpu().numpy()

        hdf5_file.close()
        print("[z]")
        print('HERE BSP_AE.test_dae3 is done.')

