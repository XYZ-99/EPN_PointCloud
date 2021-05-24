import os
import sys

import torch
import torch.nn as nn

import json

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
import SPConvNets.utils as M

class SegSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(SegSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        self.upblock = nn.ModuleList()
        for block_param in params['upsample']:
            self.upblock.append(M.BasicUpsampleSO3ConvBlock(block_param))

        self.outblock = M.RelSO3OutBlockR(params['outblock']) # TODO: change the output block
        self.na_in = params['na'] # TODO: ?
        self.invariance = True    # TODO: ?

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = torch.cat((x[:,0], x[:,1]),dim=0)
        x = M.preprocess_input(x, self.na_in, False)

        skipped_sph_pc = [None]
        for block_i, block in enumerate(self.backbone):
            x = block(x)
            skipped_sph_pc.append(x)

        # The last one is not needed
        skipped_sph_pc.pop()

        for upblock in self.upblock:
            # x.xyz.shape:  torch.Size([4, 3, 128])
            # x.feats.shape:  torch.Size([4, 128, 128, 60])
            # memory leakage
            x = upblock(x, skipped_sph_pc.pop())

        f1, f2 = torch.chunk(x.feats,2,dim=0)
        x1, x2 = torch.chunk(x.xyz,2,dim=0)

        # TODO: new outblock (for the grasp rotation, width and quality for now)
        confidence, quats = self.outblock(f1, f2, x1, x2)

        return confidence, quats

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

def build_model(opt,
                # mlps=[[32,32], [64,64], [128,128], [128,128]],
                mlps=[[32,32], [64,64]],
                # upsample_mlps=[[128], [64], [32], [32]], # TODO: to be decided
                upsample_mlps=[[64], [32]],
                out_mlps=[128, 64],
                strides=[2, 2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8, #0.4, 0.36
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):

    device = opt.device
    input_num= opt.model.input_num
    dropout_rate= opt.model.dropout_rate
    temperature= opt.train_loss.temperature
    so3_pooling =  opt.model.flag
    input_radius = opt.model.search_radius
    kpconv = opt.model.kpconv

    na = 1 if opt.model.kpconv else opt.model.kanchor

    # to accomodate different input_num
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'upsample': [],
              'na': na
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    weighted_sigma = [sigma_ratio * radii[0]**2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                if stride_conv:
                    neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            # one-inter one-intra policy
            block_type = 'inter_block' if na != 60  else 'separable_block'

            inter_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(inter_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    dim_in = mlps[-1][-1]
    for i, block in enumerate(upsample_mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            block_type = 'inter_block' if na != 60  else 'separable_block'

            if j == 0 and i != 3:
                linked_block_dim = mlps[-(i+1)][-1]
                dim_in += linked_block_dim

            kernel_size = 1 # TODO: ?
            nidx = i + 1    # TODO: ?
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density)) # TODO: ?
            lazy_sample = i != 0 or j != 0 # TODO: ?
            inter_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': 1,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na
                }
            }
            block_param.append(inter_param)
            dim_in = dim_out

        params['upsample'].append(block_param)


    # TODO: Modify the outblock
    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
        'fc': [64],
        'k': 40,
        'pooling': so3_pooling,
        'temperature': temperature,
        'kanchor': na,
        'representation': opt.model.representation
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = SegSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)
