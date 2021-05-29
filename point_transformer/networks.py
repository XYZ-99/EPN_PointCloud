import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

from .pointnet2_modules import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation


class PointNet2Msg(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, cfg, out_dim, net_type='camera', in_dim=0):
        super(PointNet2Msg, self).__init__()
        net_cfg = cfg['pointnet'][net_type]

        self.out_dim = out_dim
        self.in_dim = in_dim
        self.old_pnet = cfg['old_pnet']
        if self.old_pnet:
            self.in_dim = 3
        self.knn = 'knn' in cfg['network'] and cfg['network']['knn']
        self.skip_link = cfg['network']['skip_link']
        self.sa1 = PointNetSetAbstractionMsg(npoint=net_cfg['sa1']['npoint'],
                                             radius_list=net_cfg['sa1']['radius_list'],
                                             nsample_list=net_cfg['sa1']['nsample_list'],
                                             in_channel=self.in_dim + 3,
                                             mlp_list=net_cfg['sa1']['mlp_list'], knn=self.knn)

        self.sa2 = PointNetSetAbstractionMsg(npoint=net_cfg['sa2']['npoint'],
                                             radius_list=net_cfg['sa2']['radius_list'],
                                             nsample_list=net_cfg['sa2']['nsample_list'],
                                             in_channel=self.sa1.out_channel + 3,
                                             mlp_list=net_cfg['sa2']['mlp_list'], knn=self.knn)

        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=self.sa2.out_channel + 3,
                                          mlp=net_cfg['sa3']['mlp'], group_all=True, knn=self.knn)

        self.fp3 = PointNetFeaturePropagation(in_channel=(self.sa2.out_channel + self.sa3.out_channel
                                                          if self.skip_link
                                                          else self.sa3.out_channel),
                                              mlp=net_cfg['fp3']['mlp'])
        self.fp2 = PointNetFeaturePropagation(in_channel=(self.sa1.out_channel + self.fp3.out_channel
                                                          if self.skip_link
                                                          else self.fp3.out_channel),
                                              mlp=net_cfg['fp2']['mlp'])
        self.fp1 = PointNetFeaturePropagation(in_channel=(self.in_dim + 3 + self.fp2.out_channel
                                                          if self.skip_link
                                                          else self.fp2.out_channel),
                                              mlp=net_cfg['fp1']['mlp'])

        self.conv1 = nn.Conv1d(self.fp1.out_channel, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.out_dim)
        self.drop1 = None  # if not cfg['network']['use_dropout'] else nn.Dropout(0.5)

        self.device = cfg['device']

    def forward(self, input):
        l0_xyz, l0_points = input[:, :3], input[:, 3:]
        if self.old_pnet:
            l0_points = l0_xyz
        if len(l0_points) == 0:
            l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points if self.skip_link else None, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points if self.skip_link else None, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1) if self.skip_link else None, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        if self.drop1 is not None:
            x = self.drop1(feat)
        else:
            x = feat
        return x




class PointNet2(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, num_channels_R=2, R_dim=3):
        super(PointNet2, self).__init__()
        cfg = {
            "channel_mult": 4,
            "div": 4,
            "pos_mlp_hidden_dim": 64,
            "attn_mlp_hidden_mult": 4,
            "pre_module": {
                "channel": 16,
                "nsample": 16
            },
            "down_module": {
                "npoint": [256, 64, 32, 16],
                "nsample": [16, 16, 16, 16, 16],
                "attn_channel": [16, 32, 64, 64],
                "attn_num": [2, 2, 2, 2]
            },
            "up_module": {
                "attn_num": [1, 1, 1, 1]
            },
            "heads": {
                "R": [128, num_channels_R * R_dim, None],
                "T": [128, 3, None],
                "N": [128, 3, 'sigmoid'],
                "M": [128, num_channels_R, 'softmax'],
            }
        }
        k = cfg['channel_mult']
        div = cfg["div"]
        pos_mlp_hidden_dim = cfg["pos_mlp_hidden_dim"]
        attn_mlp_hidden_mult = cfg["attn_mlp_hidden_mult"]
        pre_module_channel = cfg["pre_module"]["channel"]
        pre_module_nsample = cfg["pre_module"]["nsample"]
        self.pre_module = nn.ModuleList([
            MLP(dim=1, in_channel=3, mlp=[pre_module_channel * k] * 2, use_bn=True, skip_last=False),
            PointTransformerResBlock(dim=pre_module_channel * k,
                                     div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                     attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                                     num_neighbors=pre_module_nsample)
        ])
        self.down_module = nn.ModuleList()
        down_cfg = cfg["down_module"]

        last_channel = pre_module_channel
        attn_channel = down_cfg['attn_channel']
        down_sample = down_cfg['nsample']
        for i in range(len(attn_channel)):
            out_channel = attn_channel[i]
            self.down_module.append(PointTransformerDownBlock(npoint=down_cfg['npoint'][i],
                                                              nsample=down_sample[i],
                                                              in_channel=last_channel * k,
                                                              out_channel=out_channel * k,
                                                              num_attn=down_cfg['attn_num'][i],
                                                              div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                              attn_mlp_hidden_mult=attn_mlp_hidden_mult))
            last_channel = out_channel
        up_channel = attn_channel[::-1] + [pre_module_channel]
        up_sample = down_sample[::-1]
        self.up_module = nn.ModuleList()
        up_cfg = cfg["up_module"]
        up_attn_num = up_cfg['attn_num']
        for i in range(len(attn_channel)):
            self.up_module.append(PointTransformerUpBlock(up_sample[i], up_channel[i] * k, up_channel[i + 1] * k, up_attn_num[i],
                                                          div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                          attn_mlp_hidden_mult=attn_mlp_hidden_mult))

        self.heads = nn.ModuleDict()
        head_cfg = cfg['heads']
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=pre_module_channel * k, mlp=mlp[:-1], use_bn=True, skip_last=True,
                                  last_acti=mlp[-1])

    def forward(self, xyz):  # xyz: [B, 3, N]
        xyz_list, points_list = [], []
        points = self.pre_module[0](xyz)
        points = self.pre_module[1](xyz, points)
        xyz_list.append(xyz)
        points_list.append(points)

        for down in self.down_module:
            xyz, points = down(xyz, points)
            xyz_list.append(xyz)
            points_list.append(points)

        for i, up in enumerate(self.up_module):
            points = up(xyz_list[- (i + 1)], xyz_list[- (i + 2)], points, points_list[- (i + 2)])

        output = {}
        for key, head in self.heads.items():
            output[key] = head(points)

        return output

from .point_transformer_modules import PointTransformerResBlock, PointTransformerDownBlock, PointTransformerUpBlock, MLP

class PointTransformer(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, num_channels_R=2, R_dim=3):
        super(PointTransformer, self).__init__()
        cfg = {
            "channel_mult": 4,
            "div": 4,
            "pos_mlp_hidden_dim": 64,
            "attn_mlp_hidden_mult": 4,
            "pre_module": {
                "channel": 16,
                "nsample": 16
            },
            "down_module": {
                "npoint": [256, 64, 32, 16],
                "nsample": [10, 16, 16, 16],
                "attn_channel": [16, 32, 64, 64],
                "attn_num": [2, 2, 2, 2]
            },
            "up_module": {
                "attn_num": [1, 1, 1, 1]
            },
            "heads": {
                "R": [128, num_channels_R * R_dim, None],
                "T": [128, 3, None],
                "N": [128, 3, 'sigmoid'],
                "M": [128, num_channels_R, 'softmax'],
            }
        }
        k = cfg['channel_mult']
        div = cfg["div"]
        pos_mlp_hidden_dim = cfg["pos_mlp_hidden_dim"]
        attn_mlp_hidden_mult = cfg["attn_mlp_hidden_mult"]
        pre_module_channel = cfg["pre_module"]["channel"]
        pre_module_nsample = cfg["pre_module"]["nsample"]
        self.pre_module = nn.ModuleList([
            MLP(dim=1, in_channel=3, mlp=[pre_module_channel * k] * 2, use_bn=True, skip_last=False),
            PointTransformerResBlock(dim=pre_module_channel * k,
                                     div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                     attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                                     num_neighbors=pre_module_nsample)
        ])
        self.down_module = nn.ModuleList()
        down_cfg = cfg["down_module"]

        last_channel = pre_module_channel
        attn_channel = down_cfg['attn_channel']
        down_sample = down_cfg['nsample']
        for i in range(len(attn_channel)):
            out_channel = attn_channel[i]
            self.down_module.append(PointTransformerDownBlock(npoint=down_cfg['npoint'][i],
                                                              nsample=down_sample[i],
                                                              in_channel=last_channel * k,
                                                              out_channel=out_channel * k,
                                                              num_attn=down_cfg['attn_num'][i],
                                                              div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                              attn_mlp_hidden_mult=attn_mlp_hidden_mult))
            last_channel = out_channel
        up_channel = attn_channel[::-1] + [pre_module_channel]
        up_sample = down_sample[::-1]
        self.up_module = nn.ModuleList()
        up_cfg = cfg["up_module"]
        up_attn_num = up_cfg['attn_num']
        for i in range(len(attn_channel)):
            self.up_module.append(PointTransformerUpBlock(up_sample[i], up_channel[i] * k, up_channel[i + 1] * k, up_attn_num[i],
                                                          div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                          attn_mlp_hidden_mult=attn_mlp_hidden_mult))

        self.heads = nn.ModuleDict()
        head_cfg = cfg['heads']
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=pre_module_channel * k, mlp=mlp[:-1], use_bn=True, skip_last=True,
                                  last_acti=mlp[-1])

    def forward(self, xyz):  # xyz: [B, 3, N]
        xyz_list, points_list = [], []
        points = self.pre_module[0](xyz)
        points = self.pre_module[1](xyz, points)
        xyz_list.append(xyz)
        points_list.append(points)

        for down in self.down_module:
            xyz, points = down(xyz, points)
            xyz_list.append(xyz)
            points_list.append(points)

        for i, up in enumerate(self.up_module):
            points = up(xyz_list[- (i + 1)], xyz_list[- (i + 2)], points, points_list[- (i + 2)])

        output = {}
        for key, head in self.heads.items():
            output[key] = head(points)

        return output


if __name__ == '__main__':
    model = PointTransformer()

    input = torch.randn((1, 1024, 3))

    output = model(input)
    for key, value in output.items():
        print(key, value.shape)
