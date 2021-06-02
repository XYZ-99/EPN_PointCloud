import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from .pointnet2_modules import knn_point, farthest_point_sample, gather_operation, group_operation, \
    three_nn, three_interpolate


class MLP(nn.Module):
    def __init__(self, dim, in_channel, mlp, use_bn=True, skip_last=True, last_acti=None):
        super(MLP, self).__init__()
        layers = []
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            layers.append(conv(last_channel, out_channel, 1))
            if use_bn and (not skip_last or i != len(mlp) - 1):
                layers.append(bn(out_channel))
            if (not skip_last or i != len(mlp) - 1):
                layers.append(nn.ReLU())
            last_channel = out_channel
        if last_acti is not None:
            if last_acti == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif last_acti == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                assert 0, f'Unsupported activation type {last_acti}'
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PointTransformerTransitionDown(nn.Module):
    # nsample: n_neighbor?
    def __init__(self, npoint, nsample, in_channel, out_channel):
        super(PointTransformerTransitionDown, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp = MLP(dim=2, in_channel=in_channel + 3, mlp=[out_channel], use_bn=True, skip_last=False)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, C, N = xyz.shape
        S = self.npoint
        fps_idx = farthest_point_sample(xyz.permute(0, 2, 1), S).int() # [B, npoint]
        new_xyz = gather_operation(xyz, fps_idx)  # [B, C, S]
        _, group_idx = knn_point(self.nsample, new_xyz.transpose(-1, -2), xyz.transpose(-1, -2))
        grouped_xyz = group_operation(xyz, group_idx)  # [B, C, S, nsample]
        grouped_xyz -= new_xyz.view(B, C, S, 1)
        if points is not None:
            grouped_points = group_operation(points, group_idx)   # [B, D, S, nsample]
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=1)
        else:
            grouped_points = grouped_xyz

        grouped_points = self.mlp(grouped_points)  # [B, D, S, nsample]

        new_points = torch.max(grouped_points, -1)[0]  # [B, D', S]

        return new_xyz, new_points


class PointTransformerTransitionUp(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(PointTransformerTransitionUp, self).__init__()
        self.mlp = MLP(dim=1, in_channel=low_channel, mlp=[high_channel], use_bn=True, skip_last=False)

    def forward(self, xyz_low, xyz_high, points_low, points_high):
        """
        Input:
            xyz_high: input points position data, [B, C, N]
            xyz_low: sampled input points position data, [B, C, S]
            points_high: input points data, [B, high_channel, N]
            points_low: input points data, [B, low_channel, S]
        Return:
            new_points: upsampled points data, [B, high_channel, N]
        """
        xyz_high = xyz_high.permute(0, 2, 1)
        xyz_low = xyz_low.permute(0, 2, 1)

        B, N, C = xyz_high.shape
        _, S, _ = xyz_low.shape

        points_low = self.mlp(points_low)
        if S == 1:
            interpolated_points = points_low.repeat(1, 1, N)
        else:
            dist, idx = three_nn(xyz_high, xyz_low)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = three_interpolate(points_low, idx, weight)  # [B, C, N]

        new_points = interpolated_points + points_high
        return new_points


class PointTransformerLayer(nn.Module):
    def __init__( self, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16):
        super(PointTransformerLayer, self).__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Conv1d(dim, dim * 3, 1, bias=False) # query key value

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_mlp_hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(pos_mlp_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_mlp_hidden_mult, 1),
            nn.ReLU(),
            nn.Conv2d(dim * attn_mlp_hidden_mult, dim, 1),
        )

    def forward(self, xyz, points):
        """
        Seems D = D' ?
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: [B, D', N]
        """
        B, C, N = xyz.shape

        _, group_idx = knn_point(self.num_neighbors, xyz.transpose(-1, -2), xyz.transpose(-1, -2))
        grouped_xyz = group_operation(xyz, group_idx)  # [B, C, N, nsample]
        grouped_xyz = xyz.view(B, C, N, 1) - grouped_xyz  # [B, C, N, nsample]
        rel_pos_emb = self.pos_mlp(grouped_xyz)  # [B, D', N, nsample]
        # get queries, keys, values
        q, k, v = self.to_qkv(points).chunk(3, dim=-2)  # [B, 3 * D', N] -> 3 * [B, D', N]
        qk_rel = q.view(B, -1, N, 1) - group_operation(k, group_idx)  # [B, D', N, nsample]
        v = group_operation(v, group_idx)  # [B, D', N, nsample]

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)  # [B, D', N, nsample]
        attn = sim.softmax(dim=-1)
        agg = torch.sum(attn * v, dim=-1)  # [B, D', N]
        return agg

class BatchMLP(nn.Module):
    """
    :param dim in [1, 2]: the number of dimensions to be convolved
    :param mlp: [], a list of dimensions

    [(nb, )c2, c1(, np, na)] x [nb, c1, np, na]
    # TODO: or maybe [(nb, )c2, c1(, np), na] x [nb, c1, np, na]
    boina, bina -> bona
    -> [nb, c2, np, na]
    """
    def __init__(self, dim, in_channel, mlp, use_bn=False, skip_last=False, last_acti=None):
        super(BatchMLP, self).__init__()
        self.layers = []

        if use_bn or skip_last or last_acti is not None:
            raise NotImplementedError("In BatchMLP: options not implemented.")
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            # TODO: size to be determined
            W = torch.empty((out_channel, last_channel), requires_grad=True)
            # TODO: size and init method to be determined
            B = torch.randn(out_channel, requires_grad=True)
            nn.init.xavier_normal_(W, gain=nn.init.calculate_gain('relu'))
            self.layers.append(nn.Parameter(W)) # TODO: Maybe does not need Parameter()
            self.layers.append(nn.Parameter(B))
            last_channel = out_channel

        # self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        :param: [nb, c1, np, na]
        :return: [nb, c2, np, na]
        """
        nb, c1, np, na = x.shape
        for i, W in enumerate(self.layers):
            if i % 2 == 0:
                # TODO: Unsure about this operation
                W = W[None, :, :, None, None].expand(nb, -1, -1, np, na).to(x.device)
                x = torch.einsum("boina, bina -> bona", W, x)
            else:
                # TODO: Unsure about this operation
                B = W[None, :, None, None].expand(nb, -1, np, na).to(x.device)
                x = x + B

        return x


class PointTransformerBatchLayer(nn.Module):
    """
    :param xyz: [nb, 3, np]
    :param feats: [nb, c1, np, na]

    :return: xyz: [nb, 3, np]
    :return: feats: [nb, c1, np, na]

    [nb, c1, c1(, np, na)] x [nb, c1, np, na] -> [nb, c1, np, na]

    grouped_xyz: [nb, 3, np, nn, na]
    rel_pos_emb: [nb, c1, np, nn, na]
    q, k, v: 3 * [nb, c1, np, na]
    -> qk_rel: [nb, c1, np, nn, na]
    -> v: [nb, c1, np, nn, na]
    v = v + rel_pos_emb

    sim: [nb, c1, np, nn, na]
    -> attn: [nb, c1, np, nn, na]

    new_feats: [nb, c1, np, na]

    [na, 3, 3] x [nb, 3, np] -> [nb, 3, np, na]
    aij,bjn->bina
    """
    def __init__(self, dim, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16):
        super(PointTransformerBatchLayer, self).__init__()
        self.dim = dim
        self.num_neighbors = num_neighbors

        to_qkv = torch.empty((dim*3, dim), requires_grad=True)
        nn.init.xavier_normal_(to_qkv, gain=nn.init.calculate_gain('relu'))
        self.register_parameter("to_qkv", nn.Parameter(to_qkv))

        self.pos_mlp = []
        pos_mlp1 = torch.empty((pos_mlp_hidden_dim, 3), requires_grad=True)
        nn.init.xavier_normal_(pos_mlp1, gain=nn.init.calculate_gain('relu'))
        self.pos_mlp.append(nn.Parameter(pos_mlp1))
        self.pos_mlp.append(nn.ReLU()) # FIXME

        pos_mlp2 = torch.empty((dim, pos_mlp_hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(pos_mlp2, gain=nn.init.calculate_gain('relu'))
        self.pos_mlp.append(nn.Parameter(pos_mlp2))
        # self.pos_mlp = nn.ModuleList(*pos_mlp)

        self.attn_mlp = []
        attn_mlp1 = torch.empty((dim * attn_mlp_hidden_mult, dim), requires_grad=True)
        nn.init.xavier_normal_(attn_mlp1, gain=nn.init.calculate_gain('relu'))
        self.attn_mlp.append(nn.Parameter(attn_mlp1))
        self.attn_mlp.append(nn.ReLU()) # FIXME

        attn_mlp2 = torch.empty((dim, dim * attn_mlp_hidden_mult), requires_grad=True)
        nn.init.xavier_normal_(attn_mlp2, gain=nn.init.calculate_gain('relu'))
        self.attn_mlp.append(nn.Parameter(attn_mlp2))
        # self.attn_mlp = nn.ModuleList(*attn_mlp)

    def forward(self, xyz, feats, anchors):
        """

        :param xyz: [nb, 3, np]
        :param feats: [nb, c, np, na]

        :return: feats: [nb, c, np, na]
        """
        B, C, N = xyz.shape

        _, group_idx = knn_point(self.num_neighbors, xyz.transpose(-1, -2), xyz.transpose(-1, -2))
        grouped_xyz = group_operation(xyz, group_idx)
        grouped_xyz = xyz.view(B, C, N, 1) - grouped_xyz  # [B, C, N, nsample]

        # [na, 3, 3] x [nb, 3, np, nn] -> [nb, 3, np, nn, na]
        grouped_xyz = torch.einsum("aij, bjnk -> binka", anchors, grouped_xyz)
        rel_pos_emb = grouped_xyz
        # [nb, 3, np, nn, na] -> [nb, c, np, nn, na]
        # [c, c0], [nb, c0, np, nn, na]
        # Finally: [nb, dim, np, nn, na]
        for i, mlp in enumerate(self.pos_mlp):
            if i != 1:
                rel_pos_emb = torch.einsum("ji, binka->bjnka", mlp.to(xyz.device), rel_pos_emb)
            else:
                # ReLU
                rel_pos_emb = mlp(rel_pos_emb)

        # [3*dim, dim] x [nb, dim, np, na] -> [nb, 3*dim, np, na]
        qkv = torch.einsum("ji, bina -> bjna", self.to_qkv, feats)
        # Each: [nb, dim, np, na] -> [nb, dim, na, np] -> [nb, dim*na, np]
        q, k, v = torch.transpose(qkv, 2, 3).contiguous().view(B, -1, N).chunk(3, dim=-2)

        # [nb, dim*na, np, nn]
        qk_rel = q.view(B, -1, N, 1) - group_operation(k, group_idx)
        v = group_operation(v, group_idx)
        # [nb, dim*na, np, nn] -> [nb, dim, na, np, nn] -> [nb, dim, np, nn, na]
        v = v.view(B, self.dim, -1, N, self.num_neighbors).permute(0, 1, 3, 4, 2).contiguous()
        v = v + rel_pos_emb
        qk_rel = qk_rel.view(B, self.dim, -1, N, self.num_neighbors).permute(0, 1, 3, 4, 2).contiguous()

        # [nb, dim, np, nn, na] -> [nb, dim, np, nn, na]
        sim = qk_rel + rel_pos_emb
        for i, mlp in enumerate(self.attn_mlp):
            if i != 1:
                sim = torch.einsum("ji, binka->bjnka", mlp.to(xyz.device), sim)
            else:
                # ReLU
                sim = mlp(sim)

        attn = sim.softmax(dim=-2)
        agg = torch.sum(attn * v, dim=-2) # [nb, dim, np, na]
        return agg


class PointTransformerResBlock(nn.Module):
    def __init__( self, dim, div=4, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4, num_neighbors=16):
        super(PointTransformerResBlock, self).__init__()
        mid_dim = dim // div
        self.transformer_layer = PointTransformerLayer(mid_dim, pos_mlp_hidden_dim,
                                                       attn_mlp_hidden_mult, num_neighbors)
        self.before_mlp = nn.Conv1d(dim, mid_dim, 1)
        self.after_mlp = nn.Conv1d(mid_dim, dim, 1)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_points: [B, D', N]
        """
        input_points = points
        points = self.before_mlp(points)
        points = self.transformer_layer(xyz, points)
        points = self.after_mlp(points)
        return input_points + points


class PointTransformerDownBlock(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, num_attn, div=4, pos_mlp_hidden_dim=64, attn_mlp_hidden_mult=4):
        super(PointTransformerDownBlock, self).__init__()
        self.down = PointTransformerTransitionDown(npoint, nsample, in_channel, out_channel)
        attn = []
        for i in range(num_attn):
            attn.append(PointTransformerResBlock(out_channel, div, pos_mlp_hidden_dim, attn_mlp_hidden_mult, nsample))
        self.attn = nn.ModuleList(attn)

    def forward(self, xyz, points):
        xyz, points = self.down(xyz, points)
        for layer in self.attn:
            points = layer(xyz, points)

        return xyz, points


class PointTransformerUpBlock(nn.Module):
    def __init__(self, nsample, low_channel, high_channel, num_attn, div=4, pos_mlp_hidden_dim=64,
                 attn_mlp_hidden_mult=4):
        super(PointTransformerUpBlock, self).__init__()
        self.up = PointTransformerTransitionUp(low_channel, high_channel)
        attn = []
        for i in range(num_attn):
            attn.append(PointTransformerResBlock(high_channel, div, pos_mlp_hidden_dim, attn_mlp_hidden_mult, nsample))
        self.attn = nn.ModuleList(attn)

    def forward(self, xyz_low, xyz_high, points_low, points_high):
        points = self.up(xyz_low, xyz_high, points_low, points_high)
        for layer in self.attn:
            points = layer(xyz_high, points)

        return points
