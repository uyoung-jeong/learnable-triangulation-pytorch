from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

# original code: https://github.com/CalciferZh/minimal-hand/blob/master/wrappers.py
class IKNet_Baseline(nn.Module):
    def __init__(self, args, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        self.args = args
        self.depth = args.iknet_depth
        self.widths = args.iknet_width
        self.in_feautres = self.num_joints * 3 # initial input size of the model
        self.output_size = self.num_joints * 4

        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_features=in_features, out_features=width))
            layers.append(nn.BatchNorm1d(num_features=width))
            layers.append(nn.Sigmoid())
            if i == 0:
                in_features = width

        self.feature_layers = nn.Sequential(*layers)
        self.theta_raw_layer = nn.Linear(in_features=width, out_features=self.output_size)

        self.init_weights(pretrained_path='')

    # 3d_keypoints: [batch_size, n_joints, 3]
    # theta: quaternion format
    def forward(self, 3d_keypoints):
        x = self.feature_layers(3d_keypoints)
        theta_raw = self.theta_raw_layer(x)
        theta_raw = torch.reshape(theta_raw, (-1, self.num_joints, 4))
        eps = torch.finfo(torch.float32).eps
        norm = torch.max(torch.norm([theta_raw, eps], dim=-1, keepdim=True))

        theta_pos = torch.div(theta_raw, norm)
        theta_neg = theta_pos * -1
        theta = torch.where(theta_pose[:,:,0:1] > 0, theta_pos, theta_neg)

        return theta

    def init_weights(self, pretrained_path=''):
        if pretrained_path != '':
            print("ERROR: pretrained model loading is not implemented")
            exit()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, std=1.0e-3)
