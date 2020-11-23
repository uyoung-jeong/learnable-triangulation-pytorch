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

        self.num_joints = config.smpl.num_joints

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        self.args = args
        self.config = config
        self.device = device
        self.depth = args.iknet_depth
        self.widths = args.iknet_width
        self.in_features = self.num_joints * 3 # initial input size of the model
        self.angle_dim = 4 if args.output_type == 'quaternion' else 6
        self.output_size = self.num_joints * self.angle_dim

        in_features = self.in_features
        layers = []
        for i in range(self.depth):
            layers.append(nn.Linear(in_features=in_features, out_features=self.widths))
            if args.batchnorm == 1:
                layers.append(nn.BatchNorm1d(num_features=self.widths))
            
            activation_type = args.activation.lower()
            if activation_type == 'leakyrelu':
                layers.append(nn.LeakyReLU(inplace=False))
            elif activation_type == 'relu':
                layers.append(nn.ReLU(inplace=False))
            elif activation_type == 'softsign':
                layers.append(nn.Softsign())
            else:
                layers.append(nn.Sigmoid())

            if i == 0:
                in_features = self.widths

        self.feature_layers = nn.Sequential(*layers)

        self.theta_raw_layer = nn.Linear(in_features=self.widths, out_features=self.output_size)

        self.init_weights(pretrained_path=config.model.checkpoint)

    # keypoints_3d: [batch_size, n_joints, 3]
    # theta: quaternion format
    def forward(self, keypoints_3d):
        x = keypoints_3d
        if len(x.shape) == 3: # need to flatten joint array
            x = torch.reshape(x, (-1, self.in_features))

        x = self.feature_layers(x)
        theta_raw = self.theta_raw_layer(x)
        theta_raw = torch.reshape(theta_raw, (-1, self.num_joints, self.angle_dim))

        if self.args.norm_raw_theta == 1:
            eps = torch.zeros((*theta_raw.shape[:2], 1)) + torch.finfo(torch.float32).eps

            norm = torch.max(torch.norm(theta_raw, dim=-1, keepdim=True), eps.to(self.device))

            theta_pos = torch.div(theta_raw, norm)
            theta_neg = theta_pos * -1
            return torch.where(theta_pos[:,:,0:1] > 0, theta_pos, theta_neg)
        else: 
            return torch.where(theta_raw[:,:,0:1] > 0, theta_raw, theta_raw * -1)

    def init_weights(self, pretrained_path=''):
        if pretrained_path != '':
            pretrained_dict = torch.load(pretrained_path)
            self.load_state_dict(pretrained_dict, strict=True)
            print("Successfully loaded pretrained weights of IKNet_Baseline")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                #nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight)
                #nn.init.uniform_(m.weight)
