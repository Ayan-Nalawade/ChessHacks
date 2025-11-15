import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class ChessModel(nn.Module):
    def __init__(self, game, args):
        
        super(ChessModel,self).__init__()
        #Game stuff
        self.planes, self.boardX, self.boardY = game.getBoardSize
        self.numActions = game.getActionSize()
        self.args = args

# Architecture parameters
        self.in_channels = 18
        self.trunk_channels = 128
        self.n_blocks = 6
        
        # Trunk: Input convolution
        self.conv_input = nn.Conv2d(self.in_channels, self.trunk_channels, 
                                     kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(self.trunk_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.trunk_channels) for _ in range(self.n_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(self.trunk_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, self.numActions)
        
        # Value head
        self.value_conv = nn.Conv2d(self.trunk_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        """
        Input: x of shape (batch_size, 18, 8, 8)
        Output: (policy_logits, value)
            - policy_logits: shape (batch_size, 4672) - unnormalized log probabilities
            - value: shape (batch_size, 1) - position evaluation in [-1, 1]
        """
        # Trunk
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 8 * 8)  # Flatten
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 1 * 8 * 8)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value        

