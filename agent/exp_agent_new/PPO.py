import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Actor(nn.Module):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.conv3 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 * 12 * 64
        )

        self.img_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((64 * 12 * 12), 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )

        self.info_fc = nn.Sequential(
            nn.Linear(6, 1024),
            nn.Tanh(),
        )
        self.decison = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_actions),
        )

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        img_feature = self.conv3(img_feature)
        img_feature = self.img_fc(img_feature.view(img_feature.size(0), -1))
        info_feature = self.info_fc(info)

        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        actions = self.decison(combined)
        action_prob = F.softmax(actions)
        return action_prob


class BaseLine(nn.Module):

    def __init__(self):
        super(BaseLine, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.conv3 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 * 12 * 64
        )


        self.info_fc = nn.Sequential(
            nn.Linear(6, 1024),
            nn.Tanh(),
        )
        self.img_fc = nn.Sequential(  # 64*12*12
            nn.Linear((64 * 12 * 12), 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )

        self.value = nn.Sequential( # 1024+1024
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        img_feature = self.conv3(img_feature)
        img_feature = self.img_fc(img_feature.view(img_feature.size(0), -1))
        info_feature = self.info_fc(info)

        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        value = self.value(combined)
        return value


class Agent_p:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            batch_size=200,
            clip_param=0.2,
            max_grad_norm=0.5,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.batch_size = batch_size
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0
        self.s_screen_memory = []
        self.s_info_memory = []
        self.a_memory = []
        self.p_memory = []
        self.r_memory = []
        self.s_screen_memory_ = []
        self.s_info_memory_ = []

        self.gpu_enable = torch.cuda.is_available()


        # total learning step
        self.global_batch_counter = 0

        self.cost_his = []
        self.actor_net = Actor(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!!')
            self.actor_net = self.actor_net.cuda()
            self.actor_net.load_state_dict(torch.load('model/new_net/model_After_file98.pkl'))
            # self.critic_net = self.critic_net.cuda()

    def choose_action(self, img_obs, info_obs):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        action_probs = self.actor_net(img_obs, info_obs)
        a = Categorical(action_probs)
        action = a.sample()
        if self.gpu_enable:
            action = action.cpu()
        prob = action_probs[:, action.item()].item()
        return action
