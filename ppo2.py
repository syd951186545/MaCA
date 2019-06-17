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

# Parameters
parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()
Transition = namedtuple('Transition', ['state', 'move_action', 'attack_action', 'a_log_prob', 'reward', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


class Actor(nn.Module):
    def __init__(self, n_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
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
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        actions = self.decision_fc(feature)
        action_prob = F.softmax(actions)
        return action_prob


class BaseLine(nn.Module):

    def __init__(self):
        super(BaseLine, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
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
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        value = self.self.value(feature)
        return value


class Agent_p:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=100,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory_counter = 0
        self.s_screen_memory = []
        self.s_info_memory = []
        self.a_memory = []
        self.p_memory = []
        self.r_memory = []
        self.s__screen_memory = []
        self.s__info_memory = []

        self.gpu_enable = torch.cuda.is_available()

        # total learning step
        self.learn_step_counter = 0

        self.cost_his = []
        self.actor_net, self.critic_net = Actor(self.n_actions), BaseLine()
        if self.gpu_enable:
            print('GPU Available!!')
            self.actor_net = self.actor_net.cuda()
            self.critic_net = self.critic_net.cuda()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

    def store_transition(self, s, a, p, r, s_):
        self.s_screen_memory.append(s['screen'])
        self.s_info_memory.append(s['info'])
        self.a_memory.append(a)
        self.p_memory.append(p)
        self.r_memory.append(r)
        self.s__screen_memory.append(s_['screen'])
        self.s__info_memory.append(s_['info'])
        self.memory_counter += 1

    def __clear_memory(self):
        self.s_screen_memory.clear()
        self.s_info_memory.clear()
        self.a_memory.clear()
        self.p_memory.clear()
        self.r_memory.clear()
        self.s__screen_memory.clear()
        self.s__info_memory.clear()
        self.memory_counter = 0

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
        return action, prob

    def learn(self):
        # pre possess mem
        s_screen_mem = torch.FloatTensor(np.array(self.s_screen_memory))
        s_info_mem = torch.FloatTensor(np.array(self.s_info_memory))
        a_mem = torch.LongTensor(np.array(self.a_memory))
        r_mem = torch.FloatTensor(np.array(self.r_memory))
        r_mem = r_mem.view(self.memory_counter, 1)
        p_mem = torch.FloatTensor(np.array(self.p_memory))
        p_mem = p_mem.view(self.memory_counter, 1)
        s_screen_mem_ = torch.FloatTensor(np.array(self.s__screen_memory))
        s_info_mem_ = torch.FloatTensor(np.array(self.s__info_memory))
        if self.gpu_enable:
            s_screen_mem = s_screen_mem.cuda()
            s_info_mem = s_info_mem.cuda()
            a_mem = a_mem.cuda()
            p_mem = p_mem.cuda()
            r_mem = r_mem.cuda()
            s_screen_mem_ = s_screen_mem_.cuda()
            s_info_mem_ = s_info_mem_.cuda()

        with torch.no_grad():
            target_v = r_mem + args.gamma * self.critic_net(s_screen_mem_, s_info_mem_).gather(1, a_mem)

        advantage = (target_v - self.critic_net(s_screen_mem, s_info_mem)).detach()

        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(len(a_mem)), self.batch_size, True)):
                # epoch iteration, PPO core!!!
                my_action, my_prob = self.actor_net(s_screen_mem[index], s_info_mem[index])
                ratio = torch.exp(my_prob - p_mem[index])

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(s_screen_mem[index], s_info_mem[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        self.__clear_memory()