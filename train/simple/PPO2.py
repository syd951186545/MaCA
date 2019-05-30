import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

import gym
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
                in_channels=3,
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
        move_action_prob = F.softmax(actions[:15])  # （360度）16方位移動
        attack_action_prob = F.softmax(actions[16:])  # 0不攻擊，剩下為长期攻击目标，短期攻击目标
        return move_action_prob, attack_action_prob


class BaseLine(nn.Module):

    def __init__(self):
        super(BaseLine, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
            nn.Conv2d(
                in_channels=3,
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


class Agent:
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
        self.training_step = 0
        self.buffer = []

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

    def store_transition(self, s, ma, aa, log_p, r, s_):
        self.buffer.append(Transition(s, ma, aa, r, log_p, s_))
        self.memory_counter += 1

    def __clear_memory(self):
        self.buffer.clear()
        self.memory_counter = 0

    def choose_action(self, img_obs, info_obs):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        move_action_prob, attack_action_prob = self.actor_net(img_obs, info_obs)
        m = Categorical(move_action_prob)
        move_action = m.sample()
        a = Categorical(attack_action_prob)
        attack_action = a.sample()
        if self.gpu_enable:
            move_action = move_action.cpu()
            attack_action = attack_action.cpu()
        prob = move_action_prob[:, move_action.item()].item() * attack_action_prob[:, attack_action.item()].item()
        return move_action, attack_action, prob

    def learn(self):
        self.training_step += 1

        s_screen = torch.FloatTensor(t.state["screen"] for t in self.buffer)
        s_info = torch.FloatTensor(t.state["info"] for t in self.buffer)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_s_screen = torch.FloatTensor(t.next_state["screen"] for t in self.buffer)
        next_s_info = torch.FloatTensor(t.next_state["info"] for t in self.buffer)

        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net(next_s_screen, next_s_info)

        advantage = (target_v - self.critic_net(s_screen, s_info)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity), self.batch_size, True)):
                # epoch iteration, PPO core!!!
                ma_probs, aa_probs = self.actor_net(s_screen[index], s_info[index])
                ma_prob = ma_probs[action[index][0]]
                aa_prob = aa_probs[action[index][3]]
                ratio = torch.exp(ma_prob * aa_prob - old_action_log_prob)

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(s_screen[index], s_info[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]
