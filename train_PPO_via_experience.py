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
        self.info_fc = nn.Sequential(
            nn.Linear(6, 256),
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
            nn.Linear(6, 256),
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
        value = self.value(feature)
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
        self.writer = SummaryWriter('./tensorbordX')

        # total learning step
        self.global_batch_counter = 0

        self.cost_his = []
        self.actor_net, self.critic_net = Actor(self.n_actions), BaseLine()
        if self.gpu_enable:
            print('GPU Available!!')
            self.actor_net = self.actor_net.cuda()
            self.critic_net = self.critic_net.cuda()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, p, s_):
        self.s_screen_memory.append(s['screen'])
        self.s_info_memory.append(s['info'])
        self.a_memory.append(a)
        self.p_memory.append(p)
        self.r_memory.append(r)
        self.s_screen_memory_.append(s_['screen'])
        self.s_info_memory_.append(s_['info'])
        self.memory_counter += 1

    def save_model(self, step_counter_str):
        torch.save(self.actor_net.state_dict(), 'model/simple/model_' + step_counter_str + '.pkl')
        # torch.save(self.critic_net.state_dict(), 'model/simple/model_' + step_counter_str + '.pkl')

    def __clear_memory(self):
        self.s_screen_memory.clear()
        self.s_info_memory.clear()
        self.a_memory.clear()
        self.p_memory.clear()
        self.r_memory.clear()
        self.s_screen_memory_.clear()
        self.s_info_memory_.clear()
        self.memory_counter = 0

    def learn(self):
        # pre possess mem
        s_screen_mem = torch.FloatTensor(np.array(self.s_screen_memory))
        s_info_mem = torch.FloatTensor(np.array(self.s_info_memory))
        a_mem = torch.LongTensor(np.array(self.a_memory))
        r_mem = torch.FloatTensor(np.array(self.r_memory))
        r_mem = r_mem.view(self.memory_counter, 1)
        p_mem = torch.FloatTensor(np.array(self.p_memory))
        p_mem = p_mem.view(self.memory_counter, 1)
        s_screen_mem_ = torch.FloatTensor(np.array(self.s_screen_memory_))
        s_info_mem_ = torch.FloatTensor(np.array(self.s_info_memory_))
        if self.gpu_enable:
            s_screen_mem = s_screen_mem.cuda()
            s_info_mem = s_info_mem.cuda()
            a_mem = a_mem.cuda()
            p_mem = p_mem.cuda()
            r_mem = r_mem.cuda()
            s_screen_mem_ = s_screen_mem_.cuda()
            s_info_mem_ = s_info_mem_.cuda()
        for index in BatchSampler(SubsetRandomSampler(range(self.memory_counter)), self.batch_size, True):
            self.global_batch_counter += 1
            # 计算exp的状态动作概率，策略网络计算my的状态动作概率
            my_probs = self.actor_net(s_screen_mem[index], s_info_mem[index])
            exp_action = a_mem[index]
            # ATTACK_IND_NUM = 21
            with torch.no_grad():
                action_id_index = exp_action[:, 0] * ATTACK_IND_NUM + exp_action[:, 3]

            ratio = torch.exp(my_probs[:, action_id_index][:, 0] - p_mem[index])
            # 考虑计算效率要不要拿到for循环外面先计算
            with torch.no_grad():
                target_v = r_mem[index] + self.gamma * self.critic_net(s_screen_mem_[index], s_info_mem_[index])
                advantage = (target_v - self.critic_net(s_screen_mem[index], s_info_mem[index])).detach()

            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
            self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.global_batch_counter)
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            value_loss = F.smooth_l1_loss(self.critic_net(s_screen_mem[index], s_info_mem[index]), target_v)
            self.critic_net_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_net_optimizer.step()
            # self.save_model(str(self.global_batch_counter))
            # print("train_batch = {}\n".format(self.global_batch_counter))

        if self.gpu_enable:
            torch.cuda.empty_cache()
        self.__clear_memory()


if __name__ == '__main__':
    """
    MAP_PATH = 'maps/1000_1000_fighter10v10.map'
    RENDER = True
    MAX_EPOCH = 1000
    BATCH_SIZE = 200
    LR = 0.01  # learning rate
    EPSILON = 0.9  # greedy policy
    GAMMA = 0.9  # reward discount
    """
    from get_experience_sa_distribution import get_train_experience_sarps_
    from get_experience_sa_distribution import get_SA_distribution_from_data

    DETECTOR_NUM = 0
    FIGHTER_NUM = 10
    COURSE_NUM = 24
    ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
    ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM

    agent_amzing1 = Agent_p(ACTION_NUM)
    state_actions_dic = get_SA_distribution_from_data("./experiences")
    for i in range(1000):
        file = "./experiences/" + "fix_rule_experience_{}.json".format(i)
        get_train_experience_sarps_(agent_amzing1, file, state_actions_dic, model="simple")
        agent_amzing1.learn()
        agent_amzing1.save_model("After_file{}".format(i))
        print("train_file_finish : fix_rule_experience_{}.json".format(i))
    print("all experiences has been used to train")
