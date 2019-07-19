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

        self.value = nn.Sequential(  # 1024+1024
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
            max_grad_norm=10,
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
        torch.save(self.actor_net.state_dict(), 'model/new_net/model_' + step_counter_str + '.pkl')
        # torch.save(self.critic_net.state_dict(), 'model/simple/model_' + step_counter_str + '.pkl')

    def __clear_memory(self):
        self.s_screen_memory = []
        self.s_info_memory = []
        self.a_memory = []
        self.p_memory = []
        self.r_memory = []
        self.s_screen_memory_ = []
        self.s_info_memory_ = []
        self.memory_counter = 0

    def learn(self):
        # pre possess mem
        self.s_screen_memory = torch.FloatTensor(np.array(self.s_screen_memory))
        self.s_info_memory = torch.FloatTensor(np.array(self.s_info_memory))
        self.a_memory = torch.LongTensor(np.array(self.a_memory))
        self.r_memory = torch.FloatTensor(np.array(self.r_memory))
        self.r_memory = self.r_memory.view(self.memory_counter, 1)
        self.p_memory = torch.FloatTensor(np.array(self.p_memory))
        self.p_memory = self.p_memory.view(self.memory_counter, 1)
        self.s_screen_memory_ = torch.FloatTensor(np.array(self.s_screen_memory_))
        self.s_info_memory_ = torch.FloatTensor(np.array(self.s_info_memory_))
        for index in BatchSampler(SubsetRandomSampler(range(self.memory_counter)), self.batch_size, True):
            self.global_batch_counter += 1

            if self.gpu_enable:
                s_screen_mem = self.s_screen_memory[index].cuda()
                s_info_mem = self.s_info_memory[index].cuda()
                a_mem = self.a_memory[index].cuda()
                p_mem = self.p_memory[index].cuda()
                r_mem = self.r_memory[index].cuda()
                s_screen_mem_ = self.s_screen_memory_[index].cuda()
                s_info_mem_ = self.s_info_memory_[index].cuda()
            else:
                s_screen_mem = self.s_screen_memory[index]
                s_info_mem = self.s_info_memory[index]
                a_mem = self.a_memory[index]
                p_mem = self.p_memory[index]
                r_mem = self.r_memory[index]
                s_screen_mem_ = self.s_screen_memory_[index]
                s_info_mem_ = self.s_info_memory_[index]

            # 计算exp的状态动作概率，策略网络计算my的状态动作概率
            my_probs = self.actor_net(s_screen_mem, s_info_mem)
            # ATTACK_IND_NUM = 21
            with torch.no_grad():
                action_id_index = a_mem[:, 0] * ATTACK_IND_NUM + a_mem[:, 3]
                if torch.min(action_id_index) < 0 or torch.max(action_id_index) >= 504:
                    print(torch.min(action_id_index))
                    print(torch.max(action_id_index))
                    raise Exception("坑1")

            ratio = (my_probs.gather(1, action_id_index.view(self.batch_size, 1)) / p_mem)
            with torch.no_grad():
                target_v = r_mem + self.gamma * self.critic_net(s_screen_mem_, s_info_mem_)
            # target_v = 当前状态采取动作后的价值，减去当前状态价值。表示了动作的adv
            advantage = (target_v - self.critic_net(s_screen_mem, s_info_mem)).detach()

            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
            self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.global_batch_counter)
            self.actor_optimizer.zero_grad()
            action_loss.backward()
            # 限制梯度的最大范数，防止梯度爆炸
            # nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            value_loss = F.smooth_l1_loss(self.critic_net(s_screen_mem, s_info_mem), target_v)
            self.critic_net_optimizer.zero_grad()
            value_loss.backward()
            # nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_net_optimizer.step()

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
    LEARN_INTERVAL = 100

    agent_amzing1 = Agent_p(ACTION_NUM)
    state_actions_dic = get_SA_distribution_from_data("./experiences")
    for i in range(34):
        file = "./experiences/" + "fix_rule_experience_{}.json".format(i)
        get_train_experience_sarps_(agent_amzing1, file, state_actions_dic, model="simple_beta")
        agent_amzing1.learn()
        agent_amzing1.save_model("After_file{}".format(i))
        print("train_file_finish : fix_rule_experience_{}.json".format(i))
    print("all experiences has been used to train")
