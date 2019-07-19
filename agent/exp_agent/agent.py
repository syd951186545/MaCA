#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sun yu dong
@contact:
@software: PyCharm
@file: agent.py
@time:
@desc:
"""

import os
from agent.base_agent import BaseAgent
from agent.exp_agent import PPO
import interface
from world import config
import copy
import random
import numpy as np

DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 24
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


class Agent(BaseAgent):
    def __init__(self):
        """
        Init this agent
        :param size_x: battlefield horizontal size
        :param size_y: battlefield vertical size
        :param detector_num: detector quantity of this side
        :param fighter_num: fighter quantity of this side
        """
        BaseAgent.__init__(self)
        self.obs_ind = 'simple_beta'
        if not os.path.exists('model/model_After_file438.pkl'):
            print('Error: agent model data not exist!')
            exit(1)
        self.fighter_model = PPO.Agent_p(ACTION_NUM)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def __reset(self):
        pass

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """

        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                tmp_img_obs = obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = obs_dict['fighter'][y]['info']
                tmp_action = self.fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                # action formation
                true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                if true_action[3] == 0 and true_action[0] - tmp_info_obs[3] * 15!=0:
                    true_action[0] = true_action[0] + (true_action[0] - tmp_info_obs[3] * 15) / abs(
                        true_action[0] - tmp_info_obs[3] * 15) * 15

            fighter_action.append(copy.deepcopy(true_action))
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action
