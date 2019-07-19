from train.simple.ppo2 import Agent_p
import os
import copy
import numpy as np
from agent.fix_rule.agent import Agent
from interface import Environment
import json
from obs_construct.simple import construct

# 1v1 和2v2直接向前冲对死，每局都是一样的情况，学不到更多的状态
MAP_PATH = 'maps/1000_1000_fighter10v10.map'

RENDER = True
MAX_EPOCH = 5000
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


def store_transition(file, s, a, r, s_):
    exp = json.dumps([s, a, str(r), s_])
    file.write(exp)
    file.write("\n")


if __name__ == "__main__":
    # create blue agent 蓝色为固定规则，红色为PPO2
    blue_agent = Agent()
    red_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = red_agent.get_obs_ind()
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    red_agent.set_map_info(size_x, size_y, red_detector_num, red_fighter_num)

    record_experience_num = -1
    file_order = 197
    # execution
    for x in range(MAX_EPOCH):
        step_cnt = 0
        env.reset()
        while True:
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()
            # get action
            # get  action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            red_action = red_agent.get_action(red_obs_dict, step_cnt)
            red_detector_action = red_action[0]
            red_fighter_action = red_action[1]

            # step
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            transed_red_action = []
            for agent_x in red_fighter_action:
                # agent_x["course"] //= 15
                true_action = [0, 1, 0, 0]
                true_action[0] = int(agent_x["course"] // 15)
                if agent_x["course"] == 360:  # 无语，明明说好了0-359，居然还出现了360
                    true_action[0] = 23
                if agent_x["hit_target"] == 0:
                    true_action[3] = 0
                elif agent_x["missile_type"] == 2:
                    # 根据导弹类型0-2和打击目标编号1-10，对应到0-20的战斗目标编号
                    true_action[3] = int(agent_x["hit_target"] + FIGHTER_NUM)
                else:
                    true_action[3] = int(agent_x["hit_target"] * agent_x["missile_type"])
                if true_action[0] < 0 or true_action[0] > 23 or true_action[3] > 20 or true_action[3] < 0:
                    raise Exception("action problem")

                transed_red_action.append(true_action)

            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            # save repaly(没有侦察机)

            current_red_obs = copy.deepcopy(red_obs_dict)
            red_obs_dict, blue_obs_dict = env.get_obs()
            next_red_obs = red_obs_dict

            record_experience_num += 1
            if record_experience_num % 9000 == 0:
                file_order += 3
                if record_experience_num != 0: experience_file.close()
                experience_file = open("./experiences/fix_rule_experience_{}.json".format(file_order), "a")

            # store_transition(experience_file, current_red_obs, transed_red_action, list(fighter_reward), next_red_obs)

            # until done, perform a learn

            if step_cnt > 300:
                break

            step_cnt += 1
