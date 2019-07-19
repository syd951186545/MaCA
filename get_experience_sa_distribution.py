import os
import json
from obs_construct.simple_beta import construct

size_x = 1000
size_y = 1000
red_detector_num = 0
red_fighter_num = 10


def get_SA_distribution_from_data(folder="./experiences"):
    state_actions_dic = {}
    for _, _, files in os.walk(folder, topdown=False):
        for file in files:
            with open(folder + "/" + file, "r") as f_json:
                for line in f_json.readlines():
                    empty_list = []
                    exp = json.loads(line)
                    current_obs_dic = exp[0]
                    action_list = exp[1]

                    for y in range(red_fighter_num):
                        if not current_obs_dic["fighter_obs_list"][y]["alive"]:
                            continue
                        cod = current_obs_dic["fighter_obs_list"][y]
                        cod.pop("id")  # 删除观测中的id编号干扰
                        cod.pop("last_action")  # 删除观测中的last_action编号干扰,构建simple_obs时单独加
                        current_obs = str(cod.values())
                        action_ = tuple(action_list[y])
                        state_actions_dic.setdefault(current_obs, empty_list)
                        state_actions_dic[current_obs].append(action_)
    return state_actions_dic


def get_train_experience_sarps_(fighter_model, file, state_actions_dic, model="simple_beta"):
    """
    :param state_actions_dic: samples for statistics distribution
    :param fighter_model: your agent using experience,which needs conclude a func named:store_transition
    :param model: simple or else is raw
    :return:
    """
    with open(file, "r") as f_json:
        for line in f_json.readlines():
            exp = json.loads(line)
            current_obs_dic = exp[0]
            action_list = exp[1]
            reward_list = exp[2]
            reward_list = reward_list[1:-1].split(",")
            next_obs_dic = exp[3]
            simple_obs_class = construct.ObsConstruct(size_x, size_y, red_detector_num, red_fighter_num)
            tem_simple_obs_dic = simple_obs_class.obs_construct(current_obs_dic)
            next_simple_obs_dic = simple_obs_class.obs_construct(next_obs_dic)
            for y in range(red_fighter_num):
                if not current_obs_dic["fighter_obs_list"][y]["alive"]:
                    continue
                # s,a raw to calculate probility
                cod = current_obs_dic["fighter_obs_list"][y]
                cod.pop("id")
                cod.pop("last_action")  # 删除观测中的last_action编号干扰,构建simple_obs时单独加
                current_obs = str(cod.values())
                action_ = tuple(action_list[y])
                pro = float(state_actions_dic[current_obs].count(action_) / len(state_actions_dic[current_obs]))
                # a,r
                tmp_action = action_list[y]
                tmp_reward = float(reward_list[y])

                # store img obs
                if model == "simple_beta":
                    # s
                    tem_simple_obs = {}
                    tem_simple_obs["screen"] = tem_simple_obs_dic['fighter'][y]['screen']
                    tem_simple_obs["screen"] = tem_simple_obs["screen"].transpose(2, 0, 1)
                    tem_simple_obs["info"] = tem_simple_obs_dic['fighter'][y]['info']

                    # tmp_alive_obs = tem_simple_obs_dic['fighter'][y]['alive']
                    # s_
                    next_simple_obs = {}
                    next_simple_obs["screen"] = next_simple_obs_dic['fighter'][y]['screen']
                    next_simple_obs["screen"] = next_simple_obs["screen"].transpose(2, 0, 1)
                    next_simple_obs["info"] = next_simple_obs_dic['fighter'][y]['info']
                    # tmp_alive_obs = next_simple_obs_dic['fighter'][y]['alive']
                    fighter_model.store_transition(tem_simple_obs, tmp_action, tmp_reward, pro,
                                                   next_simple_obs)
                # store raw obs
                else:
                    fighter_model.store_transition(current_obs_dic["fighter_obs_list"][y], tmp_action,
                                                   tmp_reward,
                                                   pro, next_obs_dic["fighter_obs_list"][y])


if __name__ == '__main__':
    folder = "./experiences"
    fighter_model = None
    state_actions_dic = get_SA_distribution_from_data()
    file = "./experiences/fix_rule_experience_0.json"
    get_train_experience_sarps_(fighter_model, file, state_actions_dic, )
