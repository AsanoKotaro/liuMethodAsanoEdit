# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:04:12 2022

@author: asano
"""

def run(env,algorithm_1,algorithm_2,epsilon = 0.1):
    """
    
    Parameters
    ----------
    env:
        環境のインスタンス.
    algorithm_1 : TYPE
        agent 1 の　アルゴリズム.
    algorithm_2 : TYPE
        agent 1 の　アルゴリズム.

    Returns
    -------
    joint_actions.

    """
    s_1 = env.sense_observations('1') # 最初の状態　agent1の状態をゲットする
    s_2 = env.sense_observations('2') # 最初の状態　agent2の状態をゲットする
    joint_actions = []         
    
    actions = [0,1,2,3]
    a = random.uniform(0,1)
    #print(a)
    if a < epsilon :
        a_1_sign = random.choice(actions) # agent1 はランダムに行動を選択する
        a_2_sign = random.choice(actions) # agent2 はランダムに行動を選択する
        #print("ランダム")
    else:    
        a_1_sign = algorithm_1.choose_action(s_1) # agent1 はA3Cにより行動を選択する
        #a_1 = actions[a_1_sign] 
        a_2_sign = algorithm_2.choose_action(s_2) # agent2 はA3Cにより行動を選択する
        #a_2 = actions[a_2_sign] 
    
    joint_actions.append(a_1_sign)
    joint_actions.append(a_2_sign)
    
    return joint_actions

def random_joint_actions(actions):
    joint_actions = []
    action_1 = random.choice(actions)
    action_2 = random.choice(actions)
    joint_actions.append(action_1)
    joint_actions.append(action_2)
    return joint_actions

def data_shuffer(share_percent,data_r1,data_r2):# データ共有率    
    #print(share_percent)
    #print("Sharing model is {}".format(share_model))
    data_num = len(data_r1)
    sample_num = int(share_percent * data_num)
    #print("sample_num is {}".format(sample_num))
    
    
    sample_data_r1 = []
    sample_data_r2 = []
    
    sample_data_r1 = random.sample(data_r1,sample_num)# エージェント1のデータからサンプリングした共有データ
    sample_data_r2 = random.sample(data_r2,sample_num)# エージェント2のデータからサンプリングした共有データ
    
       
            
    #print("sample_data_r1 is {}".format(sample_data_r1))
    #print("sample_data_r2 is {}".format(sample_data_r2))
    #elif share_model == 3 :
        
        
    learning_data_r1 = data_r1 + sample_data_r2 # エージェント2の共有データはエージェント1の学習用
    #print(str(len(self.learning_data_r1)))
    learning_data_r2 = data_r2 + sample_data_r1 # エージェント1の共有データはエージェント2の学習用
    
    return learning_data_r1,learning_data_r2