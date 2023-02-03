# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:04:28 2022

@author: asano
"""
import threading
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import copy
import time
import csv

from OtherMethod import run,random_joint_actions,data_shuffer
from ACNet import ACNet
from Worker import Worker
from Maze_1 import Maze_1#(Maze_1モジュール(ファイル)内のMaze_1メソッドをインポートした)
from Maze_2 import Maze_2
from Maze_2A import Maze_2A
from Maze_2AB import Maze_2AB

if __name__ == "__main__":
    
    start_time = time.time() #実行開始時間
    
    GAMMA = 0.9
    ENTROPY_BETA = 0.001
    LR_A = 0.001    # Actorの学習率
    LR_C = 0.001    # Criticの学習率
    
    
        
    
    #--------------------実験パラメータ------------------------
    
    trials = 10 # 試行回数
    episode_num = 100 #　エピソードの数
    max_steps_num = 40#毎エピソードのステップの数の上限    迷路環境1：40　迷路環境2，2(A),2(AB):100
    is_share_data = True#データ共有：True 共有しない：False
    N_WORKERS = 4# マルチスレッドのスレッド数
    
    share_percent = 0.25#経験データ共有率X_A
    
  
    env = Maze_1()#迷路環境1　(Maze_1をインスタンス化)
    #env = Maze_2()#迷路環境2
    #env = Maze_2A()#迷路環境2A
    #env = Maze_2AB()#迷路環境2AB
    
    #--------------------実験パラメータ------------------------
    
    
    
    
    actions = env.actions # 行動集合['up','down','left','right'](Maze_1 メソッド内にあり)
    
    episodes_buffer_s1,episodes_buffer_a1 = [], []# 各episodeのrobot1 の状態、行動を記録する
    episodes_buffer_s2,episodes_buffer_a2 = [], []# 各episodeのrobot2 の状態、行動を記録する
    episodes_buffer_r = []# 共通の報酬を記録する
    episodes_dones = []
    last_obs_r1 = []# 各episodeのrobot1 の最終状態を記録する
    last_obs_r2 = []# 各episodeのrobot2 の最終状態を記録する
     
       
    
    trials_episodes_steps = []
    trials_episodes_rewards = []
    GOAL_EPISODES = []
    one_goal_episodes = []
    trap_episodes = []
    
    trial_one_goal_episodes_in_episodes = []#全試行の毎エピソード中，１つだけゴールしたエピソードの数を記録する.eg.123エピソード目の時、ゴールエピソードの数を記録する
    trial_two_goal_episodes_in_episodes = []#全試行の毎エピソード目の中2つだけゴールエピソードを記録する.eg.123エピソード目の時、ゴールエピソードの数を記録する
    trial_trap_episodes_in_episodes = []#全試行の毎エピソード目の中罠にはまったゴールエピソードを記録する.
    
    
    
    str_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(start_time))#strftimeは日付や時間を文字列に変更
    path = 'learning_data/log'+str(str_time)+'.csv'
 
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ['trial','Episode','reward','steps','two','one_more','trap']
        csv_write.writerow(csv_head)
      
    
    
    
    for trial in range(trials):  
        
        
        tf.compat.v1.reset_default_graph()#デフォルトのグラフスタックをクリアし、グローバルデフォルトのグラフをリセットします。
        SESS = tf.compat.v1.Session() # tensorflowのsessionを作る．TensorFlow操作を実行するためのクラスです．
        #SESS = tf.keras.backend.get_session()
        
        
        
        
        OPT_A = tf.compat.v1.train.RMSPropOptimizer(LR_A, name='RMSPropA') # Actorのoptimizerを生成する（RMSProp）　Root mean squared propagation optimizer
        OPT_C = tf.compat.v1.train.RMSPropOptimizer(LR_C, name='RMSPropC') # Criticのoptimizerを生成する（RMSProp）
        
        GOAL_EPISODE_R = [] # 目標達成のエピソードの報酬和を保存する
        GOAL_EP = 0 # 目標達成のエピソードの数
        one_goal_ep = 0 #一つゴールに到達したエピソードの数
        trap_ep = 0#罠にはまったエピソード数
        
        one_goal_episodes_in_episodes = [] #毎エピソード目の中１つゴールエピソードを記録する
        two_goal_episodes_in_episodes = [] #毎エピソード目の中2つゴールエピソードを記録する
        trap_episodes_in_episodes = [] #毎エピソード目の中罠にはまったエピソードを記録する
        
        
        robot1_name = 'robot_1' 
        robot2_name = 'robot_2' 
        
        GLOBAL_NET_SCOPE_1 = 'Global_Net_1'# パラメータサーバーのNNの名前
        GLOBAL_NET_SCOPE_2 = 'Global_Net_2'# パラメータサーバーのNNの名前    
        GLOBAL_AC_1 = ACNet(GLOBAL_NET_SCOPE_1)  # robot1のパラメータサーバーを生成する
        GLOBAL_AC_2 = ACNet(GLOBAL_NET_SCOPE_2)  # robot2のパラメータサーバーを生成する
        
        AC_1 = ACNet(robot1_name, GLOBAL_AC_1)# robot1のACネットワークを生成する
        AC_2 = ACNet(robot2_name, GLOBAL_AC_2)# robot2のACネットワークを生成する
        
        # Coordinatorにより、Sessionの中にある多数のThreadを管理する
        # tf.train.Coordinator()によりインスタンスを生成する
        inif = tf.compat.v1.global_variables_initializer()
        COORD = tf.train.Coordinator()
        SESS.run(inif) # 全ての変数を初期化する   
       
        
        print("試行 "+str(trial+1)+" 開始!")
        
        episodes_steps = [] #　全てのepisodeの歩数を記録する
        episodes_rewards = [] #　全てのepisodeの累積報酬を記録する
        
        episodes_buffer_data_r1 = []
        episodes_buffer_data_r2 = []
        
        
       
        for episode in range(episode_num):
            

            
            #print("Episode "+str(episode+1)+" starts!")
            
            s_1 = env.sense_observations('1') # 最初の状態　robot1の状態をゲットする
            s_2 = env.sense_observations('2') # 最初の状態　robot2の状態をゲットする
           
            
            buffer_s1, buffer_a1 = [], []# robot1 の状態、行動を記録する
            buffer_s2, buffer_a2 = [], []# robot2 の状態、行動を記録する
            buffer_r = []# 共通の報酬を記録する
            
            data_buffer = [] #全てのデータを記録する
            
            done_sign = False
            
            total_steps = 0 # episodeの歩数を記録する
            ep_r = 0 # episodeのrewardを累積する(共通の報酬和)
            
            buffer_data_r1 = [] # robot1 のデータを記録 s,a,s_,r,done
            buffer_data_r2 = [] # robot1 のデータを記録
            
            
            
            result_log = []
            
            for step in range(max_steps_num):
                
                #data.appendでdataの末尾に要素を追加
                data = []# excel書き込み内容
                data.append(str(trial+1))#trial
                data.append(str(episode+1))#episode
                #data.append(str(step+1))#step
                
                joint_actions = []
                
                if episode/episode_num > 0.75 :#半分エピソードの後
                    joint_actions_sign = run(env, AC_1, AC_2,0.01) # epsilon = 0.01
                elif episode/episode_num > 0.5 :
                    joint_actions_sign = run(env, AC_1, AC_2,0.05) # epsilon = 0.05
                else:
                    joint_actions_sign = run(env, AC_1, AC_2) # joint actions のインデックスを獲得する　eg.[0,1]
                    
                
                joint_actions.append(actions[joint_actions_sign[0]])
                joint_actions.append(actions[joint_actions_sign[1]])
                
                #joint_actions = random_joint_actions(actions)
                
                #data.append(str(joint_actions[0]))#robot1 action
                #data.append(str(joint_actions[1]))#robot2 action
                
                #print("Episode "+str(episode+1)+" : step "+str(step+1)+" robot 1 action: "+str(joint_actions[0])+" robot 2 action: "+str(joint_actions[1]))    
                s_, r, episode_done,goals_num = env.step(joint_actions)# 次の状態、報酬、目標に到達標識をゲットする
                
                
                
                #　通路の報酬を変える。0～1/4ステップ数：0.04，1/4~1/2ステップ数：-0.01,1/2ステップ数以後:-0.04
                
                if (step+1) >= 0 and (step+1) < (max_steps_num*(1/4)) :
                    #print("今は0～1/4ステップ数：0.04")
                    if r == -1.04 :
                        r = -0.96
                    elif r == -0.14 :
                        r = -0.06
                    elif r == -0.08 :
                        r = 0.08
                    elif r == 9.96 :
                        r = 10.04
                    elif r == 19.96 :
                        r = 20.04
                    else :
                        r = r
                elif (step+1) >= (max_steps_num*(1/4)) and (step+1) < (max_steps_num*(1/2)) :
                    #print("今は1/4~1/2ステップ数：-0.01")
                    if r == -1.04 :
                        r = -1.01
                    elif r == -0.14 :
                        r = -0.11
                    elif r == -0.08 :
                        r = -0.02
                    elif r == 9.96 :
                        r = 9.99
                    elif r == 19.96 :
                        r = 19.99
                    else :
                        r = r
                else:
                    #print("今は1/2ステップ数以後:-0.04")
                    r = r
                
                s1_ = s_[0] # robot1 の次状態
                s2_ = s_[1] # robot2 の次状態
                
                ep_r += r # 報酬を累積する
                total_steps += 1 # 歩数を累積する
                
                buffer_s1.append(s_1)##遷移後の状態を記録
                buffer_a1.append(joint_actions_sign[0])##選択した行動を記録
                #buffer_a1.append(joint_actions[0])
               
                buffer_s2.append(s_2)
                buffer_a2.append(joint_actions_sign[1])
                #buffer_a2.append(joint_actions[1])
                 
                buffer_r.append(r)##得た報酬を記録
                
                s_1 = s1_ ##状態変数に次状態を上書き
                s_2 = s2_ 
                
                env.render()##time.sleep(0.01)を行っている0.01秒処理を停止する
                
                             
                ##episode_done＝Trueで環境をリセット　
                if episode_done: 
                    
                    
                    env.reset() # 環境をリセットする
                    
                    done_sign = True
                    
                    
                    break
            ##---------------------------------------------------------1Episode終了-----------------------------------------------------
            if goals_num == 2:# 目標達成する
                GOAL_EPISODE_R.append(ep_r)
                GOAL_EP += 1 
            elif goals_num == 1:
                one_goal_ep += 1
            elif goals_num == -3:
                trap_ep += 1
            elif goals_num == -2:
                trap_ep += 1
                one_goal_ep += 1
            elif goals_num == -1:
                trap_ep += 1
                GOAL_EPISODE_R.append(ep_r)
                GOAL_EP += 1 
                
            one_goal_episodes_in_episodes.append(one_goal_ep) #毎エピソードの１回のみゴールしたエピソードの回数を記録する
            two_goal_episodes_in_episodes.append(GOAL_EP) #毎エピソード目の中2つゴールエピソードを記録する
            trap_episodes_in_episodes.append(trap_ep)#毎エピソード目の中罠にはまったエピソードを記録する
                
            
            
            if done_sign == False: # 目標達成していない　最大歩数使い切った
                env.reset() # 環境をリセットする
            
            #print("Episode "+str(episode+1)+" is over!")
            result_log.append(str(trial+1))#trial
            result_log.append(str(episode+1))#episode
            result_log.append(str(ep_r))#total_reward
            result_log.append(str(total_steps))#total_step
            
            result_log.append(str(GOAL_EP))#2goals
            result_log.append(str(GOAL_EP + one_goal_ep))#one more
            result_log.append(str(trap_ep))
            
            with open(path,'a') as f:#'a'は追記モード
                    csv_write = csv.writer(f)
                    csv_write.writerow(result_log)
            
            episodes_steps.append(total_steps)
            episodes_rewards.append(ep_r)
            
            
            
            buffer_data_r1.append(buffer_s1)
            buffer_data_r1.append(buffer_a1)
            buffer_data_r1.append(s1_)
            buffer_data_r1.append(buffer_r)
            buffer_data_r1.append(episode_done)
            
            buffer_data_r1.append(ep_r)
            
            episodes_buffer_data_r1.append(buffer_data_r1)
            
            buffer_data_r2.append(buffer_s2)
            buffer_data_r2.append(buffer_a2)
            buffer_data_r2.append(s2_)
            buffer_data_r2.append(buffer_r)
            buffer_data_r2.append(episode_done)
            
            buffer_data_r2.append(ep_r)
            
            episodes_buffer_data_r2.append(buffer_data_r2)
    
            learning_episodes_buffer_data_r1 = copy(episodes_buffer_data_r1)
            learning_episodes_buffer_data_r2 = copy(episodes_buffer_data_r2)
            data_buffer.append(learning_episodes_buffer_data_r1)
            data_buffer.append(learning_episodes_buffer_data_r2)
            
            
            if is_share_data:#データを共有する
                learning_data_r1,learning_data_r2 = data_shuffer(share_percent,learning_episodes_buffer_data_r1,learning_episodes_buffer_data_r2)
            else:
                learning_data_r1,learning_data_r2 = learning_episodes_buffer_data_r1,learning_episodes_buffer_data_r2
                
                
                
                
            #学習部分
            #ランダム方式，マルチスレッド
            if is_share_data:
                
                workers_agen1 = []
                # 各workerを生成する
                for i in range(N_WORKERS): # N_WORKERS の数
                    i_name = 'Worker_agent1_%i' % (i+1)   # workerの名前：eg.Worker_1
                    workers_agen1.append(Worker(i_name, AC_1,learning_data_r1))
                    
                workers_agen2 = []
                # 各workerを生成する
                for i in range(N_WORKERS): # N_WORKERS の数
                    i_name = 'Worker_agent2_%i' % (i+1)   # workerの名前：eg.Worker_1
                    workers_agen2.append(Worker(i_name, AC_2,learning_data_r2))
        
                
                total_workers = []
                total_workers = workers_agen1 + workers_agen2
                
                worker_threads = []
                # 各workerのthreadを生成する
                #print("=========================="+str(N_WORKERS)+"個のスレッドが更新開始===============================")
                for worker in total_workers:# 学習をmulti threadで実行する
                    job = lambda: worker.update_A3C()
                    t = threading.Thread(target=job) # 一つのthreadを生成し、任務を配布する
                    t.start() # threadが起動する
                    worker_threads.append(t)
                COORD.join(worker_threads) #　生成したthreadをCoordinatorに入り、threadが終了するまで待機する
            else:
                
                worker_agent1_name = 'Worker_agent1'
                worker_agent2_name = 'Worker_agent2'
                worker_agent1 = Worker(worker_agent1_name, AC_1,learning_data_r1)
                worker_agent2 = Worker(worker_agent2_name, AC_2,learning_data_r2)
                worker_agent1.update_A3C()
                worker_agent2.update_A3C()
            
            
            #print("=========================================更新完了===========================================")
            
            
        SESS.close()
        

        
        #print("目標達成エピソードの報酬:{}".format(GOAL_EPISODE_R))
        print("試行　{}　中罠にはまったエピソードの数:{}".format(trial+1,trap_ep))
        print("試行　{}　中一つゴールに達成エピソードの数:{}".format(trial+1,one_goal_ep))
        print("試行　{}　中二つゴールに達成エピソードの数:{}".format(trial+1,GOAL_EP))
        
        trials_episodes_steps.append(episodes_steps)
        trials_episodes_rewards.append(episodes_rewards)
        GOAL_EPISODES.append(GOAL_EP)
        one_goal_episodes.append(one_goal_ep)
        trap_episodes.append(trap_ep)
        
        trial_one_goal_episodes_in_episodes.append(one_goal_episodes_in_episodes)
        trial_two_goal_episodes_in_episodes.append(two_goal_episodes_in_episodes)
        trial_trap_episodes_in_episodes.append(trap_episodes_in_episodes)
    
    #print("各試行のエピソードのステップ数:{}".format(trials_episodes_steps))  
    #print("各試行のエピソードの報酬:{}".format(trials_episodes_rewards)) 
    
    #print("全てのdata buffer:{}".format(data_buffer))
    #print("{}".format(len(data_buffer)))
    
    """
    # パラメータを保存する
    
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = "data/multiagent"
    save_file_name = save_path+"/singleA3C"+str(now_time)+""
    saver = tf.compat.v1.train.Saver() 
    saver.save(SESS,save_path=save_file_name)
    print("パラメータを保存した")           
    """  
    #env.destroy()
    print("全試行の平均罠にはまったエピソード数:{}".format(np.mean(trap_episodes)))
    print("全試行の平均一つゴールに達成エピソード数:{}".format(np.mean(one_goal_episodes)))
    print("全試行の平均二つゴール達成エピソード数:{}".format(np.mean(GOAL_EPISODES)))
    print("全試行の平均一つ以上のゴール達成エピソード数:{}".format(np.mean(GOAL_EPISODES) + np.mean(one_goal_episodes)))
    avg_episodes_steps = np.mean(trials_episodes_steps,axis=0)
    avg_episodes_rewards = np.mean(trials_episodes_rewards,axis=0)
    
    avg_trial_one_goal_episodes_in_episodes = np.mean(trial_one_goal_episodes_in_episodes,axis=0)
    avg_trial_two_goal_episodes_in_episodes = np.mean(trial_two_goal_episodes_in_episodes,axis=0)
    avg_trial_trap_episodes_in_episodes = np.mean(trial_trap_episodes_in_episodes,axis=0)
    
    #print(len(avg_trial_two_goal_episodes_in_episodes))
    
                
    
    
    # episodeとその歩数と累積報酬を図に出力する
    
    fig = plt.figure(figsize=(16,4))
    axL = fig.add_subplot(1, 3, 1)
    axC = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3)
    
    axL.plot(np.arange(1,len(avg_episodes_steps)+1), avg_episodes_steps)
    axL.set_title('Steps')
    axL.set_xlabel('episode')
    axL.set_ylabel('Total moving steps')
    
    axC.plot(np.arange(1,len(avg_trial_two_goal_episodes_in_episodes)+1), avg_trial_two_goal_episodes_in_episodes)
    axC.set_title('Goal Episode')
    axC.set_xlabel('episode')
    axC.set_ylabel('Goal Episode')
    
    
    axR.plot(np.arange(1,len(avg_episodes_rewards)+1), avg_episodes_rewards)
    axR.set_title('Reward')
    axR.set_xlabel('episode')
    axR.set_ylabel('Total moving reward')
    fig.show()            
    
    env.destroy()
    elapsed_time = time.time() - start_time
    
    print("trials: {},Maximum steps: {}, episodes: {}, workers: {}, elapsed_time: {}".format(trials,max_steps_num,episode_num,N_WORKERS,elapsed_time) + "[sec]")
     
        