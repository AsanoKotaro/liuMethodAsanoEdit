# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:03:53 2022

@author: asano
"""

class Worker(object):
    def __init__(self, name, algorithm,learning_data):
        
        self.name = name
                
        self.AC = algorithm
            
            
        self.learning_data = learning_data
        self.data_num  = len(self.learning_data)
            
        #print(self.data_num)
            
            
    def update_A3C(self):   
        #global GLOBAL_RUNNING_R, GLOBAL_EP,T
        
                
        #print("\n"+"=========================="+str(self.name)+"のパラメータサーバーの重みを更新中===============================")
        #share_percent = 1
        #self.data_shuffer(share_percent)
        
        #if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # 10ステップ一回パラメータサーバーの重みを更新する
        for id in range(self.data_num): # 全てのデータを使って更新
            
            episode_done = self.learning_data[id][4]# エージェント1のデータのepisode done
            last_obs = self.learning_data[id][2]# エージェント1のデータのlast_obs_r1
            episode_reward = self.learning_data[id][3]# エージェント1のデータのbuffer_r
            episodes_buffer_s = self.learning_data[id][0]# エージェント1のデータのbuffer_s1
            episodes_buffer_a = self.learning_data[id][1]# エージェント1のデータのbuffer_a1
            
            
        
            
            if episode_done: 
                v_s_ = 0   # 終端状態の状態価値は0とする
            else: 
                s_ = last_obs# エージェント1のデータのs1_
                #print("s1_ is {}".format(s1_))
                v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]#?
            #print("22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222")        
            buffer_v_target = [] # robot1 
           
            for r in episode_reward[::-1]:    # reverse buffer r
                v_s_ = r + GAMMA * v_s_ # v(s) = r + GAMMA *　v(s+1)によってtarget_vを計算する
                buffer_v_target.append(v_s_)
 
            buffer_v_target.reverse()

            #print("buffer_v_target is {}".format(buffer_v_target))
            


            episodes_buffer_s, episodes_buffer_a, buffer_v_target = np.vstack(episodes_buffer_s), np.array(episodes_buffer_a), np.vstack(buffer_v_target)
            
            #print("buffer_s1 is {}".format(buffer_s1))
            #print("buffer_a1 is {}".format(buffer_a1))
            #print("buffer_v_target is {}".format(buffer_v_target))
            feed_dict = {
                self.AC.s: episodes_buffer_s,
                self.AC.a_his: episodes_buffer_a,
                self.AC.v_target: buffer_v_target,
                }
                    
            self.AC.update_global(feed_dict)
            self.AC.pull_global()

        #print("=========================="+str(self.name)+"のパラメータサーバーの重み更新完了==============================="+"\n")
               