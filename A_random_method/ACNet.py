# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:54:08 2022

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

class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE_1 or scope == GLOBAL_NET_SCOPE_2:   # get global network
            with tf.compat.v1.variable_scope(scope):# 'Global_Net'
                #print("-----------------Global Network is created!-------------------------------")
                self.s = tf.compat.v1.placeholder(tf.float32, [None, N_S], 'S') #　observationのplaceholderを生成する
                #print("状態s is {}".format(self.s.shape))#(None, 4)
                self.a_params, self.c_params = self._build_net(scope)[-2:]# ActorとCriticのパラメータをゲットする
                #print("a_params is {}".format(self.a_params))#(None, 4)
                #print("c_params is {}".format(self.c_params))#(None, 4)
        else:   # local network, calculate losses
            with tf.compat.v1.variable_scope(scope):# eg.Worker_1
                
                #print("------------------Local "+str(scope)+" is created!-----------------------------")
                self.s = tf.compat.v1.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.compat.v1.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.compat.v1.subtract(self.v_target, self.v, name='TD_error')# TD誤差、目標値と予測値の差
                with tf.compat.v1.name_scope('c_loss'):# Criticの損失関数の定義
                    self.c_loss = tf.compat.v1.reduce_mean(tf.square(td)) # Criticの損失関数はTD誤差の二乗として定義する
                    #print("Criticの損失関数c_loss is {}".format(self.c_loss.shape))

                with tf.compat.v1.name_scope('a_loss'):# Actorの損失関数の定義
                    # Q * log π（a|s)
                    log_prob = tf.compat.v1.reduce_sum(tf.compat.v1.log(self.a_prob + 1e-5) *
                                             tf.one_hot(self.a_his, N_A, dtype=tf.float32),# eg. a_his = [0,1,3] , N_A = 4 -> [[1,0,0,0],[0,1,0,0],[0,0,0,1]]
                                             axis=1, keepdims=True)
                    
                    #print("log_prob is {}".format(log_prob.shape)) #(None, 1)
                    exp_v = log_prob * tf.compat.v1.stop_gradient(td) 
                    #print("exp_v is {}".format(exp_v.shape))# (None, 1)
                    entropy = -tf.compat.v1.reduce_sum(self.a_prob * tf.compat.v1.log(self.a_prob + 1e-5),
                                             axis=1, keepdims=True)  #　エントロピーの計算
                    #print("entropy is {}".format(entropy.shape))#(None, 1)
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    #print("self.exp_v is {}".format(self.exp_v.shape))#(None, 1)
                    self.a_loss = tf.compat.v1.reduce_mean(-self.exp_v)
                    #print("Actorの損失関数a_loss is {}".format(self.a_loss.shape))

                with tf.compat.v1.name_scope('local_grad'):#勾配を計算
                    self.a_grads = tf.compat.v1.gradients(self.a_loss, self.a_params) # Actorの勾配を計算する
                    
                    self.c_grads = tf.compat.v1.gradients(self.c_loss, self.c_params) # Criticの勾配を計算する
                    

            with tf.compat.v1.name_scope('sync'):
                with tf.compat.v1.name_scope('pull'): # パラメータサーバーから重みなどをコーピーする
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]# パラメータサーバーのActorから重みなどをコーピーする
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]# パラメータサーバーのCriticから重みなどをコーピーする
                with tf.compat.v1.name_scope('push'): # 計算した勾配を利用しパラメータサーバーの重みを更新する計算を定義する
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        # ネットワークを構成する
        w_init = tf.random_normal_initializer(0., .1)# 重みを初期化する
        with tf.compat.v1.variable_scope('actor'):# Actorのネットワークを構成する
            l_a = tf.compat.v1.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.compat.v1.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap') # 行動の選択確率のベクトルを出力する
            #print("行動選択確率ベクトルa_prob is {}".format(a_prob.shape))#(None, 4)
        with tf.compat.v1.variable_scope('critic'):# Criticのネットワークを構成する
            l_c = tf.compat.v1.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.compat.v1.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # 状態価値を出力する
            #print("状態価値関数c is {}".format(v.shape))#(None, 1)
        a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')# Actorの重みパラメータをゲットする
        c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')# Criticの重みパラメータをゲットする
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):# パラメータサーバーの重みを更新するメソッド  
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  

    def pull_global(self):# パラメータサーバーの重みをコピーするメソッド    
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        #print("prob_weights is {}".format(prob_weights.shape))#(1, 4)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())# [0,1,2,3] から　確率p [0.1,0.2,0.5,0.2] によって一個を選ぶ．p は選択確率が異なる場合  
        return action

