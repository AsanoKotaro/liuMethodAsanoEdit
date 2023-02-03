# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:29:49 2022

@author: asano
"""

import tensorflow as tf

# There can be an arbitrary number of
# axes (sometimes called "dimensions")
tf.config.run_functions_eagerly(True)
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)

with tf.compat.v1.Session() as sess:   
    print(rank_3_tensor[0][0].eval())
    
    # Turn this back off
    
