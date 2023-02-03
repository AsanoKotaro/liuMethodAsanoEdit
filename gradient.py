# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:50:37 2022

@author: asano
"""

from __future__ import annotations

import tensorflow as tf  # type: ignore

# tf.compat.v1.enable_eager_execution()

def objective(params: tf.Variable) -> tf.Tensor:
    """最小化したい目的関数"""
    # x_0^2 + x_1^2
    loss = params[0] ** 2 + params[1] ** 2
    return loss


@tf.function
def training_step(params: tf.Variable, optimizer: tf.keras.optimizers.Optimizer) -> None:
    # 自動微分する
    with tf.GradientTape(watch_accessed_variables=False) as t:
        # 勾配を計算するために追跡が必要なパラメータを登録する
        t.watch(params)  # watch_accessed_variables=True のときは不要
        # 計算グラフを構築する
        loss = objective(params)
    # 勾配を計算する
    grads = t.gradient(loss, params)
    # 勾配を使ってパラメータを更新する
    optimizer.apply_gradients([(grads, params)])
    # 現在のパラメータを出力する
    print("grads")
    # tf.print(grads)
    print("params")
    tf.print(params)
    # print(grads)
    # print(params)
    # return grads, params


def main():
    # 定数で初期化した変数を用意する

    
    tensor = tf.constant([1., 4.], dtype=tf.float32)
    params = tf.Variable(tensor, trainable=True)

    # SGD で最適化する
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)

    # イテレーションを回す
    for _ in range(20):
        training_step(params, optimizer)

    # 結果を出力する
    print("objective(params)")
    print(objective(params))


if __name__ == '__main__':
    main()