# -*- coding: utf-8 -*-
# @Time      : 2019/12/23 20:16
# @File      : manually_cure.py

from pyclustering.cluster import cluster_visualizer
import sys
import math
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import numpy as np


class CURE:
    def __init__(self, data, k, distance_func, a=0.5, c=3):
        '''
        @brief 构造函数

        @param data(array):待聚类数据
        @param k(int):聚类簇数
        @param distance_func(obj):计算距离的函数
        @param a(float):收缩因子
        @param c(int):代表点数
        '''
        self.data = data
        self.k = k
        self.distance_func = distance_func
        self.a = a
        self.c = c
        self.clusters = [[i] for i in range(len(self.data))]  # 初始化簇数
        self.means = [x for x in self.data]  # 初始化簇中心点
        self.reps = [[x, ] for x in self.data]  # 初始化簇代表点

    def calcu_distance(self, u, v):
        '''
        @brief 计算两簇之间的距离

        @param u:簇u的索引
        @param v:簇v的索引

        @return 返回两簇之间的距离
        '''
        u_reps = self.reps[u]
        v_reps = self.reps[v]
        min_distance = sys.maxsize
        for i in u_reps:
            for j in v_reps:
                distance = self.distance_func(i, j)
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def update_mean(self, u, v):
        '''
        @brief 簇u和簇v合并得簇w，计算w的mean，并更新簇中心点

        @param u(int):簇u的索引
        @param v(int):簇v的索引
        '''
        # 计算新的mean
        num_u = len(self.clusters[u])
        num_v = len(self.clusters[v])
        mean = (num_u * self.means[u] + num_v * self.means[v]) / (num_u + num_v)
        # 更新means
        self.means[u] = mean

    def update_reps(self, u, v):
        '''
        @brief 簇u和簇v合并得簇w，选择w的代表点

        @param u(int):簇u的索引
        @param v(int):簇v的索引
        '''
        data = self.clusters[u] + self.clusters[v]
        temp = []
        # 中心点少于c
        if len(data) < self.c:
            for x in data:
                rep = self.data[x] + self.a * (self.means[u] - self.data[x])
                temp.append(rep)
        # 中心点大于c
        else:
            # 找距离中心最远的点
            far_dot = data[0]
            far_distance = self.distance_func(self.data[far_dot], self.means[u])
            for x in data:
                distance = self.distance_func(self.data[x], self.means[u])
                if distance > far_distance:
                    far_distance = distance
                    far_dot = x
            rep = self.data[far_dot] + self.a * (self.means[u] - self.data[far_dot])
            temp.append(rep)
            dots = [far_dot]
            data.remove(far_dot)
            # 找k个距离前一个点最远的点
            i = self.c - 1
            while i > 0:
                i = i - 1
                far_dot = data[0]
                last_dot = dots[len(dots) - 1]
                far_distance = self.distance_func(self.data[far_dot], self.data[last_dot])
                for x in data:
                    distance = self.distance_func(self.data[x], self.data[last_dot])
                    if distance > far_distance:
                        far_distance = distance
                        far_dot = x
                rep = self.data[far_dot] + self.a * (self.means[u] - self.data[far_dot])
                temp.append(rep)
                data.remove(far_dot)
        self.reps[u] = temp

    def update(self, u, v):
        '''
        @brief 合并clusters、更新means和reps

        @param u(int):簇u的索引
        @param v(int):簇v的索引
        '''
        # 更新cluster,mean和reps列表
        self.clusters[u] = self.clusters[u] + self.clusters[v]
        self.clusters.pop(v)
        self.means.pop(v)
        self.reps.pop(v)

    def run(self):
        '''
        @brief 组装cure并运行

        @return 返回 最终聚类结果
        '''
        while len(self.clusters) > self.k:
            min_distance = self.calcu_distance(0, 1)
            min_u = 0
            min_v = 1
            for u in range(0, len(self.clusters)):
                for v in range(u + 1, len(self.clusters)):
                    distance = self.calcu_distance(u, v)
                    if min_distance > distance:
                        min_u = u
                        min_v = v
                        min_distance = distance
            self.update_mean(min_u, min_v)
            self.update_reps(min_u, min_v)
            self.update(min_u, min_v)
        return self.clusters


def manhattan_distance(i, j):
    '''
    @brief 计算i,j的曼哈顿距离

    @param i(array):点i坐标
    @param j(array):点j坐标

    @return 返回i,j的曼哈顿距离
    '''
    s = i - j
    result = 0
    for i in range(s.shape[0]):
        result += math.fabs(s[i])
    return result


def euclidea_distance(i, j):
    '''
    @brief 计算i,j的欧氏距离

    @param i(array):点i坐标
    @param j(array):点j坐标

    @return 返回i,j的欧氏距离
    '''
    s = i - j
    result = 0
    for i in range(s.shape[0]):
        result += s[i] ** 2
    return math.sqrt(result)


if __name__ == '__main__':
    # 读取fcps数据集
    sample = read_sample(FCPS_SAMPLES.SAMPLE_TETRA)
    array_sample = np.array(sample)
    # 运行cure
    clusters = CURE(array_sample, k=4, distance_func=euclidea_distance).run()
    print(clusters)
    # 可视化聚类结果
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, sample)
    visualizer.show()
