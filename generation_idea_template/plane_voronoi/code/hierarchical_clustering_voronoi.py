"""
Implement hierarchical clustering in the 'PaneVoronoi' class to group seed points based on spatial proximity
Update the 'attribution' function to first identify the nearest cluster using a simple distance metric, then calculate the closest seed point within that cluster
This approach reduces the total number of distance calculations needed
Compare execution time and accuracy of the Voronoi diagrams generated using this method against the baseline, across multiple grid sizes and seed distributions

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pane_voronoi.py
@Time    :   2023/09/21 17:03:58
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import pprint
import copy
import random
from PIL import Image
from tqdm import tqdm  # 进度条
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np


class PaneVoronoi:
    def __init__(self, seed, seed_list, n):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        # self.hash_map = self.creat_hash()
        self.hash_map = [i * i for i in range(self.n)]
        # print('哈希表计算完成')
        self.seed_list = seed_list  # 生成种子点
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # 随机化颜色，并且种子点设置为黑色
        self.count = n * 4 - 4
        self.clusters = self.create_clusters()

    def creat_hash(self):
        dic = defaultdict(int)
        for i in range(self.n):
            dic[i] = i * i
        return dic

    def creat_seed(self):
        res = []
        for _ in range(self.seed):
            res.append([random.randrange(self.n), random.randrange(self.n)])
        return res

    def colors(self):
        res = [[0, 0, 0]]
        for _ in range(self.n):
            res.append([random.randrange(99, 206) for _ in range(3)])
        return res

    def create_clusters(self):
        # Using hierarchical clustering to group seed points
        Z = linkage(self.seed_list, method='ward')
        max_d = 10  # Maximum distance for clusters, can be adjusted
        cluster_labels = fcluster(Z, max_d, criterion='distance')
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(self.seed_list[idx])
        return clusters

    def deal(self):
        # top deal
        for j in range(self.n):
            self.table[0][j] = self.attribution([0, j])
            self.visited[0][j] = True
        # bottom deal
        for j in range(self.n):
            self.table[self.n - 1][j] = self.attribution([self.n - 1, j])
            self.visited[self.n - 1][j] = True
        # left deal
        for i in range(self.n):
            self.table[i][0] = self.attribution([i, 0])
            self.visited[i][0] = True
        # right deal
        for i in range(self.n):
            self.table[i][self.n - 1] = self.attribution([i, self.n - 1])
            self.visited[i][self.n - 1] = True

    def positive_search(self):
        copy_table = copy.deepcopy(self.table)
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                if [i, j] not in self.seed_list:
                    if copy_table[i][j - 1] == copy_table[i - 1][j - 1] == copy_table[i - 1][j] == copy_table[i - 1][
                        j + 1]:
                        copy_table[i][j] = copy_table[i][j - 1]
                    else:
                        self.visited[i][j] = True
                        self.count += 1
                        copy_table[i][j] = self.attribution([i, j])
        return copy_table

    def reverse_search(self):
        copy_table = copy.deepcopy(self.table)
        for i in range(self.n - 2, 0, -1):
            for j in range(self.n - 2, 0, -1):
                if [i, j] not in self.seed_list:
                    if copy_table[i][j + 1] == copy_table[i + 1][j + 1] == copy_table[i + 1][j] == copy_table[i + 1][
                        j - 1]:
                        copy_table[i][j] = copy_table[i][j + 1]
                    else:
                        self.visited[i][j] = True
                        copy_table[i][j] = self.attribution([i, j])
        return copy_table

    def attribution_algorithm(self):
        copy_table = copy.deepcopy(self.table)
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                if [i, j] not in self.seed_list:
                    copy_table[i][j] = self.attribution([i, j])
        return copy_table

    def positive_reverse(self):
        # 正向扫描
        copy_table = self.positive_search()
        # 逆向纠错
        for i in range(self.n - 2, 0, -1):
            for j in range(self.n - 2, 0, -1):
                if [i, j] not in self.seed_list:
                    if copy_table[i][j - 1] == copy_table[i - 1][j - 1] == copy_table[i - 1][j] == copy_table[i - 1][
                        j + 1] == copy_table[i][j + 1] == copy_table[i + 1][j + 1] == copy_table[i + 1][j] == \
                            copy_table[i + 1][j - 1]:
                        pass
                    else:
                        if not self.visited[i][j]:
                            copy_table[i][j] = self.attribution([i, j])
                            self.visited[i][j] = True
                            self.count += 1
        return copy_table

    def attribution(self, point):
        # Find nearest cluster
        nearest_cluster = None
        min_dist_cluster = float('inf')
        for label, seeds in self.clusters.items():
            cluster_center = np.mean(seeds, axis=0)
            dist = (point[0] - cluster_center[0]) ** 2 + (point[1] - cluster_center[1]) ** 2
            if dist < min_dist_cluster:
                min_dist_cluster = dist
                nearest_cluster = seeds

        # Now find the nearest seed within the nearest cluster
        dic = float('inf')
        res = []
        for i in range(len(nearest_cluster)):
            tmp = (point[0] - nearest_cluster[i][0]) ** 2 + (point[1] - nearest_cluster[i][1]) ** 2
            if tmp < dic:
                res.append(self.seed_list.index(nearest_cluster[i]))
                dic = tmp
        return res[-1] + 1

    @classmethod
    def check(cls, data1, data2):
        total = len(data1) * len(data1[0])
        count = 0
        for i in tqdm(range(len(data1))):
            for j in range(len(data1[0])):
                if data1[i][j] == data2[i][j]:
                    count += 1
        print('计算完成')
        return (count / total) * 100

    @classmethod
    def paint(cls, data, name, colors):
        image = Image.new('RGB', (len(data), len(data[0])))
        put_pixel = image.putpixel
        for i in tqdm(range(len(data))):
            for j in range(len(data[0])):
                color = colors[data[i][j]]
                put_pixel((i, j), (color[0], color[1], color[2]))
        image.save(f'img/{name}.jpg')

    @classmethod
    def paint_visited(cls, data, name):
        image = Image.new('RGB', (len(data), len(data[0])))
        put_pixel = image.putpixel
        for i in tqdm(range(len(data))):
            for j in range(len(data[0])):
                if data[i][j]:
                    put_pixel((i, j), (255, 0, 0))
                else:
                    put_pixel((i, j), (255, 255, 255))
        image.save(f'img/{name}.jpg')


if __name__ == '__main__':
    seed_list = [[random.randrange(32), random.randrange(32)] for _ in range(500)]
    v = PaneVoronoi(32, seed_list, 32)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)