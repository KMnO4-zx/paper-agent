"""
Implement a method to assess grid cell complexity based on the variation in distances to seed points
Modify the 'PaneVoronoi' class to include a recursive function that refines grid cells by subdividing them when the variance of the distance measurements exceeds a certain threshold
Test the algorithm on different seed distributions and evaluate performance gains by comparing execution time and output quality against the baseline fixed-grid implementation

"""

# Modified code
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
import numpy as np

class PaneVoronoi:
    def __init__(self, seed, seed_list, n, threshold=0.1):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.threshold = threshold  # Threshold for grid cell complexity
        self.seed_list = seed_list  # 生成种子点
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # 随机化颜色，并且种子点设置为黑色
        self.count = n * 4 - 4

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

    def recursive_subdivide(self, x_start, y_start, size):
        if size <= 1:
            return
        
        distances = []
        for i in range(x_start, x_start + size):
            for j in range(y_start, y_start + size):
                dist = self.compute_distances_to_seeds([i, j])
                distances.append(dist)
        
        variance = np.var(distances)
        
        if variance > self.threshold:
            half_size = size // 2
            self.recursive_subdivide(x_start, y_start, half_size)
            self.recursive_subdivide(x_start + half_size, y_start, half_size)
            self.recursive_subdivide(x_start, y_start + half_size, half_size)
            self.recursive_subdivide(x_start + half_size, y_start + half_size, half_size)
        else:
            for i in range(x_start, x_start + size):
                for j in range(y_start, y_start + size):
                    self.table[i][j] = self.attribution(self.seed_list, [i, j])
                    self.visited[i][j] = True

    def compute_distances_to_seeds(self, point):
        distances = []
        for seed in self.seed_list:
            dist = (point[0] - seed[0]) ** 2 + (point[1] - seed[1]) ** 2
            distances.append(dist)
        return distances

    def deal(self):
        self.recursive_subdivide(0, 0, self.n)

    def attribution(self, seed, point):
        dic = float('inf')
        res = 0
        for i in range(len(seed)):
            tmp = (point[0] - seed[i][0]) ** 2 + (point[1] - seed[i][1]) ** 2
            if tmp < dic:
                res = i
                dic = tmp
        return res + 1

    @classmethod
    def paint(cls, data, name, colors):
        image = Image.new('RGB', (len(data), len(data[0])))
        put_pixel = image.putpixel
        for i in tqdm(range(len(data))):
            for j in range(len(data[0])):
                color = colors[data[i][j]]
                put_pixel((i, j), (color[0], color[1], color[2]))
        image.save(f'img/{name}.jpg')


if __name__ == '__main__':
    seed_points = [[random.randrange(32), random.randrange(32)] for _ in range(500)]
    v = PaneVoronoi(32, seed_points, 100)
    v.deal()
    v.paint(v.table, 'voronoi_adaptive', v.colors)