"""
Introduce a preprocessing step to classify the grid's symmetry based on seed distribution using simple heuristics or pattern recognition
If symmetry is detected, modify the 'attribution' function to compute Voronoi cells for one part of the grid and apply geometric transformations (mirroring, rotation) for the rest
Test this on grids with varying levels of symmetry to compare execution times and accuracy to the baseline method, evaluating computational efficiency and the scenarios where this method works best

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

    def detect_symmetry(self):
        # Simple heuristic to detect symmetry: check if seed points are symmetric along the middle lines
        x_coords = [s[0] for s in self.seed_list]
        y_coords = [s[1] for s in self.seed_list]
        x_median = self.n // 2
        y_median = self.n // 2
        symmetric_x = all(((x_median - (x - x_median)) in x_coords) for x in x_coords)
        symmetric_y = all(((y_median - (y - y_median)) in y_coords) for y in y_coords)
        return symmetric_x, symmetric_y

    def deal_with_symmetry(self, symmetric_x, symmetric_y):
        if symmetric_x:
            self.deal_half('x')
        if symmetric_y:
            self.deal_half('y')

    def deal_half(self, axis):
        half_n = self.n // 2
        if axis == 'x':
            for i in range(half_n):
                for j in range(self.n):
                    self.table[i][j] = self.attribution(self.seed_list, [i, j])
                    self.visited[i][j] = True
            for i in range(half_n, self.n):
                for j in range(self.n):
                    self.table[i][j] = self.table[self.n - i - 1][j]
                    self.visited[i][j] = True
        elif axis == 'y':
            for i in range(self.n):
                for j in range(half_n):
                    self.table[i][j] = self.attribution(self.seed_list, [i, j])
                    self.visited[i][j] = True
            for i in range(self.n):
                for j in range(half_n, self.n):
                    self.table[i][j] = self.table[i][self.n - j - 1]
                    self.visited[i][j] = True

    def deal(self):
        symmetric_x, symmetric_y = self.detect_symmetry()
        if symmetric_x or symmetric_y:
            self.deal_with_symmetry(symmetric_x, symmetric_y)
        else:
            # top deal
            for j in range(self.n):
                self.table[0][j] = self.attribution(self.seed_list, [0, j])
                self.visited[0][j] = True
            # bottom deal
            for j in range(self.n):
                self.table[self.n - 1][j] = self.attribution(self.seed_list, [self.n - 1, j])
                self.visited[self.n - 1][j] = True
            # left deal
            for i in range(self.n):
                self.table[i][0] = self.attribution(self.seed_list, [i, 0])
                self.visited[i][0] = True
            # right deal
            for i in range(self.n):
                self.table[i][self.n - 1] = self.attribution(self.seed_list, [i, self.n - 1])
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
                        copy_table[i][j] = self.attribution(self.seed_list, [i, j])
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
                        copy_table[i][j] = self.attribution(self.seed_list, [i, j])
        return copy_table

    def attribution_algorithm(self):
        copy_table = copy.deepcopy(self.table)
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                if [i, j] not in self.seed_list:
                    copy_table[i][j] = self.attribution(self.seed_list, [i, j])
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
                            copy_table[i][j] = self.attribution(self.seed_list, [i, j])
                            self.visited[i][j] = True
                            self.count += 1
        return copy_table

    def attribution(self, seed, point):
        dic = float('inf')
        res = []
        for i in range(len(seed)):
            # tmp = self.hash_map[abs(point[0] - seed[i][0])] + self.hash_map[abs(point[1] - seed[i][1])]
            tmp = (point[0] - seed[i][0]) ** 2 + (point[1] - seed[i][1]) ** 2
            if tmp < dic:
                res.append(i)
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
    seed_list = PaneVoronoi(32, [], 500).creat_seed()
    v = PaneVoronoi(32, seed_list, 500)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)