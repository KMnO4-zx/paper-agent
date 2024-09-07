"""
Develop a regression model using scikit-learn to predict Voronoi cell boundaries based on grid and seed characteristics
Train the model on data from previous computations of Voronoi diagrams with varying grid sizes and seed distributions
Integrate this model into 'PaneVoronoi', modifying the 'attribution' function to use these predictions for initial boundary estimations
Evaluate by comparing execution times and accuracy against the baseline method, focusing on scenarios with similar seed distributions to the training set

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
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle


class PaneVoronoi:
    def __init__(self, seed, seed_list, n, model=None):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.hash_map = [i * i for i in range(self.n)]
        self.seed_list = seed_list  # 生成种子点
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # 随机化颜色，并且种子点设置为黑色
        self.count = n * 4 - 4
        self.model = model

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

    def deal(self):
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
        copy_table = self.positive_search()
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
        if self.model is not None:
            # Use the regression model to predict the index of the closest seed
            features = np.array(point + [self.n] + np.array(seed).flatten().tolist()).reshape(1, -1)
            pred = self.model.predict(features)[0]
            return int(pred) + 1
        else:
            # Original method
            dic = float('inf')
            res = []
            for i in range(len(seed)):
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
    # Load a pre-trained regression model if available
    try:
        with open('voronoi_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        model = None

    # Example seed list and grid size
    seed_list = [[random.randrange(32), random.randrange(32)] for _ in range(500)]
    
    v = PaneVoronoi(32, seed_list, 32, model=model)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)