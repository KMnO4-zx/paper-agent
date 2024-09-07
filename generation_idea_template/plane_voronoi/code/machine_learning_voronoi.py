"""
Develop a machine learning model to predict the nearest seed point for grid cells
Generate training data using the existing Voronoi diagram algorithm on various grid sizes and seed distributions
Train a lightweight model, such as a neural network or decision tree, to learn the mapping between grid positions and their closest seeds
Integrate this predictive model into the 'PaneVoronoi' class, modifying the 'attribution' function to utilize predictions
Evaluate performance by comparing execution times and accuracy with the baseline approach

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
import numpy as np
from PIL import Image
from tqdm import tqdm  # 进度条
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

class PaneVoronoi:
    def __init__(self, seed, n):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.seed_list = self.creat_seed()  # 生成种子点
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # 随机化颜色，并且种子点设置为黑色
        self.count = n * 4 - 4
        self.model = None  # Placeholder for ML model

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

    def generate_training_data(self):
        data = []
        labels = []
        for i in range(self.n):
            for j in range(self.n):
                closest_seed = self.attribution(self.seed_list, [i, j])
                data.append([i, j])
                labels.append(closest_seed)
        return np.array(data), np.array(labels)

    def train_model(self):
        data, labels = self.generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model = DecisionTreeClassifier()
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy}")

    def deal(self):
        # Train the model before processing
        self.train_model()
        
        # top deal
        for j in range(self.n):
            self.table[0][j] = self.predict_attribution([0, j])
            self.visited[0][j] = True
        # bottom deal
        for j in range(self.n):
            self.table[self.n - 1][j] = self.predict_attribution([self.n - 1, j])
            self.visited[self.n - 1][j] = True
        # left deal
        for i in range(self.n):
            self.table[i][0] = self.predict_attribution([i, 0])
            self.visited[i][0] = True
        # right deal
        for i in range(self.n):
            self.table[i][self.n - 1] = self.predict_attribution([i, self.n - 1])
            self.visited[i][self.n - 1] = True

    def predict_attribution(self, point):
        if self.model:
            return self.model.predict([point])[0]
        else:
            return self.attribution(self.seed_list, point)

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
                        copy_table[i][j] = self.predict_attribution([i, j])
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
                        copy_table[i][j] = self.predict_attribution([i, j])
        return copy_table

    def attribution_algorithm(self):
        copy_table = copy.deepcopy(self.table)
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                if [i, j] not in self.seed_list:
                    copy_table[i][j] = self.predict_attribution([i, j])
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
                            copy_table[i][j] = self.predict_attribution([i, j])
                            self.visited[i][j] = True
                            self.count += 1
        return copy_table

    def attribution(self, seed, point):
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
    v = PaneVoronoi(500, 32)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)