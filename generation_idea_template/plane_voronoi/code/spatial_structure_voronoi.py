"""
Integrate a spatial data structure, such as a k-d tree or quadtree, to manage seed points
Modify the 'attribution' function to utilize this structure for efficient nearest neighbor queries
Implement the data structure from scratch or use an available library like scikit-learn's KDTree
Benchmark the Voronoi diagram generation on various grid sizes to compare execution times before and after the integration
Measure improvements in time complexity and assess scalability

"""

# Modified code
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pane_voronoi.py
@Time    :   2023/09/21 17:03:58
@Author  :   不要葱姜蒜
@Version :   1.1
@Desc    :   Integrated KDTree for efficient nearest neighbor search
'''

import pprint
import copy
import random
from PIL import Image
from tqdm import tqdm  # 进度条
from collections import defaultdict
from sklearn.neighbors import KDTree  # Importing KDTree from sklearn

class PaneVoronoi:
    def __init__(self, n, seed):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.seed_list = self.creat_seed()  # 生成种子点
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # 随机化颜色，并且种子点设置为黑色
        self.count = n * 4 - 4

        # Construct the KDTree with seed points for efficient nearest neighbor search
        self.kd_tree = KDTree(self.seed_list)

    def creat_seed(self):
        res = []
        for _ in range(self.seed):
            res.append([random.randrange(self.n), random.randrange(self.n)])
        return res

    def colors(self):
        res = [[0, 0, 0]]
        for _ in range(self.seed):
            res.append([random.randrange(99, 206) for _ in range(3)])
        return res

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
        # Use KDTree to find the nearest seed point
        dist, ind = self.kd_tree.query([point], k=1)
        return ind[0][0] + 1

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
    v = PaneVoronoi(32, 1000)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)