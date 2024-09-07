"""
Implement a boundary-based approach in the 'PaneVoronoi' class by introducing a graph-based structure to identify neighboring seed points
Add a function to compute perpendicular bisectors between these neighbors
Modify the 'attribution' function to assign points based on proximity to these boundaries
Evaluate performance improvements by comparing execution times and accuracy against the current exhaustive approach, with a focus on large grid sizes

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
from itertools import combinations
import math


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
        self.neighbor_graph = self.construct_neighbor_graph()

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

    def construct_neighbor_graph(self):
        # Create a graph structure to find neighbors
        graph = defaultdict(list)
        for (i, seed1), (j, seed2) in combinations(enumerate(self.seed_list), 2):
            graph[i].append(j)
            graph[j].append(i)
        return graph

    def compute_bisector(self, p1, p2):
        # Compute perpendicular bisector between two points p1 and p2
        mid_point = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        if p1[0] == p2[0]:  # vertical line
            slope = float('inf')
        else:
            slope = -(p2[0] - p1[0]) / (p2[1] - p1[1])
        return mid_point, slope

    def compute_boundaries(self):
        # Compute and store boundaries for each seed using bisectors
        self.boundaries = {}
        for i, neighbors in self.neighbor_graph.items():
            self.boundaries[i] = []
            for j in neighbors:
                mid_point, slope = self.compute_bisector(self.seed_list[i], self.seed_list[j])
                self.boundaries[i].append((mid_point, slope))

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
        # Utilize boundaries to improve attribution
        distances = {}
        for i, (mid_point, slope) in enumerate(self.boundaries):
            # Calculate perpendicular distance from point to bisector line
            if slope == float('inf'):
                dist = abs(point[0] - mid_point[0])
            else:
                x0, y0 = mid_point
                x, y = point
                dist = abs(slope * x - y + y0 - slope * x0) / math.sqrt(slope ** 2 + 1)
            distances[i] = dist

        # Choose the seed with the smallest perpendicular distance
        closest_seed = min(distances, key=distances.get)
        return closest_seed + 1

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
    v = PaneVoronoi(32, 500)
    v.compute_boundaries()  # Added call to compute boundaries
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)