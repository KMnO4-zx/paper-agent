"""
Introduce a preprocessing step in the 'PaneVoronoi' class that analyzes the seed distribution to detect geometric invariants like periodicity or symmetry
If such invariants are present, modify the 'attribution' function to compute Voronoi cells for one representative region and replicate the results for similar regions
Include a fallback mechanism that defaults to standard processing when no patterns are detected
Test this approach on grids with varying levels of geometric invariance and evaluate the computational efficiency gains compared to the baseline method

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


class PaneVoronoi:
    def __init__(self, seed, seed_list, n):
        self.n = n  # Grid size (assumed square)
        self.seed = seed  # Number of seed points
        self.hash_map = [i * i for i in range(self.n)]
        self.seed_list = seed_list  # List of seed points
        self.table = [[0] * self.n for _ in range(self.n)]
        self.visited = [[False] * self.n for _ in range(self.n)]
        self.colors = self.colors()  # Randomized colors, with seed points set to black
        self.count = n * 4 - 4
        self.pattern_detected = False
        self.representative_cells = {}

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

    def detect_invariants(self):
        # Heuristic to detect symmetry or periodicity in seed distribution
        pattern_count = defaultdict(int)
        for x, y in self.seed_list:
            pattern = (x % 2, y % 2)
            pattern_count[pattern] += 1
        
        # If the majority of seeds have similar patterns, consider it a detected invariant
        if any(count > len(self.seed_list) // 2 for count in pattern_count.values()):
            self.pattern_detected = True
            self.representative_cells = {pattern: None for pattern in pattern_count.keys()}

    def deal(self):
        self.detect_invariants()
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
            self.visited[self.n - 1][j] = True

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
        if self.pattern_detected:
            pattern = (point[0] % 2, point[1] % 2)
            if self.representative_cells[pattern] is not None:
                return self.representative_cells[pattern]
            
            dic = float('inf')
            res = []
            for i in range(len(seed)):
                tmp = (point[0] - seed[i][0]) ** 2 + (point[1] - seed[i][1]) ** 2
                if tmp < dic:
                    res.append(i)
                    dic = tmp
            self.representative_cells[pattern] = res[-1] + 1
            return self.representative_cells[pattern]
        
        else:
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
    seed_list = [[random.randrange(32), random.randrange(32)] for _ in range(500)]
    v = PaneVoronoi(32, seed_list, 32)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)