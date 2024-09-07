"""
Use Python's multiprocessing library to parallelize the 'attribution' function
Divide the table into chunks that can be processed independently in separate processes
Each process will handle the computation of distances for its assigned chunk
Once all processes complete, the results will be merged to form the final Voronoi diagram
Evaluate the performance by comparing execution times on varying grid sizes with and without parallel processing
Additionally, ensure thread safety and manage shared resources carefully

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
from multiprocessing import Pool, cpu_count
import numpy as np

class PaneVoronoi:
    def __init__(self, seed, seed_list, n):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.hash_map = [i * i for i in range(self.n)]
        self.seed_list = seed_list  # 生成种子点
        self.table = np.zeros((self.n, self.n), dtype=int)
        self.visited = np.zeros((self.n, self.n), dtype=bool)
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

    def deal(self):
        # Process edges of the grid
        for j in range(self.n):
            self.table[0][j] = self.attribution(self.seed_list, [0, j])
            self.visited[0][j] = True
            self.table[self.n - 1][j] = self.attribution(self.seed_list, [self.n - 1, j])
            self.visited[self.n - 1][j] = True
        for i in range(self.n):
            self.table[i][0] = self.attribution(self.seed_list, [i, 0])
            self.visited[i][0] = True
            self.table[i][self.n - 1] = self.attribution(self.seed_list, [i, self.n - 1])
            self.visited[i][self.n - 1] = True

    def positive_search(self):
        copy_table = np.copy(self.table)
        with Pool(processes=cpu_count()) as pool:
            chunks = np.array_split(range(1, self.n - 1), cpu_count())
            results = pool.starmap(self.process_chunk_positive, [(chunk, copy_table) for chunk in chunks])
            for result in results:
                for row_idx, row in result:
                    copy_table[row_idx] = row
        return copy_table

    def process_chunk_positive(self, rows, copy_table):
        local_copy = np.copy(copy_table)
        result = []
        for i in rows:
            for j in range(1, self.n - 1):
                if [i, j] not in self.seed_list:
                    if local_copy[i][j - 1] == local_copy[i - 1][j - 1] == local_copy[i - 1][j] == local_copy[i - 1][j + 1]:
                        local_copy[i][j] = local_copy[i][j - 1]
                    else:
                        self.visited[i][j] = True
                        self.count += 1
                        local_copy[i][j] = self.attribution(self.seed_list, [i, j])
            result.append((i, local_copy[i]))
        return result

    def reverse_search(self):
        copy_table = np.copy(self.table)
        with Pool(processes=cpu_count()) as pool:
            chunks = np.array_split(range(self.n - 2, 0, -1), cpu_count())
            results = pool.starmap(self.process_chunk_reverse, [(chunk, copy_table) for chunk in chunks])
            for result in results:
                for row_idx, row in result:
                    copy_table[row_idx] = row
        return copy_table

    def process_chunk_reverse(self, rows, copy_table):
        local_copy = np.copy(copy_table)
        result = []
        for i in rows:
            for j in range(self.n - 2, 0, -1):
                if [i, j] not in self.seed_list:
                    if local_copy[i][j + 1] == local_copy[i + 1][j + 1] == local_copy[i + 1][j] == local_copy[i + 1][j - 1]:
                        local_copy[i][j] = local_copy[i][j + 1]
                    else:
                        self.visited[i][j] = True
                        local_copy[i][j] = self.attribution(self.seed_list, [i, j])
            result.append((i, local_copy[i]))
        return result

    def positive_reverse(self):
        copy_table = self.positive_search()
        with Pool(processes=cpu_count()) as pool:
            chunks = np.array_split(range(self.n - 2, 0, -1), cpu_count())
            results = pool.starmap(self.process_chunk_positive_reverse, [(chunk, copy_table) for chunk in chunks])
            for result in results:
                for row_idx, row in result:
                    copy_table[row_idx] = row
        return copy_table

    def process_chunk_positive_reverse(self, rows, copy_table):
        local_copy = np.copy(copy_table)
        result = []
        for i in rows:
            for j in range(self.n - 2, 0, -1):
                if [i, j] not in self.seed_list:
                    if not (
                        local_copy[i][j - 1] == local_copy[i - 1][j - 1] == local_copy[i - 1][j] == local_copy[i - 1][j + 1] ==
                        local_copy[i][j + 1] == local_copy[i + 1][j + 1] == local_copy[i + 1][j] == local_copy[i + 1][j - 1]
                    ):
                        if not self.visited[i][j]:
                            local_copy[i][j] = self.attribution(self.seed_list, [i, j])
                            self.visited[i][j] = True
                            self.count += 1
            result.append((i, local_copy[i]))
        return result

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
    seed = 500
    n = 32
    seed_list = [[random.randrange(n), random.randrange(n)] for _ in range(seed)]
    v = PaneVoronoi(seed, seed_list, n)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)