"""
Use a priority queue to dynamically expand the influence of seed points across the grid, analogous to breadth-first search without explicitly constructing a graph
Modify the 'attribution' function to utilize this queue-based propagation, ensuring efficient handling of neighboring grid cells
Evaluate the execution time and scalability improvements compared to the baseline approach

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
from collections import defaultdict, deque

class PaneVoronoi:
    def __init__(self, seed, seed_list, n):
        self.n = n  # 边长 默认都是正方形
        self.seed = seed  # 种子点
        self.hash_map = [i * i for i in range(self.n)]
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

    def deal(self):
        # Initialize the queue with seed points
        queue = deque()
        for idx, (x, y) in enumerate(self.seed_list):
            queue.append((x, y, idx + 1))  # (x, y, seed_index)
            self.table[x][y] = idx + 1
            self.visited[x][y] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y, seed_index = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.n and not self.visited[nx][ny]:
                    self.table[nx][ny] = seed_index
                    self.visited[nx][ny] = True
                    queue.append((nx, ny, seed_index))

    def positive_reverse(self):
        return self.table

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
    seed_list = [[random.randrange(32), random.randrange(32)] for _ in range(500)]
    v = PaneVoronoi(500, seed_list, 32)
    v.deal()
    da = v.positive_reverse()
    v.paint(da, 'voronoi', v.colors)