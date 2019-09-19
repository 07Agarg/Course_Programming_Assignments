# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 05:25:25 2019

@author: ashima
"""

import numpy as np
import copy

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

def isSafe(i, j, M, N):
    return i >= 0 and i < M and j >= 0 and j < N

def find_successors(state, M, N):
    empty_coord = np.where(state == 'E')
    successors = []
    for i in range(4):
        new_x = empty_coord[0][0] + dx[i]
        new_y = empty_coord[1][0] + dy[i]
        if isSafe(new_x, new_y, M, N):
            new_state = copy.copy(state)
            val = new_state[new_x][new_y]
            new_state[new_x][new_y] = 'E'
            new_state[empty_coord[0][0]][empty_coord[1][0]] = val
            successors.append(new_state)
    return successors            

def isCompleted(state, goal, M, N):
    for i in range(M):
        for j in range(N):
            if state[i][j] != goal[i][j]:
                return False
    return True