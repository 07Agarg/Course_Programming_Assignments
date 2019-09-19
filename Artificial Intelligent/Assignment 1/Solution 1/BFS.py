# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 03:46:39 2019

@author: ashima
"""

import numpy as np
from utils import find_successors, isCompleted

def bfs(start, goal, M, N):
    queue = []
    exploredSet = []
    queue.append(start)
    while queue:
        state = queue.pop(0)
        if isCompleted(state, goal, M, N):
            return "Solved"
        exploredSet.append(state)
        successors = find_successors(state, M, N)
        for i in successors:
            if any(np.array_equal(elem, i) for elem in exploredSet):
                continue
            queue.append(i)

if __name__ == "__main__":
    #goal = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', 'E']]
    #start = [['1', '2', '3'], ['4', '5', '6'], ['7', 'E', '8']]
    M, N = 4, 4
    goal = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', 'E']]
    start = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', 'E', '11'], ['13', '14', '15', '12']]
    ans = bfs(np.asarray(start), np.asarray(goal), M, N)
    print(ans)    
    