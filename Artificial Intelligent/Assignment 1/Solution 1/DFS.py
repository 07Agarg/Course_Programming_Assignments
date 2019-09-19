# -*- coding: utf-8 -*-
"""
Created on Tue Sep 3 05:23:48 2019

@author: ashima
"""

import numpy as np
from utils import find_successors, isCompleted

def dfs(start, goal, M, N):
    stack = []
    exploredSet = []
    stack.append(start)
    while stack:
        state = stack.pop()
        if isCompleted(state, goal, M, N):
            return "Solved"
        exploredSet.append(state)
        successors = find_successors(state, M, N)
        for i in successors:
            if any(np.array_equal(elem, i) for elem in exploredSet):
                continue
            stack.append(i)
    
    
if __name__ == "__main__":
    #N = 15
    goal = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', 'E']]  
    start = [['1', '2', '3'], ['5', '8', '6'], ['7', 'E', '4']]
    #goal = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', 'E']]
    #start = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', 'E', '11'], ['13', '14', '15', '12']]
    M, N = 3, 3
    #f = open("out.txt", "w+")
    ans = dfs(np.asarray(start), np.asarray(goal), M, N)
    print(ans)