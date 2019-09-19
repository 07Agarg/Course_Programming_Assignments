    # -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:06:48 2019

@author: ashima
"""

#%matplotlib inline
import BFS
import DFS
import Astar
import IDA_Star
import time
import numpy as np
import matplotlib.pyplot as plt

def main(start, goal, A, B, N):
    t = []
    start_time_bfs = time.time()
    BFS.bfs(np.asarray(start), np.asarray(goal), A, B)
    t1 = (time.time() - start_time_bfs)
    print("BFS executing time for N = {}: {} seconds".format(N, t1))
    
    #start_time_dfs = time.time()
    #DFS.dfs(np.asarray(start), np.asarray(goal), A, B)
    #t2 = (time.time() - start_time_dfs)
    #print("DFS executing time for N = {}: {} seconds".format(N, t2))
    
    start_time_Astar = time.time()
    Astar.Astar(np.asarray(start), np.asarray(goal), A, B)
    t3 = (time.time() - start_time_Astar)
    print("AStar executing time for N = {}: {} seconds".format(N, t3))
    
    start_time_IDAStar = time.time()
    IDA_Star.IDAStar(np.asarray(start), np.asarray(goal), A, B)
    t4 = (time.time() - start_time_IDAStar)
    print("IDA Star executing time for N = {}: {} seconds".format(N, t4))
    
    t = [t1, t3, t4]
    return t

if __name__ == "__main__":    
    #N = 8
    print("For N - 8")
    M, N = 3, 3 
    goal = [['1', '2', '3'], ['5', '8', '6'], ['E', '7', '4']]
    start = [['1', '2', '3'], ['5', '6', 'E'], ['7', '8', '4']]
    
    t1 = main(start, goal, M, N, 8)
    print("\n\n\n")
    print("For N-15")
    #N = 15
    M, N = 4, 4
    goal = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', 'E']]
    start = [['1', '2', '3', '4'], ['5', '6', 'E', '8'], ['9', '10', '7', '11'], ['13', '14', '15', '12']]
    
    t2 = main(start, goal, M, N, 15)
    print("\n\n\n")
    print("For N - 24")
    #N = 24
    M, N = 5, 5
    goal = [['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '10'], ['11', '12', '13', '14', '15'], ['16', '17', '18', '19', '20'], ['21', '22', '23', '24', 'E']]
    start = [['1', '2', '3', '4', '5'], ['6', '7', '8', '9', '10'], ['11', '12', '13', 'E', '15'], ['16', '17', '18', '14', '19'], ['21', '22', '23', '24', '20']]
    
    t3 = main(start, goal, M, N, 24)
    print("\n\n\n")
    
    #Plot the graph to show analysis
    #N = [8, 16, 24]
    barwidth = 0.25
    # Set position of bar on X axis
    r1 = np.arange(len(t1))
    r2 = [x + barwidth for x in r1]
    r3 = [x + barwidth for x in r2]
 
    colors = ['r', 'b', 'g']
    
    plt.bar(r1, t1, width = barwidth, color = colors[0], label = 'N = 8')
    plt.bar(r2, t2, width = barwidth, color = colors[1], label = 'N = 15')
    plt.bar(r3, t3, width = barwidth, color = colors[2], label = 'N = 24')
    
    plt.xticks([r + barwidth for r in range(len(t1))], ['BFS', 'A-Star', 'IDA-Star'])
    plt.xlabel('Algorithms')
    plt.ylabel('Execution Time')
    plt.title('Time Analysis')
    
    plt.legend()
    plt.show()
    