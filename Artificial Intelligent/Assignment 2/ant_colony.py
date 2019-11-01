# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:16:46 2019

@author: Ashima
"""
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

iterations = 10
alpha = 1
beta = 1
#initialize with small random values
evaporation_factor = 0.8

class Ant_Colony:
    def __init__(self, distances, pheromone, total_ants):
        self.all_ants_path = []
        self.path_dist = []
        self.distances = distances
        self.total_ants = total_ants
        self.pheromone = pheromone
        
    def find_best_move(self, node, visit):
        probs = np.zeros(self.total_ants)
        for i in range(self.total_ants):
            if i not in visit: 
                probs[i] = ((self.pheromone[node][i]) ** alpha) * ((1.0/self.distances[node][i]) ** beta)
        #print("Probs: ", probs)
        norm_probs = probs/probs.sum()
        move = np.random.choice(self.total_ants, 1, p=norm_probs)[0]
        return move
    
    def calc_dist(self, cur, nextn):
        return self.distances[cur][nextn]
    
    def update_pheromone(self):
        #sorted_paths = sorted(self.all_ants_path)
        #shortest_distance = sorted_paths[0][0]
        self.pheromone = self.pheromone * (1 - evaporation_factor)
        for (path_dist, path) in self.all_ants_path:
            for i in range(len(path)-1):
            #print("path to be updated: ", (path[i], path[i+1]))
                self.pheromone[path[i]][path[i+1]] += (1/self.distances[path[i]][path[i+1]])
        
    def find_path(self, start):
        path = []
        visit = set()
        visit.add(start)
        cur = start
        path.append(start)
        path_dist = 0
        for i in range(self.total_ants-1):
            move = self.find_best_move(cur, visit)
            path.append(move)
            path_dist += self.calc_dist(cur, move)
            visit.add(move)
            cur = move
        path.append(start)
        path_dist += self.calc_dist(cur, start)
        return path_dist, path
    
    def print_all_path(self):
        print("Printing all the paths: ")
        for i in range(len(self.all_ants_path)):
            print(self.all_ants_path[i])
            
    def run(self):
        #for k iterations
        global pheromone
        for k in range(iterations):
            print("\nStarting {}th iteration ".format(k))
            self.all_ants_path = []
            #generate all possible paths (no of generated paths = no of ants)
            for i in range(self.total_ants):
                path_dist, path = self.find_path(0)
                self.all_ants_path.append((path_dist, path))
            self.print_all_path()
            self.update_pheromone()
    
    def visualize(self):
        G = nx.Graph()
        #G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
        G.add_nodes_from([1, 2, 3, 4, 5])
        for i in range(5):
            for j in range(i+1):
                if i!=j:
                    G.add_edge(i+1, j+1, weight = self.distances[i,j])
        positions = nx.spring_layout( G )
        nx.draw(G, positions)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,positions, edge_labels=labels)
        nx.draw_networkx_labels(G,positions,font_size=20,font_family='sans-serif')
        plt.savefig("Initial Matrix.png")
        plt.clf()
        
    def visualize_path(self):
        sorted_paths = sorted(self.all_ants_path)
        shortest_path = sorted_paths[0][1]
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4])
        for i in range(5):
            G.add_edge(shortest_path[i], shortest_path[i+1], weight = self.distances[shortest_path[i]][shortest_path[i+1]])
        positions = nx.spring_layout( G )
        nx.draw(G, positions)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,positions, edge_labels=labels)
        nx.draw_networkx_labels(G,positions,font_size=20,font_family='sans-serif')
        plt.savefig("Final Graph.png")
        

if __name__ == "__main__": 
    distances = np.array(np.random.randint(100, size = (5, 5)))
    #print(distances)
    #print(type(distances[0][0]))
    for i in range(len(distances)):
        distances[i][i] = 999
#    distances = np.array([[np.inf, 2, 8, 5, 7],
#                  [2, np.inf, 4, 8, 11],
#                  [8, 4, np.inf, 1, 3],
#                  [5, 8, 1, np.inf, 2],
#                  [7, 11, 3, 2, np.inf]])
    print(distances)
    pheromone = np.ones(distances.shape)*0.1
    total_ants = len(distances)
    ant = Ant_Colony(distances, pheromone, total_ants)
    ant.visualize()
    ant.run()
    ant.visualize_path()