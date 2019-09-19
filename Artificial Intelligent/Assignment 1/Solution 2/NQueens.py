# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:16:39 2019

@author: ashima
"""

import numpy as np
import random

N = 10
POPULATION_SIZE = 100

dx = [1, 1]   #, -1, -1]
dy = [-1, 1]  #, 1, -1]

#Total Non attacking pairs in Final Solution
TOTAL_NONATTACKING = ((N*(N-1))/2)

class Chromosome:
    def __init__(self):
        self.pos = []
        self.board = []
        self.fitness = 0

def isSafe(i, j):
    return i >= 0 and i < N and j >= 0 and j < N

def count_diagnol_attacks(x, y, chromosome):
    count = 0
    for i in range(2):
        n_x, n_y  = x, y
        while isSafe(n_x, n_y):
            n_x = n_x + dx[i]
            n_y = n_y + dy[i]
            if isSafe(n_x, n_y):
                if chromosome.board[n_x][n_y] == 'Q':
                    count += 1
    return count
#Calculate Column Attacks
def count_Vattacks(chromosome):
    count = 0
    for i in  range(N):
        p = np.count_nonzero(np.asarray(chromosome.board)[:,i] == 'Q')
        count += ((p * (p -1))/2)
    return count

#Calculate Row Attacks
def count_Hattacks(chromosome):
    count = 0
    for i in  range(N):
        p = np.count_nonzero(np.asarray(chromosome.board)[i] == 'Q')
        count += ((p * (p-1))/2)
    return count

#No of non attacking pairs
def calculateFitness(chromosome):
    attacks = 0
    Q_pos = np.where(chromosome.board == 'Q')
    Q_pos = np.transpose(Q_pos)
    for (x, y) in zip(range(N), chromosome.pos):
        attacks += count_diagnol_attacks(x, y, chromosome)
    attacks += count_Vattacks(chromosome)
    attacks += count_Hattacks(chromosome)
    non_attacks = TOTAL_NONATTACKING - attacks      
    return non_attacks

#Generate Board for given position array
def generate_board(all_pos):
    chromosome = []
    for i in all_pos:
        ch = ["." for _ in range(N)]
        ch[i] = "Q"
        chromosome.append(ch)
    return chromosome

#Complete State Formulation where every column of N*N board has atleast and at max 1 Queen in First Stage
def create_initial_population(population):
    
    for _ in range(POPULATION_SIZE):
        a = Chromosome()
        a.pos = random.sample(range(0, N), N)
        a.board = generate_board(a.pos)
        a.fitness = calculateFitness(a)
        population.append(a)
    return population

def reproduce_crossover(parent1, parent2):
    child = Chromosome()
    c1 = parent1.pos
    c2 = parent2.pos
    m = int(N/2)
    a = list(np.concatenate((c1[:m], c2[m:])))
    b = [x for x in range(N) if x not in a]
    a = list(set(a))
    a.extend(b)
    child.pos = a
    if child.pos == c1 or child.pos == c2:     #IMP: if child matches with any of the parent array, then mutate randomly 2 positions
        num = random.sample(range(0, N), 2)
        child.pos[num[0]], child.pos[num[1]] = child.pos[num[1]], child.pos[num[0]]
    child.board = generate_board(child.pos)
    child.fitness = calculateFitness(child)
    return child

def reproduce_mutate(parent2):
    num = random.sample(range(0, N), 6)
    child = parent2
    child.pos[num[0]], child.pos[num[1]] = child.pos[num[1]], child.pos[num[0]]
    child.pos[num[2]], child.pos[num[3]] = child.pos[num[3]], child.pos[num[2]]
    child.pos[num[4]], child.pos[num[5]] = child.pos[num[5]], child.pos[num[4]]
    return child

def print_best2population(population):
    print("Printing population")
    for i in range(3):
        print("Population {} :  {}, {}, {}".format(i, population[i].pos, population[i].board, population[i].fitness))

def GeneticAlgo():
    population = []
    create_initial_population(population)
    found = False
    i = 1
    while not found:
        population.sort(key = lambda x: x.fitness, reverse = True)
        print("Iteration: {}, Best Chromosome: {}, Fitness of Best Chromosome: {}".format(i, population[0].pos, population[0].fitness))
        i += 1
        if i == 100000:
            print_best2population(population)
            found = True
            return "OOPS! EXCEEDED!"
        if population[0].fitness == TOTAL_NONATTACKING:
            found = True
            return population[0].board
        new_generation = []
        new_generation.extend(population[:30])
        #Keep 30% best population, and generate 70% new population
        s = int((70*POPULATION_SIZE)/100)
        for _ in range(s):
            parent1 = random.choice(population[:10])
            parent2 = random.choice(population[:10])
            #Mutate the arrays after every 50 iterations
            if i % 25 == 0:
                child = reproduce_mutate(parent2)
            if i % 50 == 0:
                child = reproduce_mutate(parent1)
            else: 
                child = reproduce_crossover(parent1, parent2)
            new_generation.append(child)
        population = new_generation
    
if __name__ == "__main__":
    ans = GeneticAlgo()
    print(ans)