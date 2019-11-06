# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:06:39 2019

@author: Ashima
"""

import numpy as np
import random
import itertools
import pickle
from time import sleep

#player = ['1','-1','0']
player = [1, -1, 0]
all_possible_states = [[list(i[0:3]),list(i[3:6]),list(i[6:10])] for i in itertools.product(player, repeat = 9)]
n_states = len(all_possible_states) # 2 players, 9 spaces
n_actions = 9   # 9 spaces

Qtable1 = np.ones((n_states, n_actions)) * 0.6
Qtable2 = np.ones((n_states, n_actions)) * 0.6

gamma = 0.7
learning_rate = 0.2
episodes = 20000
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 

dict1 = {}
dict1[(0, 0)] = 0
dict1[(0, 1)] = 1
dict1[(0, 2)] = 2
dict1[(1, 0)] = 3
dict1[(1, 1)] = 4
dict1[(1, 2)] = 5
dict1[(2, 0)] = 6
dict1[(2, 1)] = 7
dict1[(2, 2)] = 8

class tictactoe:
    def __init__(self):
        self.playerNum = 1      #initially Player 1 gets a chance playerNum = -1 denotes player -1 is playing
        self.rows = 3
        self.cols = 3
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.done = False
        
    def findVacantLoc(self):
        locations = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    locations.append((i, j))
        return locations
    
    def checkdraw(self):
        if len(self.findVacantLoc()) == 0:
            self.done = True
            return True
        return False
    
    def checkWinDraw(self):
        sumval = 0
        #Return 0 for draw case, 1 if player1 wins, -1 if player2 wins, None if not from any of these cases 
        
        #Check Diagnols
        sumval = sum(self.board[i][i] for i in range(self.rows))     #diagnol1
        sumval1 = sum(self.board[i][self.rows-1-i] for i in range(self.rows))  #diagnol2
        if sumval == 3 or sumval1 == 3:
            self.done = True    
            return 100, 0
        if sumval == -3 or sumval1 == -3:
            self.done = True
            return 0, 100
        
        #Check Rows and Cols    
        row_sums = np.sum(np.array(self.board), 1)
        col_sums = np.sum(np.array(self.board), 0)
        if row_sums[0] == 3 or row_sums[1] == 3 or row_sums[2] == 3 or col_sums[0] == 3 or col_sums[1] == 3 or col_sums[2] == 3:
            self.done = True
            return 100, 0
        elif row_sums[0] == -3 or row_sums[1] == -3 or row_sums[2] == -3 or col_sums[0] == -3 or col_sums[1] == -3 or col_sums[2] == -3:
            self.done = True
            return 0, 100
        
        #Check Draw Case
        if self.checkdraw() == True:
            return 10, 50
        self.done = False
        return 10, 50
            
    def move(self, action):
        self.board[action[0]][action[1]] = self.playerNum 
        if self.playerNum == 1: 
            self.playerNum = -1
            
        elif self.playerNum == -1:
            self.playerNum = 1
        reward1, reward2 = self.checkWinDraw()  #Reward gets +1s if player 1 wins, -1 if player 2 wins, 0.5 draws, 0 if none of these cases
        return reward1, reward2
            
    def reset_board(self):
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.done = False
        self.playerNum = 1
    
    def display(self):
        #print(self.board)
        for i in range(0, self.rows):
            print('-------------')
            out = '| '
            for j in range(0, self.cols):
                if self.board[i][j] == 1:
                    token = 'x'
                if self.board[i][j] == -1:
                    token = 'o'
                if self.board[i][j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
    
    def learnToPlay(self, p1, p2, episodes):
        reward_list = []
        for episode in range(episodes):
            print("Episode num: ", episode)
            self.reset_board()
            while not self.done:
                #Player 1
                locs = self.findVacantLoc()
                state = all_possible_states.index(self.board)
                action, loc = p1.nextAction(locs, self.board, self.playerNum)
                reward1, reward2 = self.move(loc)
                next_state = all_possible_states.index(self.board)      
                Qtable1[state, action] = (1.0 - learning_rate)*Qtable1[state, action] + learning_rate*(reward1 + gamma * np.max(Qtable1[next_state, :]))# - Qtable1[state, action]
                Qtable2[state, action] = (1.0 - learning_rate)*Qtable2[state, action] + learning_rate*(reward2 + gamma * np.max(Qtable2[next_state, :]))
                if self.done == True:
                    reward_list.append(reward1)
                    break
                else:
                    #Player 2
                    locs = self.findVacantLoc()
                    state = all_possible_states.index(self.board)
                    action, loc = p2.nextAction(locs, self.board, self.playerNum)
                    reward1, reward2 = self.move(loc) 
                    next_state = all_possible_states.index(self.board)  
                    Qtable1[state, action] = (1.0 - learning_rate)*Qtable1[state, action] + learning_rate*(reward1 + gamma * np.max(Qtable1[next_state, :]))# - Qtable1[state, action]
                    Qtable2[state, action] = (1.0 - learning_rate)*Qtable2[state, action] + learning_rate*(reward2 + gamma * np.max(Qtable2[next_state, :]))
                    if self.done == True:
                        break
            print("Scores: ", sum(reward_list)/(episode+1))
        
        p1.saveQVal("P1_QVal_20k", Qtable1)
        p2.saveQVal("P2_QVal_20k", Qtable2)
        
    def loadQVal(self, filename, i):
        file = open(filename, 'rb')
        if i == 1:
            global Qtable1
            Qtable1 = pickle.load(file)
        elif i == 2:
            global Qtable2
            Qtable2 = pickle.load(file)
        file.close()    
        
    def testPlay(self, p1, p2):
        self.loadQVal("P1_QVal_20k", 1)
        self.loadQVal("P2_QVal_20k", 2)
        while not self.done:
            #Player 1 
            locs = self.findVacantLoc()
            #print("1 board: ", self.board)
            print("Available Positions: ", len(locs))
            action, loc = p1.nextAction(locs, self.board, self.playerNum)
            reward1, reward2 = self.move(loc)
            self.display()
            if reward1 == 100:
                print("First Player wins!-1")
            elif reward1 == 20:
                print("Match Draws!-1")
            elif reward2 == 100:
                print("Second Player wins!-1")
            else:
                #Player 2
                locs = self.findVacantLoc()
                #print("2 board: ", self.board)
                print("Available Positions: ", len(locs))
                action, loc = p2.nextAction(locs, self.board, self.playerNum)
                reward1, reward2 = self.move(loc)
                self.display()
                if reward1 == 100:
                    print("First Player wins!-2")
                elif reward1 == 20:
                    print("Match Draws!-2")
                elif reward2 == 100:
                    print("Second Player wins!-2")

    
class Player:
    def __init__(self, exp_rate, train):
        self.decay_rate = 0.05
        self.train = train
        self.exploration_rate = exp_rate
        print("exploration rate: ", self.exploration_rate)
        
    def reset(self):
        self.states = []
        
#    def updateExplorationRate(self, episode):
#        self.exploration_rate = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-self.decay_rate*episode)
        
    def saveQVal(self, filename, Qtable):
        file = open(str(filename), 'wb')
        pickle.dump(Qtable, file)
        file.close()
        
    def nextAction(self, locations, board, playerNum):
        if self.train:
            offset = np.random.uniform(0, 1)            #self.exploration_rate <= 0.3 explore, else exploitation
            if offset <= self.exploration_rate:
                #Exploration 
                #print("Exploring...")
                foundLoc = locations[np.random.choice(len(locations))]
                action = dict1[foundLoc]
            else:
                #Exploitation 
                #print("Exploiting...")
                state = all_possible_states.index(board)
                maxvalue = -9999
                action = 0
                if playerNum == 1:
                    for loc in locations:
                        if maxvalue <= Qtable1[state, dict1[loc]]:
                            maxvalue = Qtable1[state, dict1[loc]]
                            action = dict1[loc]
                            foundLoc = loc
                elif playerNum == -1:
                    for loc in locations:
                        if maxvalue <= Qtable2[state, dict1[loc]]:
                            maxvalue = Qtable2[state, dict1[loc]]
                            action = dict1[loc]
                            foundLoc = loc
                #print("action, found Loc: \n", action, foundLoc)
            return action, foundLoc
        else:
            if playerNum == -1:
                while True:
                    row = int(input("Input Row: "))
                    col = int(input("Input Col: "))
                    loc = (row, col)
                    if loc in locations:
                        return dict1[loc], loc
            else:
                state = all_possible_states.index(board)
                maxvalue = -9999
                for loc in locations:
                    if maxvalue <= Qtable1[state, dict1[loc]]:
                        maxvalue = Qtable1[state, dict1[loc]]
                        action = dict1[loc]
                        foundLoc = loc
                return action, foundLoc
        
if __name__ == "__main__": 
    #Two Agents player1, player2
    player1 = Player(0.3, True)               #Set True when training, false when testing against human player
    player2 = Player(0.3, True)
    
    ttt = tictactoe()
    
    print("Teach to Play")
    #ttt.learnToPlay(player1, player2, episodes)
    
    #sleep(300)
    print("\n\nTest Learner")
    player1 = Player(0.0, False)
    
    player2 = Player(0.0, False)
    
    ttt1 = tictactoe()
    ttt1.testPlay(player1, player2)