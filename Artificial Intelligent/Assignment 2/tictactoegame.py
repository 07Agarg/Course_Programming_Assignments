# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 03:53:01 2019

@author: Ashima
"""
import pickle
import numpy as np

class tictactoe:
    def __init__(self):
        self.playerNum = 1      #initially Player 1 gets a chance playerNum = -1 denotes player -1 is playing
        self.rows = 3
        self.cols = 3
        self.board = np.zeros((self.rows, self.cols))
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
            return 1
        elif sumval == -3 or sumval1 == -3:
            self.done = True
            return -1
        #Check Draw Case
        elif self.checkdraw() == True:
            return 0
        #Check Rows and Cols    
        else:
            for i in range(self.rows):
                sumval = sum(self.board[i,:])
                sumval1 = sum(self.board[:, i])
                if sumval == 3 or sumval1 == 3:
                    self.done = True
                    return 1
                elif sumval1 == 3 or sumval1 == -3:
                    self.done = True
                    return -1
        self.done = False
        return None
            
    def move(self, action):
        #print("board1: ", self.board)
        self.board[action[0]][action[1]] = self.playerNum 
        #print("board2: ", self.board)
        #self.display()
        if self.playerNum == 1: 
            self.playerNum = -1
            
        elif self.playerNum == -1:
            self.playerNum = 1
            
    def reset_board(self):
        self.board = np.zeros((3, 3))
        self.done = False
        self.playerNum = 1
    
    def allocate_reward(self, p1, p2):
        result = self.checkWinDraw()
        if result == -1:
            p1.backpropogate_reward(0)
            p2.backpropogate_reward(1)
        elif result == 1:
            p1.backpropogate_reward(1)
            p2.backpropogate_reward(0)
        else:
            p1.backpropogate_reward(0.1)
            p2.backpropogate_reward(0.5)
            
    def display(self):
        print(self.board)
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
        for episode in range(episodes):
            print("Episode num: ", episode)
            while not self.done:
                #Player 1
                locs = self.findVacantLoc()
                action = p1.nextAction(locs, self.board, self.playerNum)
                self.move(action)
                p1.addstate(self.board)
                if self.checkWinDraw() is not None:
                    self.allocate_reward(p1, p2)
                    self.reset_board()
                    p1.reset()
                    p2.reset()
                    break
                else:
                    #Player 2
                    locs = self.findVacantLoc()
                    action = p2.nextAction(locs, self.board, self.playerNum)
                    self.move(action) 
                    p2.addstate(self.board)
                    if self.checkWinDraw() is not None:
                        self.allocate_reward(p1, p2)
                        self.reset_board()
                        p2.reset()
                        p1.reset()
                        break
        
        p1.saveQVal("P1_QVal")
        p2.saveQVal("P2_QVal")
    
    def testPlay(self, p1, p2):
        while not self.done:
            locs = self.findVacantLoc()
            action = p1.nextAction(locs, self.board, self.playerNum)
            print("current action: ", action)
            self.move(action)
            self.display()
            win = self.checkWinDraw()
            if win is not None:
                if win == 1:
                    print("First Player wins!")
                elif win == 0:
                    print("Match Draws!")
            else:
                locs = self.findVacantLoc()
                action = p2.nextAction(locs)
                self.move(action)
                self.display()
                win = self.checkWinDraw()
                if win is not None:
                    if win == -1:
                        print("Second Player wins!")
                    elif win == 0:
                        print("Match Draws!")
            
            
            
class Player:
    def __init__(self, exp_rate):
        self.states = []
        self.state_value = {}
        self.exploration_rate = exp_rate
        self.decay_rate = 0.999
        self.learning_rate = 0.2
        self.gamma= 0.9
        
    def addstate(self, board):
        boardkey = str(board.reshape(3*3))
        self.states.append(boardkey)
        
    def reset(self):
        self.states = []
        
    def saveQVal(self, filename):
        file = open(str(filename), 'wb')
        pickle.dump(self.state_value, file)
        file.close()
        
    def loadQVal(self, filename):
        file = open(filename, 'rb')
        self.state_value = pickle.load(file)
        file.close()
        
    def nextAction(self, locations, board, playerNum):
        offset = np.random.uniform(0, 1)
        if offset <= self.exploration_rate:
            #Exploration 
            action = locations[np.random.choice(len(locations))]
        else:
            #Exploitation 
            new_board = board.copy()
            max_value = -99999
            value = 0
            for loc in locations:
                new_board[loc[0]][loc[1]] = playerNum
                boardkey = str(new_board.reshape(3*3))
                if self.state_value.get(boardkey):
                    if value >= max_value:
                        value = max_value 
                        action = loc
                else:
                    action = loc
                        
        return action
    
    def backpropogate_reward(self, reward):
        for state in reversed(self.states):
            if self.state_value.get(state) is None:
                self.state_value[state]= 0
            self.state_value[state] += self.learning_rate*(self.decay_rate*reward - self.state_value[state])
            reward = self.state_value[state]
                
class HumanPlayer:
    def __init__(self):
        self.row = 3
        self.col = 3
        
    def nextAction(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action

    # append a hash state
    def addState(self, state):
        pass

    # at the end of game, backpropagate and update states value
    def feedReward(self, reward):
        pass

    def reset(self):
        pass


    
if __name__ == "__main__": 
    #Two Agents player1, player2
    player1 = Player(0.3)
    player2 = Player(0.3)
    
    ttt = tictactoe()
    
    print("Teach to Play")
    ttt.learnToPlay(player1, player2, 1000)
    
    print("\n\nTest Learner")
    player1 = Player(0)
    player1.loadQVal("P1_QVal")
    
    player2 = HumanPlayer()
    
    ttt1 = tictactoe()
    ttt1.testPlay(player1, player2)
    