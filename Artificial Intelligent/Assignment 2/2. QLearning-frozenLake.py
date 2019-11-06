# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:55:59 2019

@author: Ashima
"""
import random
import gym
import numpy as np 
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

n_actions = env.action_space.n
n_states = env.observation_space.n

learning_rates = np.linspace(0.2, 0.6, 10)
gammas = np.linspace(0.7, 0.99, 15)        
decay_rates = np.linspace(0.002, 0.006, 10)


#Optimal Hyperparameters
episodes = 20000                     
gamma = 0.99    
lr = 0.2    
max_steps = 99
decay_rate = 0.005      
exploration_rate = 1.0      
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 

Q = np.zeros((n_states, n_actions))

def q_learn(gamma, lr, exploration_rate, decay_rate):
    #print("gamma: ", gamma)
    reward_list = []
    scores = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        #step = 0
        for step in range(max_steps):
            #step += 1
            offset = random.uniform(0, 1)
            if offset <= exploration_rate:
                #Exploration
                action = env.action_space.sample()
            else: 
                #Exploitation
                action = np.argmax(Q[state,:])
                
            next_state, reward, done, _ = env.step(action)
            
            #Update Q value
            Q[state, action] = Q[state, action] + lr*(reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            total_rewards += reward
            state = next_state
            if done == True:
                break
        exploration_rate = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
        #env.render()
        reward_list.append(total_rewards)
        scores.append(sum(reward_list)/episodes)
    return scores
    #score = str(sum(reward_list)/episodes)
    #return score

if __name__ == "__main__": 
    #Settings 1
#    scores = []
#    for gamma in gammas:
#        for lr in learning_rates:
#            score = q_learn(gamma, lr, exploration_rate, decay_rate)
#            scores.append(score)
#            print("Gamma: {}, Learning Rate: {}, Score: {}".format(gamma, lr, score))
    #plt.plot(gammas, scores)
    #plt.plot(learning_rates, scores)
    
    #Settings 2
#    scores = []
#    for decay_rate in decay_rates:
#        score = q_learn(gamma, lr, exploration_rate, decay_rate)
#        scores.append(score)
#        print("Decay_rate: {}, Score: {}".format(decay_rate, score))
    
    print("Plot Score Vs time")
    scores = q_learn(gamma, lr, exploration_rate, decay_rate)
    plt.plot(np.arange(20000), scores)