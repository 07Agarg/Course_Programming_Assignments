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

episodes = 20000                     
learning_rate = 0.5         
gamma= 0.95                  
max_steps = 99

decay_rate = 0.005      
exploration_rate = 1.0      
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 

Q = np.zeros((n_states, n_actions))
reward_list = []

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
        Q[state, action] = Q[state, action] + learning_rate*(reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        
        total_rewards += reward
        state = next_state
        if done == True:
            break
    exploration_rate = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    env.render()
    reward_list.append(total_rewards)

print("Score over time: ", str(sum(reward_list)/episodes))