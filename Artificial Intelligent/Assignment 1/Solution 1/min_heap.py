# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 04:55:04 2019

@author: ashima
"""

import numpy as np

class Heap:
    def __init__(self, Capacity):
        self.heap = []
        self.length = 0
        self.capacity = Capacity
    
    def get_parent(self, index):
        return int(index/2)
    
    def isEmpty(self):
        if self.length == 0:
            return True
        return False
    
    def right_child(self, index):
        return 2*index + 2
    
    def left_child(self, index):
        return 2*index + 1
    
    def get_top(self):
        if self.length == 0:
            return None
        return self.heap[0]
    
    def addNode(self, node):
        if self.length >= self.capacity:
            return None
        self.heap.append(node)
        self.length = self.length + 1
        self.heapify_up()
    
    def compare(self, a, b):
            return a < b
        
    def delNode(self):
        if self.length == 0:
            print("Empty Heap")
            return 
        elif self.length == 1:
            top = self.get_top()
            self.__init__(self.capacity)
            return top
        else :
            top = self.get_top()
            self.heap[0], self.heap[self.length - 1] = self.heap[self.length - 1], self.heap[0]
            del(self.heap[-1])
            self.length = self.length - 1
            self.heapify_down(0)
            return top
 
    def heapify_up(self):
        if self.length == 0:
            return 
        else:
            i = self.length - 1
            while i != 0 and self.compare(self.heap[i].get_f(), self.heap[self.get_parent(i)].get_f()):
                self.heap[i] , self.heap[self.get_parent(i)] = self.heap[self.get_parent(i)], self.heap[i]
                i = self.get_parent(i)
    
    def heapify_down(self, i):
        l = self.left_child(0)
        r = self.right_child(0)
        smallest = i
        if l < self.length and self.compare(self.heap[l].get_f(), self.heap[smallest].get_f()):
            self.heap[l], self.heap[smallest] = self.heap[smallest], self.heap[l]
        if r < self.length and self.compare(self.heap[r].get_f(), self.heap[smallest].get_f()):
            self.heap[r], self.heap[smallest] = self.heap[smallest], self.heap[r]
        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify_down(smallest)
            
    def print_heap(self):
        for i in range(self.length):
            #print(self.heap[i].f)
            print(self.heap[i].state_string, self.heap[i].parent)
        