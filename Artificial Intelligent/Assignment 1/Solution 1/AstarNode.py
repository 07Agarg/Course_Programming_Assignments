# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 05:05:16 2019

@author: ashima
"""


class Node:
    def __init__(self, state, parent, g, h):
        self.state = state              	#current_state
        self.parent = parent                #parent
        self.g = g                          #total cost incurred to reach at current node
        self.h = h                          #heuristic cost from current node
        self.f = self.g + self.h
    
    def get_f(self):    
        return self.f
    
    def get_state(self):
        return self.state
    
    def get_g(self):
        return self.g
   
#Working Heap    
"""
Heap Test
a = Node()
a.f = 10
b = Node()
b.f = 6
c = Node()
c.f = 1
heap = Heap(100)
heap.addNode(a)
heap.addNode(b)
heap.addNode(c)
heap.print_heap()
print("Here")
print(heap.delNode().f)
print("Remaining")
heap.print_heap()
"""
