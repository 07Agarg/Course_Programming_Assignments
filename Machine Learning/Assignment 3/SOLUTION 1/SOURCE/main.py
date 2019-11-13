# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:43:55 2019

@author: Ashima
"""

import config
import data
import model

if __name__ == "__main__":
    #print("Start")
    data = data.Data()
    data.read(config.TRAIN_INPUT, config.TRAIN_LABELS, True)
    print("Train data read successfully")
    
        
