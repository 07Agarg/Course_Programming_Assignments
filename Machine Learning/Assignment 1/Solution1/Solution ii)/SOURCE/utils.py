# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:33:15 2019

@author: ashima
"""

import os
import config
import ast

def find_min_rmse_val():
    with open(os.path.join(config.OUT_DIR, config.RMSE_VAL_NORMAL_FILE) , "r") as file:
        rmse_val_list = ast.literal_eval(file.read())              
        rmse_min_val = rmse_val_list.index(min(rmse_val_list))
        print("Index of min RMSE on VAL Set ", rmse_min_val)
        print("Min Val of RMSE on VAL Set ", min(rmse_val_list))
        return rmse_min_val
    
print(find_min_rmse_val())