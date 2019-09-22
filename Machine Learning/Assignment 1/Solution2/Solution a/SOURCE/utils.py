# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:34:47 2019

@author: ashima
"""

import config
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(acc_list, acc_list1, acc_list2):
    x = np.arange(config.NUM_EPOCHS)
    plt.plot(np.asarray(x), np.asarray(acc_list), label = 'No regularization', color = 'r')
    plt.plot(np.asarray(x), np.asarray(acc_list1), label = 'L1 Regularization', color = 'b')
    plt.plot(np.asarray(x), np.asarray(acc_list2), label = 'L2 Regularization', color = 'g')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Accuracy vs Epochs Plot')
    plt.legend()
    plt.savefig(config.OUT_DIR + 'Combined Accuracy_Plot.jpg')
    plt.show()

def plot_error(error_list, error_list1, error_list2):
    x = np.arange(config.NUM_EPOCHS)
    plt.plot(np.asarray(x), np.asarray(error_list), label = 'No regularization', color = 'r')
    plt.plot(np.asarray(x), np.asarray(error_list1), label = 'L1 Regularization', color = 'b')
    plt.plot(np.asarray(x), np.asarray(error_list2), label = 'L2 Regularization', color = 'g')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title('Error vs Epochs Plot')
    plt.legend()
    plt.savefig(config.OUT_DIR + 'Combined Error_Plot.jpg')
    plt.show()