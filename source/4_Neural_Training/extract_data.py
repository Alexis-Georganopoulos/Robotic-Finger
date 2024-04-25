# -*- coding: utf-8 -*-

import numpy as np
import re
import collections
import matplotlib.pyplot as plt

def extract(filename):
    
    my_test = open(filename,"r")
    lines = my_test.readlines()
    
    my_data = [[]]
    
    temp = lines[1]
    temp = ','.join(re.findall(r'.{1,16}',temp))
    temp = temp.split(',')
    
    a = len(lines)
    b = len(temp)

    for j in range(len(lines)):
        
        line1 = lines[j]
        line11 = ','.join(re.findall(r'.{1,16}',line1))
        line111 = line11.split(',')
    
        mylist = []
    
        for i in range(len(line111)):
            mylist.append(float(line111[i]))
        
        my_data = np.append(my_data,[mylist])
    
    my_data = np.resize(my_data,(a,b))
    
    return my_data

def unshuffle(marray,indices):
    # we assume len(marray) = len(indices)
    mydic = dict(zip(indices,marray))
    omydic = collections.OrderedDict(sorted(mydic.items()))
    myout = [0]*len(indices)
    
    count = 0
    for k, v in omydic.items():
        myout[count] = v
        count = count + 1
    
    return myout

def macroplot(mbigarray,indices,suppvec = []):
    fig, axs = plt.subplots(5,4,figsize = [20,10])
    fig.tight_layout()
    row = 0
    col = 0
    for i in range(19):
        axs[row,col].plot(unshuffle(indices,indices),
                          unshuffle(mbigarray[:,i],indices),'r.')
        axs[row,col].set(xlabel = 'index',ylabel = 'X{}[mV]'.format(i+1))
        if any(suppvec):
            axs[row,col].plot(unshuffle(indices[suppvec],suppvec),
                              unshuffle(mbigarray[suppvec,i],suppvec),'bo')
        col = col + 1
        if col == 4:
            col = 0
            row = row + 1

    red, = axs[4,3].plot([],[],'r.')
    blue, = axs[4,3].plot([],[],'bo')
    axs[4,3].legend([red,blue],['training data','support vectors'])

