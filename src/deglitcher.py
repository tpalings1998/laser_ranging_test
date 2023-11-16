# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 10:38:08 2023

@author: tijsp
"""



from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



class deglitcher:
    
    def __init__(self):
        #self.Q = Q
        return
    
    def rising_edge(self,A,threshold=1):
        A = A.astype(np.float64)
        diff = np.diff(A)
        return np.where(diff==threshold)[0]
    
    def falling_edge(self,A):
        A = A.astype(np.float64)
        diff = np.diff(A)
        return np.where(diff==-1)[0]
    
    def deglitch(self,Q):
        # Mean edge detection = higher resolution.
        # Copy the original array to not mess it up in memory
        Q_deglitched = np.copy(Q).astype(np.float64)
        # Find the index differences between transitions
        
        # if exactly 1 then rising else falling
        rising_edges  = self.rising_edge(Q_deglitched)
        # Rough beat frequency is found by the mode of the differences in the rising edges
        u_beat = stats.mode(np.diff(rising_edges))[0][0]
        print(u_beat)
        # Threshold moreira =25% of beat frequency, rounded to integer
        u_t = int(0.25*u_beat)

        
        
        for i in rising_edges:
            #select region to check for glitches
            to_check    = range(i,i+u_t)
            # get all the rising edges indexes belonging in this region relative to the "to check" region
            try:
                small_edges = np.where(np.diff(Q[to_check])==1)[0]
            except IndexError:
                small_edges = np.where(np.diff(Q[i:-1])==1)[0]
                
            if len(small_edges)>1:
            # calculate the average index for the rising edges
            # There is a bug where the rising edges of glitches at the falling edge
            # are deglitched as well, so check for lots of ones to basically see if falling or rising
                if len(np.where(Q_deglitched[to_check]==1)[0])>len(np.where(Q_deglitched[to_check]==0)[0]):
                    mean_edge = int(np.average(small_edges))
                    #print(i+mean_edge)
                    Q_deglitched[i+mean_edge:i+u_t] = 1
                
        # repeat the procedure to get rid of the residual terms that get left behind when setting ones
        rising_edges  = self.rising_edge(Q_deglitched)
        #falling_edge = np.where(diff==-1)[0]
        
        for i in rising_edges:
            to_check    = range(i,i+u_t)
            # get all the rising edges indexes belonging in this region relative to the "to check" region
            try:
                small_edges = np.where(np.diff(Q[to_check])==1)[0]
            except IndexError:
                small_edges = np.where(np.diff(Q[i:-1])==1)[0]
            if len(small_edges)>1:
                if len(np.where(Q_deglitched[to_check]==1)[0])>len(np.where(Q_deglitched[to_check]==0)[0]):
            # All the rising edges are set, but the ones that are left over from before mean-selecting must be set to 0
                    Q_deglitched[i:i+small_edges[-1]] = 0
                    
        diff = np.diff(Q_deglitched)
        falling_edges = np.where(diff==-1)[0]
                
        

        for i in falling_edges:
            #For edge detection, the index shifts one, and then again for taking the difference, so compensate
            i+=2
            to_check    = range(i-u_t,i)
            # get all the rising edges indexes belonging in this region relative to the "to check" region
            small_edges = np.where(np.diff(Q_deglitched[to_check])==-1)[0]
            if len(small_edges)>1:

                mean_edge = int(np.average(small_edges))
                #The starting point is i-ut, until i-ut+mean, +1 to compensate for slicing
                Q_deglitched[i-u_t:i-u_t+mean_edge+1] = 1
                Q_deglitched[i-u_t+mean_edge+1:i] = 0
        
        
        return Q_deglitched
    
    
    def consistency_check(self,Q):
        
        R = self.rising_edge(Q)
        F = self.falling_edge(Q)
        
        average_time_rising = int(np.average(np.diff(R)))
        std_time_rising = int(np.std(np.diff(R)))
        average_time_falling = int(np.average(np.diff(F)))
        std_time_falling = int(np.std(np.diff(F)))
        
        
        
        dR = np.where((np.diff(R))<(average_time_rising-std_time_rising))[0] + 1
        dF = np.where((np.diff(R))<(average_time_falling-std_time_falling))[0] + 1
        Q[R[dR]:F[dF]] = 0
        return Q
            
    
        