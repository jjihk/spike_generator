# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:07:40 2022

@author: ADmin
"""

import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from brian2 import *
from math import factorial, exp
import h5py

# %matplotlib inline

class Spike_generator :
    def __init__(self, excitatory, firing_rate, duration, delta, connectivity, refractory) :
        self.num_of_neurons = len(connectivity[0]) # total neurons
        self.excitatory = excitatory # excitatory weight [1ms, 2ms, 3ms] 
        self.firing_rate = firing_rate #spontaneous firing rate
        self.duration = duration # duration time
        self.delta = delta # unit time
        self.num_of_bin = round(duration/delta) #total time / unit time
        self.history_window = len(excitatory) # The period during which excitatory affects.
        self.connectivity = connectivity # network 
        self.spike_record = {} # record how many seconds ago a spike occurred
        self.refractory = round(refractory/ms)
        
        
    def spike_gen(self) :
        
        self.spike_record = [[0 for col in range(self.history_window)] for row in range(self.num_of_neurons)] #reset
        count = [0 for col in range(self.num_of_neurons)] #reset
        time = [[0 for col in range(self.num_of_bin)] for row in range(self.num_of_neurons)] #reset        
        

        for k in range (self.num_of_bin) :
            
            probability = self.__weight() # probability = exp (ln(firing rate) + weight) * unit time
            # print(self.spike_record)                              # The value should be obtained from the spike record up to this time.
            print(probability)
            
            
            for i in range (self.num_of_neurons) : # Check whether this neuron is refractory time or not.
                    
                if(i==2) :
                    if 1 in self.spike_record[i] :
                        self.__spike_record_update(self.spike_record[i]) # In refractory time, the probability is not calculated.
                        
                    else :
                        
                        r = np.random.rand()    # Random number to compare with probability
                        probability[i] += np.log(self.firing_rate/Hz) #probability = exp (ln(firing rate) + weight) * unit time
                        probability[i] = np.exp(probability[i])*(delta/second) #probability = exp (ln(firing rate) + weight) * unit time
                        
                        self.__spike_record_update(self.spike_record[i])
                                
                               
                        if(probability[i] >= r) : # spike occured 
                            time[i][k] = (k+1)*delta # record current time
                            count[i] +=1 # The number of spikes
                            for j in range (self.history_window) :
                                if (self.spike_record[i][j] == 0) :
                                    self.spike_record[i][j] =1
                                    break
                                else :
                                    pass
                                
                        else : # update spike history
                            pass    
                    
                else :
                    
                    probability[i] += np.log(self.firing_rate/Hz) #probability = exp (ln(firing rate) + weight) * unit time
                    probability[i] = np.exp(probability[i])*(delta/second) #probability = exp (ln(firing rate) + weight) * unit time
                    
                    self.__spike_record_update(self.spike_record[i])
                            
                           
                     # spike occured 
                    time[i][k] = (k+1)*delta # record current time
                    count[i] +=1 # The number of spikes
                    for j in range (self.history_window) :
                        if (self.spike_record[i][j] == 0) :
                            self.spike_record[i][j] =1
                            break
                        else :
                            pass
                            
                    
                # if 1 in self.spike_record[i] :
                #     self.__spike_record_update(self.spike_record[i]) # In refractory time, the probability is not calculated.
                    
                # else :
                    
                #     r = np.random.rand()    # Random number to compare with probability
                #     probability[i] += np.log(self.firing_rate/Hz) #probability = exp (ln(firing rate) + weight) * unit time
                #     probability[i] = np.exp(probability[i])*(delta/second) #probability = exp (ln(firing rate) + weight) * unit time
                    
                #     self.__spike_record_update(self.spike_record[i])
                            
                           
                #     if(probability[i] >= r) : # spike occured 
                #         time[i][k] = (k+1)*delta # record current time
                #         count[i] +=1 # The number of spikes
                #         for j in range (self.history_window) :
                #             if (self.spike_record[i][j] == 0) :
                #                 self.spike_record[i][j] =1
                #                 break
                #             else :
                #                 pass
                            
                #     else : # update spike history
                #         pass    
                
            # print(probability)
            # print(self.spike_record)
        
        
        # print(time) 
        #print(count)

        spike_time = [0 for col in range(sum(count)) ]    
        sub_index  = [[] for row in range(self.num_of_neurons)]
        index =[]
        
        k=0
        for i in range (self.num_of_neurons) :
            for j in range (self.num_of_bin) :
                if (time[i][j]) : # spike time
                    spike_time[k] = time[i][j] #record spike time
                    k +=1
                else :
                    pass
                
        for i in range (self.num_of_neurons) :
            sub_index[i] = i*np.ones(count[i]).astype(int) # Save index as many spikes 
            index += list(sub_index[i]) 
        
        # print(index)
        # print(spike_time)
        # print(count)
        return index,spike_time

    # def verification(self, probability) :
        

    def __weight(self) : # weight Calculation
        sum_weight = {}
        for j in range (self.num_of_neurons) : #reset
            sum_weight[j] = 0
        
        #print(self.spike_record)
        for i in range (self.num_of_neurons) :
            for j in range (self.history_window) :
                k = self.spike_record[i][j]
                if(k==0) :
                    pass
                else :
                    sum_weight = self.__weight_update(sum_weight,i, k-1)
                
        #print(sum_weight)    
        return sum_weight
                
    def __weight_update(self, sum_weight ,source_index, history) : 
        for j in range (self.num_of_neurons) : # if neuron[source_index] spiked before
            if (self.connectivity[source_index][j] == 1) : # if a neuron source has an excitability effect on neuron j
                sum_weight[j] += self.excitatory[history] # excitatory weight is added to j's weight
            else :
                pass
        
        return sum_weight
    

    def __spike_record_update(self, spike_record) :
        for i in range(self.history_window) :
            if (spike_record[i] == 0 or spike_record[i] == self.history_window) :
                spike_record[i] = 0
            else :
                spike_record[i] = spike_record[i] + 1
        


             
    def Draw_spike(self, index, spike_time) : #draw raster plot

        time = [[]for row in range(self.num_of_neurons)]

        k=0
        for i in range(self.num_of_neurons) :
            time[i].extend(spike_time[k:k+index.count(i)]/second)
            k+=index.count(i)
        # print(time)
        plt.figure(dpi = 600)
        lineoffsets1 = range(0,self.num_of_neurons)        
        plt.eventplot(time, lineoffsets=lineoffsets1,color='black',alpha=0.6)
        plt.xticks(range(0,int(self.duration/second)+1,1))
        plt.yticks(range(0,self.num_of_neurons))
        plt.xlim(0,self.duration/second)
        plt.ylim([-0.5,self.num_of_neurons-0.5])
        plt.xlabel("time(s)")
        plt.ylabel("neuron number")
        
    def Draw_hist(self, repetition,lambda_,draw_num_of_spike = False ,draw_isi = False) :
        
        if (draw_num_of_spike == True and draw_isi == True) : 
            num_of_spikes = self.__reset()
            isi = self.__reset()
        elif (draw_num_of_spike == True and draw_isi == False) :
            num_of_spikes = self.__reset()
        elif  (draw_num_of_spike == False and draw_isi == True) :        
            isi = self.__reset()
        else :
            pass
        
        
        for i in range(repetition) : 
            index, spike_time = A.spike_gen() # repeat simulation
            # print(spike_time)
            if(draw_num_of_spike == True ) :
                
                for j in range(self.num_of_neurons) : # Record the firing rate every iteration
                    num_of_spikes[j].append(index.count(j))
                    # print(len(spike_time))
            else :
                pass
            
            if(draw_isi == True) : # Record the inter-spike interval every iteration
                isi = self.__isi_distribution(spike_time,index,isi)
                # print(spike_time)
                # print(isi)
            else :
                pass
                
        #print(num_of_spikes)
        #print(isi)
        bin_edges =  np.arange(0,150,5)
        x = np.arange(150)
        if(draw_num_of_spike == True) :
            for k in range(self.num_of_neurons) :
                figure(dpi=600)
                pd1 = np.array([self.pois_dist(n, lambda_[k]*(self.duration / (1000*ms))) for n in range(150)])
                plt.xlabel('num_of_spikes')
                plt.ylabel('density')
                plt.title('neuron {} Firing rate distribution'.format(k))
                plt.hist(num_of_spikes[k],bin_edges,alpha = 0.5, density = True)
                plt.plot(x, pd1, color='lightcoral')
                plt.xlim(0,150)
                plt.ylim(0,0.11)
                plt.xticks(np.arange(0,150,10))
                plt.yticks(np.arange(0,0.11,0.02))
                plt.show()
        else :
            pass
        
        if(draw_isi == True) :   
            for i in range (self.num_of_neurons) :
                figure(dpi=600)
                x = np.arange(0,max(isi[i]/second),0.001)
                x1 = np.arange(0,max(isi[i]/second),0.005)
                plt.hist((isi[i]/second),x1,density = True)
                plt.plot(x,(lambda_[i])*np.exp(-(lambda_[i])*(x)))
                print(isi[i])
                plt.xlabel('time(ms)')
                plt.ylabel('density')
                plt.xticks(np.arange(0,0.151,0.025), np.arange(0,151,25))
                plt.yticks(np.arange(0,110,20),np.arange(0,0.11,0.02))
                plt.xlim(0,0.15)
                plt.ylim(0,100)
                plt.title('neuron{} ISI distribution'.format(i))
                plt.show()
                #########################################################average
                # a = sum(isi[i])/len(isi[i])
                # print(a)
                #########################################################
        else :
            pass
        
    def __isi_distribution(self,spike_time,index,isi) :      

        for i in range(self.num_of_neurons) :
            for j in range(index.count(i)-1) :
                if(i==0) :
                    isi[i].append((spike_time[j+1]-spike_time[j]))
                else :
                    a=0
                    for k in range(i) :
                        a += index.count(k)
                    isi[i].append((spike_time[j+a+1]-spike_time[j+a]))

        return isi
    

   
    def pois_dist(self, n, lamb):
        pd = (lamb ** n) * exp(-lamb) / factorial(n)
        return pd     


    def __reset(self) : 
        subject = {}
        for i in range(self.num_of_neurons) :
            subject[i] = []
        return subject


excitatory = [ 1,1,1,1,1,1,1,1,1,1 ]
firing_rate =5 * Hz
duration = 10*ms
delta = 1*ms
refractory = 1*ms


connectivity = [  [ 0, 0, 1 ],
                  [ 0, 0, 1 ],
                  [ 0, 0, 0 ]]


# connectivity = [  [  0,  0,  1 ],
#                   [  1,  0,  1 ],
#                   [  1,  0,  0 ] ]
# [38,20,38]

# connectivity = [  [ 0, 1, 1, 0, 0 ],
#                   [ 0, 0, 1, 0, 0 ],
#                   [ 0, 0, 0, 1, 1 ], 
#                   [ 1, 1, 0, 0, 1 ],
#                   [ 1, 1, 0, 0, 0 ]  ]
# [50,70,63,35,50]

# connectivity = [  [ 0, 1, 0, 1, 1, 0, 0, 0 ],
#                   [ 0, 0, 1, 1, 1, 1, 0, 0 ],
#                   [ 0, 0, 0, 0, 1, 1, 0, 0 ], 
#                   [ 0, 0, 0, 0, 1, 0, 1, 0 ],
#                   [ 0, 0, 0, 0, 0, 1, 0, 1 ],
#                   [ 0, 0, 0, 0, 0, 0, 0, 0 ],
#                   [ 0, 0, 0, 0, 1, 0, 0, 1 ],
#                   [ 0, 0, 0, 1, 0, 1, 0, 0 ]
#                   ]
# [20,25,28,55,115,110,33,65,83]

    
Repetition = 1000
A = Spike_generator( excitatory, firing_rate, duration, delta, connectivity, refractory)
B,c=A.spike_gen()
# print(B);print(c)
# A.Draw_spike(B,c)
# A.Draw_hist(Repetition,[20,20,20],True,True)
