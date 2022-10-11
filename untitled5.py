# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:07:40 2022

@author: ADmin
"""

import numpy as np
# import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
from brian2 import *
from math import factorial, exp
# import h5py
# %matplotlib inline

class Spike_generator :
    def __init__(self, weight, spontaneous_firing_rate, operation_time, time_step, network, refractory) :
        self.num_of_neurons = len(network[0]) # total neurons
        self.weight = weight # excitatory weight [1ms, 2ms, 3ms,...] 
        self.spontaneous_firing_rate = spontaneous_firing_rate # spontaneous firing rate
        self.operation_time = operation_time # operation_time
        self.time_step = time_step # unit time
        self.num_of_bin = round(operation_time/time_step) # Number of calculations during operation time
        self.history_window = len(weight) # The period during which weight affects.
        self.network = network # network 
        self.spike_record = {} # record how many seconds ago a spike occurred
        self.refractory = round(refractory/ms)
        
        
    def spike_gen(self) :
        
        self.spike_record = [[0 for col in range(self.history_window)] for row in range(self.num_of_neurons)] #reset
        count = [0 for col in range(self.num_of_neurons)] #reset
        time = [[0 for col in range(self.num_of_bin)] for row in range(self.num_of_neurons)] #reset        

        
        for k in range (self.num_of_bin) :
            
            probability = self.__weightsum() # probability =  weight
            # print(self.spike_record)       # for test
            # print(probability)             # for test
            
            
            for i in range (self.num_of_neurons) : # Repeat for all neurons at the current time

                if 1 in self.spike_record[i] :  # Check whether this neuron is refractory time or not.
                    self.__spike_record_update(self.spike_record[i]) # In refractory time, the probability is not calculated.
                    
                else :
                    
                    r = np.random.rand()    # Random number to compare with probability
                    probability[i] += np.log(self.spontaneous_firing_rate/Hz) #probability = ln(firing rate) + weight
                    probability[i] = np.exp(probability[i])*(time_step/second) #probability = exp (ln(firing rate) + weight) * unit time

                    self.__spike_record_update(self.spike_record[i])
                           
                    if(probability[i] >= r) : # spike occured 
                        time[i][k] = (k+1)*time_step # record current time
                        count[i] +=1 # The number of spikes
                        for j in range (self.history_window) :
                            if (self.spike_record[i][j] == 0) :
                                self.spike_record[i][j] =1
                                break
                            else :
                                pass
                            
                    else : # no spike
                        pass    
                
            # print(probability)  #for test
            # print(self.spike_record)  #for test

        
        # print(time) #for test
        print(count) #for test

        #########  for test (Record burst)
        # if(sum(count[0])>50) :
        #     f = open("C:/Users/my/Desktop/test.txt","a")
        #     f.write(str(count)+"\n")
        #     f.close()
        #########
        
        
        ######### save spike time
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
        return index, spike_time 

    # def verification(self, probability) :
        

    def __weightsum(self) : # weight Calculation
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
            if (self.network[source_index][j] == 1) : # if a neuron source has an excitability effect on neuron j
                sum_weight[j] += self.weight[history] # excitatory weight is added to j's weight
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
        
    def Draw_hist(self, repetition,lambda_,lambda_1,draw_num_of_spike = False ,draw_isi = False) :
        
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
        bin_edges =  np.arange(0,30,1)#=
        x = np.arange(30)#=
        if(draw_num_of_spike == True) :
            for k in range(self.num_of_neurons) :
                figure(dpi=600)
                pd1 = np.array([self.pois_dist(n, lambda_[k]*(self.duration / (1000*ms))) for n in range(30)])#poisson#=
                pd2 = np.array([self.pois_dist(n, lambda_1[k]*(self.duration / (1000*ms))) for n in range(30)])#poisson#=
                plt.xlabel('num_of_spikes')
                plt.ylabel('density')
                plt.title('neuron {} Firing rate distribution'.format(k))
                plt.hist(num_of_spikes[k],bin_edges,alpha = 0.5, density = True)
                if(lambda_1[k] == 0) :
                    pass
                else :
                    plt.plot(x, pd2, color='green')
                plt.plot(x, pd1, color='lightcoral')
                plt.xlim(0,30)#=
                plt.ylim(0,0.11)
                plt.xticks(np.arange(0,30,10))#=
                plt.yticks(np.arange(0,0.21,0.02))
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
                # print(isi[i])
                plt.xlabel('time(ms)')
                plt.ylabel('density')
                plt.xticks(np.arange(0,0.501,0.050), np.arange(0,501,50))
                plt.yticks(np.arange(0,15,2),np.arange(0,0.015,0.002))
                plt.xlim(0,0.50)
                plt.ylim(0,15)
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




    def Draw_weight_hist(self, repetition) :
           
        num_of_spikes = self.__reset()
        for i in range(repetition) : 
            index, spike_time = self.spike_gen() # repeat simulation
            # print(spike_time)

                
            for j in range(self.num_of_neurons) : # Record the firing rate every iteration
                num_of_spikes[j].append(index.count(j))
                # print(len(spike_time))
        
        for k in range(self.num_of_neurons):
            num_of_spikes[k].sort()
            total_num = len(num_of_spikes[k])
            total_num = round(total_num/2)
            num_of_spikes[k] = num_of_spikes[k][total_num]
        return num_of_spikes
     


def weight (tau, duration) :
    weight = [0 for col in range (duration)] 
    
    for i in range (duration) :
        # weight[i] = 0.5*np.log(2)*np.exp(-(i/tau))
        weight[i] = 0.1*np.exp(-(i/tau)) * 12.5
    return weight



weight_duration = 40
tau = 20
ref_excitatory = weight(tau, weight_duration)
weight = [ 0 for col in range (weight_duration)]
spontaneous_firing_rate =5* Hz
operation_time = 1000*ms
delta = 1*ms
refractory = 1*ms
network = [  [ 0, 0, 1, 0, 0, 0, 0, 0 ],
                  [ 1, 0, 0, 0, 0, 0, 0, 0 ],
                  [ 0, 0, 0, 0, 0, 0, 1, 0 ], 
                  [ 0, 1, 1, 0, 1, 0, 0, 0 ],
                  [ 0, 1, 0, 0, 0, 0, 0, 0 ],
                  [ 0, 0, 0, 0, 1, 0, 1, 0 ],
                  [ 0, 0, 0, 0, 0, 0, 0, 0 ],
                  [ 1, 0, 0, 1, 0, 0, 0, 0 ] ]
# network = [ [ 0 , 1 ],
#                   [ 0 , 0 ] ]
Repetition = 1000


# print(weight)
# ////////////////////////////////////////
# z = {}
# for i in range (len(network[0])) :
#     z[i] = []

# n = np.arange(1,35,0.5)
# for i in range (len(n)) :

#     for l in range(weight_duration) :
#         weight[l] = ref_excitatory[l]* (n[i])
#     print(weight)
#     A = Spike_generator( weight, spontaneous_firing_rate, operation_time, time_step, network, refractory)
#     # print(weight)
#     num_of_spikes = A.Draw_weight_hist(Repetition)
#     # print(num_of_spikes)
#     for j in range (len(network[0])) :
#         z[j].append(num_of_spikes[j])
    
# print(z)    

# for k in range(len(network[0])) :
#       figure(dpi=600)
#       plt.xlabel('Multiplication of reference weight')
#       plt.ylabel('num_of_spikes')
#       plt.title('neuron {} Firing rate distribution'.format(k))
#       # x = [ n  in range(20)]
#       x = n
#       plt.plot(x,z[k])
#       plt.ylim(1,1000)
#       plt.yscale('log')
#       # plt.yticks(np.arange(0,31,5))
#       plt.show()
# /////////////////////////////////////////////////////
# for i in range(weight_duration) :
    # ref_excitatory[i] *= 35


# aa = sum(ref_excitatory)
# print(aa)
A = Spike_generator( ref_excitatory, spontaneous_firing_rate, operation_time, time_step, network, refractory)
# B,c,count=A.spike_gen()

# for i in range(1000) :    
#     B,c=A.spike_gen()
# print(B);print(c)
# A.Draw_spike(B,c)
A.Draw_hist(Repetition,[8,8,8,6,7,5,8,5],[0,0,0,0,0,0,0,0],True,True)