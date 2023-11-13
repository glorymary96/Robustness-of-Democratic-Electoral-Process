#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:39:07 2023

@author: glory
"""
# Program to implement the robustness of the electoral system in a single electoral unit for Plurality systems and different values of confidence bound parameter.

#Include the Python file 'Strategies_w0.py' in the same folder of this program execution.

# Strategies_w0.py : Defines the influence vector.

# Create a folder with name 'MULT_ORD' to save the results.


import Strategies_w0 as STRG

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

# Not necessarily connected (NNC) network is considered

import numpy as np
import timeit
import multiprocessing as mp
import itertools
import os
import shutil
import random
start = timeit.default_timer()     

# Function to order the parties from left to right
def ordering(x,Num_ppl_party):
    for i in range(Num_ppl_party):
        argmax=np.argmax(x[i])
        left_elements=x[i,0:argmax]
        right_elements=x[i,argmax+1:Num_party]
        if len(left_elements)>0:
            left_elements=np.sort(left_elements)
            for j in range(len(left_elements)):
                x[i,j]=left_elements[j]
        if len(right_elements)>0:
            right_elements=np.sort(right_elements)[::-1]
            pos=argmax+1
            for j in range(len(right_elements)):
                x[i,pos]=right_elements[j]
                pos+=1             
    return x          

#Function to generate the natural opinion of agents uniformaly in a simplex.
def x0_spec_gen(num_party,Num_ppl_party):
    x0=np.zeros((Num_ppl_party,Num_party))
    Total_agents=0
    while Total_agents<Num_ppl_party:
        x=np.random.RandomState().uniform(low=0,high=1,size=(10000000,Num_party))
        x=x[np.where(np.sum(x,axis=1)<1.01)]
        x=x[np.where(np.sum(x,axis=1)>0.99)]
        for i in range(len(x)):
            x[i,np.argmax(x[i])],x[i,num_party]=x[i,num_party],x[i,np.argmax(x[i])]
        if Total_agents+len(x)<Num_ppl_party:
            x0[Total_agents:Total_agents+len(x)]=x
        else:
            required_number=Num_ppl_party-Total_agents
            x0[Total_agents:Num_ppl_party]=x[0:required_number]
        Total_agents+=len(x) 
    x0=x0/np.sum(x0,axis=1).reshape(Num_ppl_party,1) 
    x0=ordering(x0,Num_ppl_party)
    return x0   

# Function to generate natural opinion with almost same number of agents in each party
# Generates natural opinion and saves it in a 'npz' file
    
def Nat_type(num):
    x0=np.zeros((Num_ppl,Num_party)) 
    Num_ppl_count=0
    for num_party in range(Num_party):
        x0[Num_ppl_count:Num_ppl_count+Num_ppl_list[num,num_party]]=x0_spec_gen(num_party,Num_ppl_list[num,num_party])
        Num_ppl_count+=Num_ppl_list[num,num_party]  
    np.savez('MULTIPARTY_NO/NO_'+str(Num_party)+'_'+str(num)+'.npz',x0=x0)    
        
# Function to parallelize the generation of natural opinion   
def Natural_opinion():
    pool=mp.Pool(mp.cpu_count())
    pool.map(Nat_type,range(Num_simulation))
    pool.close() 
  
# Function to determine the winner of the election
# INPUT: Num_votes (Number of votes attained by each of the party)
# OUTPT: f1 returns the winner party       
def winner(Num_votes): 
    f1=np.argmax(Num_votes)
    return f1     

# Function to compute the effort needed to change the election's outcome
# INPUT :x0 (Natural opinion, matrix of size [Number of agents, Number of parties])
#        Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$), matrix of size [Number of agents, Number of agents]
#        Y (Final outcome, matrix of size [Number of agents, Number of parties])
#        Num_votes (An array with number of votes per party)
#        wp (Next winner of election after the influence)
#        w0 (Influence strength vector in support of 'wp' party)

# OUTPUT :E_min : sum of the influence vector
#         Num_agents_influenced: Number of agents influenced to change the election's outcome

def effort(x0,Mat_inv,Y,Num_votes,wp,w0):
    x0_copy=x0.copy()
    trial=(np.argsort(x0).argsort()[:,wp]==Num_party-1)*1*np.arange(1,Num_ppl+1,1)-1
    indices=np.delete(trial,trial==-1) # Finding the positions were winnerto be is already the winner
    x0_copy[indices]+=100
    z=np.argsort(np.linalg.norm((1/Num_party)-x0_copy,axis=1))  # Agents to be influenced first
 
    VECTOR1=x0+(0.05*w0)
    VECTOR=np.zeros((Num_ppl,Num_party))+0.1*w0
    count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
    while count>0:
        w0=0.5*w0
        VECTOR1[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=x0[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]+w0
        VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=0.5*VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]
        count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
        
    iter=0    
    f1=winner(Num_votes)
    f=1
    E_min=0
    while f==1 and iter<15:
        f=0
        b = 0
        while f1!=wp and b!=Num_ppl-1:
            T=np.zeros((Num_ppl,Num_party)) 
            for i in range(Num_party):
                T[:,i]=VECTOR[z[b],i]*Mat_inv[:,z[b]]
            Y=Y+T
            E_min+=np.sum(abs(VECTOR[z[b]]))
            Num_votes1=np.sum((np.argsort(Y).argsort()==Num_party-1)*1,axis=0)
            f1=winner(Num_votes1) 
            b=b+1
            if b==Num_ppl-1:
                f=1
                iter=iter+1
    Num_agents_influenced = (iter*len(x0))+b    
    return [E_min,Num_agents_influenced]

# Function to compute the election outcome and to determine the winner and runner-ups
def outcome(h):
    num,eps=int(M[h][0]), int(M[h][1])
    data=np.load('MULTIPARTY_NO/NO_'+str(Num_party)+'_'+str(num)+'.npz')
    x0=data['x0']
    e=e_range[eps]
    A=np.zeros((Num_ppl,Num_ppl))
    for i in range(len(x0)):
        Matrix=np.tile(x0[i],(len(x0),1))
        A[i,:]=((np.linalg.norm(Matrix-x0,ord=1,axis=1))<=e)*1
    A=A-np.identity(len(x0))
    
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    Y=np.matmul(Mat_inv,x0) 
    
    Num_votes=np.sum((np.argsort(Y).argsort()==Num_party-1)*1,axis=0)
    #print(Num_votes,Num_votes_initial)
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    WParg=np.argsort(Num_votes)
    party=Num_party-2
    
    for j in range(Num_party-1):  
        wp=int(WParg[party])
        Val_w0=STRG.Strategies(Num_party,wp)
        Effort[j],NUM_AGENTS_INFLUENCED[j]=effort(x0,Mat_inv,Y,Num_votes,wp, Val_w0)
        party=party-1
    np.savez('MULTIPARTY/Multipleparty_'+str(h)+'.npz', eps=eps, num=num, Effort=Effort,Num_votes=Num_votes,NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED)                       


if __name__ == "__main__": 
    Num_simulation=1000 # Number of simulation
    Num_ppl=2001  # Number of people or agents
    neps=201  # Number of epsilon to be considered
    e_range=np.linspace(0,2,neps)  # COmmunication distance range
    construction_norm=1  # Norm used to construct the network
    Par_start=3 # Party start
    Par_end=8 # Party end
    M = [(j, k) for j, k in itertools.product(range(0, Num_simulation, 1),range(0, neps, 1))]  
    
    os.mkdir('MULTIPARTY_NO')
    
    # To create a list of number of agents supporting each party 
    for Num_party in range(Par_start,Par_end):
        num_range=int(Num_ppl/Num_party)
        Num_ppl_list=np.zeros((Num_simulation,Num_party), dtype=int)
        for num in range(Num_simulation):
            R=0
            while R!=Num_ppl:
                Arr=np.array(random.sample(range(num_range-10,num_range+10), Num_party))
                R=np.sum(Arr)
            Num_ppl_list[num]=Arr
            
        Natural_opinion()    
    
    for Num_party in range(Par_start,Par_end):      
        os.mkdir('MULTIPARTY')
        pool=mp.Pool(mp.cpu_count())
        pool.map(outcome,range(len(M)))
        pool.close() 
        
        EFF=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_VOTES=np.zeros((Num_simulation,neps,Num_party))
        NUM_AGENTS_INFLUENCED=np.zeros((Num_simulation,neps,Num_party-1))

        
        for h in range(Num_simulation*neps):
            data=np.load('MULTIPARTY/Multipleparty_'+str(h)+'.npz')
            num,eps=int(M[h][0]),int(M[h][1])
            EFF[num,eps]=data['Effort']
           
            NUM_AGENTS_INFLUENCED[num,eps]=data['NUM_AGENTS_INFLUENCED']
            NUM_VOTES[num,eps,:]=data['Num_votes']
    
            
        np.savez('MULT_ORD/Multiparty_ordering_'+str(Num_party)+'.npz',Effort=EFF,e_range=e_range,NUM_VOTES=NUM_VOTES,NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED)
        shutil.rmtree('MULTIPARTY')
        
    shutil.rmtree('MULTIPARTY_NO')   

stop = timeit.default_timer()   
print('Time:',(stop-start)/(60*60))  
