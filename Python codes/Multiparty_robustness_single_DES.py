#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:48:14 2023

@author: glory
"""

# Program to implement the robustness of the electoral system in a single electoral unit for different electoral systems and different values of confidence bound parameter.
# Include the python file 'Nat_opn_generator.py' and 'Strategies_w0.py' in the same folder of this program execution.
# Nat_opn_generator.py : Generates the natural opinion in the simplex with equal density and volume for each party.
# Strategies_w0.py : Defines the influence vector.

# Create a folder with name 'EPS_CRITICAL' to save the results.

import Nat_opn_generator as OPN_GEN

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


start = timeit.default_timer()     

# Function to generate natural opinion with almost same number of agents in each party
# Generates natural opinion and saves it in a 'npz' file

def Nat_type(num):
    x0=OPN_GEN.Random_generation(Num_ppl, Num_party)
    Num_votes=np.sum(np.argsort(x0).argsort()==Num_party-1,axis=0)
    while np.max(Num_votes)-np.min(Num_votes)>20:
        x0=OPN_GEN.Random_generation(Num_ppl, Num_party)
        Num_votes=np.sum(np.argsort(x0).argsort()==Num_party-1,axis=0)
        
    np.savez('MULTIPARTY_NO/NO_'+str(Num_party)+'_'+str(num)+'.npz',x0=x0)          
    
# Function to parallelize the generation of natural opinion   
def Natural_opinion():
    pool=mp.Pool(mp.cpu_count())
    pool.map(Nat_type,range(Num_simulation))
    pool.close() 

# Function to determine the winner of the election
# INPUT: y (Final opinion of agents)
#        VOTING_TYPE: electoral systems (Plurality, Ranked Choice Voting(RCV), TRS (Two-round system))
# OUTPT: f1 returns the winner party 

def winner(y, VOTING_TYPE): 
    N_V=np.zeros((Num_party-1,Num_party))
    
    if VOTING_TYPE=='Plurality':
        Num_votes=np.sum(np.argsort(y).argsort()==Num_party-1,axis=0)
        f1=np.argmax(Num_votes)
    
    elif VOTING_TYPE=='RCV':
        Num_votes=np.sum(np.argsort(y).argsort()==Num_party-1,axis=0)
        y_temp=y.copy()
        N_V[0]=Num_votes
        pos_nv=0
        while (np.max(Num_votes)/Num_ppl)*100<50:
            Num_votes[Num_votes==0]=1e5
            pos=np.argmin(Num_votes)
            y_temp[:,pos]=0
            Num_votes=np.sum(np.argsort(y_temp).argsort()==Num_party-1,axis=0)
            pos_nv+=1
            N_V[pos_nv]=Num_votes
        f1=np.argmax(Num_votes)    

    elif VOTING_TYPE=='TRS':
        y_temp=y.copy()
        Num_votes=np.sum(np.argsort(y_temp).argsort()==Num_party-1,axis=0)
        arg_sort=np.argsort(Num_votes)
        y_temp[:,arg_sort[0:Num_party-2]]=0
        Num_votes=np.sum(np.argsort(y_temp).argsort()==Num_party-1,axis=0)
        f1=np.argmax(Num_votes)      
    
    return f1,Num_votes,N_V

# Function to compute the effort needed to change the election's outcome
# INPUT :x0 (Natural opinion, matrix of size [Number of agents, Number of parties])
#        Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$), matrix of size [Number of agents, Number of agents]
#        Y (Final outcome, matrix of size [Number of agents, Number of parties])
#        Num_votes (An array with number of votes per party)
#        wp (Next winner of election after the influence)
#        w0 (Influence strength vector in support of 'wp' party)
#        VOTING_TYPE: electoral system

# OUTPUT :E_min : sum of the influence vector
#         Num_agents_influenced: Number of agents influenced to change the election's outcome

def effort(Num_party,x0,Mat_inv,Y,wp,w0,VOTING_TYPE):
    x0_copy=x0.copy()
    trial=(np.argsort(x0).argsort()[:,wp]==Num_party-1)*1*np.arange(1,Num_ppl+1,1)-1
    indices=np.delete(trial,trial==-1) # Finding the positions were winnerto be is already the winner
    x0_copy[indices]+=100
    z=np.argsort(np.linalg.norm((1/Num_party)-x0_copy,axis=1))  # Agents to be influenced first
 
    VECTOR1=x0+(0.05*w0)
    VECTOR=np.zeros((Num_ppl,Num_party))+0.05*w0
    count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
    while count>0:
        w0=0.5*w0
        VECTOR1[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=x0[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]+w0
        VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=0.5*VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]
        count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
        
    iter=0  
    f1,Num_votes,N_V=winner(Y,VOTING_TYPE)
    f=1
    E_min=0
    Num_votes1=Num_votes.copy()
    while f==1 and iter<15:
        f=0
        b = 0
        while f1!=wp and b!=Num_ppl-1:
            T=np.zeros((Num_ppl,Num_party)) 
            for i in range(Num_party):
                T[:,i]=VECTOR[z[b],i]*Mat_inv[:,z[b]]
            Y=Y+T
            E_min+=np.sum(abs(VECTOR[z[b]]))
            f1,Num_votes1,N_V=winner(Y,VOTING_TYPE) 
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
    x0=data['x0'] #Natural opinion
    e=e_range[eps] #Confidence bound parameter
    
    #Constructing the adjacency matrix
    A=np.zeros((Num_ppl,Num_ppl))
    Mat_dist=np.zeros((Num_ppl,Num_ppl))
    for i in range(Num_ppl):
        Dist=np.linalg.norm(np.tile(x0[i],(Num_ppl,1))-x0,ord=construction_norm,axis=1)
        Mat_dist[i,:]=Dist
        A[i,:]=(Dist<=e)*1
    A=A-np.identity(Num_ppl)
    Mat_dist=np.multiply(A,Mat_dist)
    Max=np.argsort(x0).argsort()==Num_party-1
    
    # Finding the agents that belong to the party and outside
    INDICES=[] #Agents within the party
    INTER_INDICES=[] #Agents outside the party
    for num_party in range(Num_party):
        indices=Max[:,num_party]*np.arange(1,Num_ppl+1,1)-1
        indices=np.delete(indices,indices==-1)
        INDICES.append(indices)
        INTER_INDICES.append(np.setdiff1d(np.arange(0,Num_ppl,1),indices))
    
    # Fing the number of agents having more inter connections than intra connections
    MAX_NEIGHBOURS=np.zeros((Num_party,Num_party))
    INTER_NEIGHBOURS=np.zeros((Num_party,Num_party))
    for num_par_1 in range(Num_party): 
        Num_agents=np.zeros((Num_party))
        for num in range(len(INDICES[num_par_1])): # looping over the agents that belong to party num_par_1
            for num_par_2 in range(Num_party):
                Num_agents[num_par_2]=np.sum(A[INDICES[num_par_1][num],:][INDICES[num_par_2]])
            indices=np.setdiff1d(np.arange(0,Num_party,1),np.argmax(Num_agents))   
            if Num_agents[np.argmax(Num_agents)]>np.all(Num_agents[indices]):  # Determining the max neighbors for the agent initially in num_par_1
                MAX_NEIGHBOURS[num_par_1,np.argmax(Num_agents)]+=1 
            if Num_agents[np.argmax(Num_agents)]>Num_agents[num_par_1]: # Inter neighbors
                INTER_NEIGHBOURS[num_par_1,np.argmax(Num_agents)]+=1    
        
    # $Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where L, D, and \mathbb{I} is the laplacian, degree matrix and identity matrix resp.
    
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    y=np.matmul(Mat_inv,x0) 
    Y=y.copy()
    
    # Counting the number of agents that changed their opinion from one party to another
    AGENTS_TRANSITION=np.zeros((Num_party,Num_party))
    for num_par_1 in range(Num_party):
        for num in range(len(INDICES[num_par_1])):
            if np.argmax(x0[num])==np.argmax(y[num]):
                AGENTS_TRANSITION[num_par_1,num_par_1]+=1
            else:
                AGENTS_TRANSITION[num_par_1,np.argmax(y[num])]+=1
    
    #Normalising over the initial number of agents belongign to each party
    NORM_MAX_NEIGHBOURS=np.zeros((Num_party,Num_party))
    NORM_INTER_NEIGHBOURS=np.zeros((Num_party,Num_party))
    NORM_AGENTS_TRANSITION=np.zeros((Num_party,Num_party))
    for num in range(Num_party):
        NORM_AGENTS_TRANSITION[num]=AGENTS_TRANSITION[num]/len(INDICES[num])
        NORM_MAX_NEIGHBOURS[num]=MAX_NEIGHBOURS[num]/len(INDICES[num])
        NORM_INTER_NEIGHBOURS[num]=INTER_NEIGHBOURS[num]/len(INDICES[num])
    
    # Determing the party of each agent after interaction with other agents (at final state)
    PRIORITY_VOTES_PAR=np.zeros((Num_party,Num_party,Num_party))
    FINAL_INDICES=[] #Agents within the party
    FINAL_INTER_INDICES=[] #Agents outside the party
    Max=np.argsort(y).argsort()==Num_party-1
    for num_party in range(Num_party):
        indices=Max[:,num_party]*np.arange(1,Num_ppl+1,1)-1
        indices=np.delete(indices,indices==-1)
        FINAL_INDICES.append(indices)
        FINAL_INTER_INDICES.append(np.setdiff1d(np.arange(0,Num_ppl,1),indices))
        
    #Determing the ranking of each party
    for rank in range(Num_party):
        for num_par_1 in range(Num_party):
            PRIORITY_VOTES_PAR[rank,num_par_1]=np.sum(np.argsort(Y[FINAL_INDICES[num_par_1]]).argsort()==Num_party-rank-1,axis=0)
            
    np.savez('MULTIPARTY/Multipleparty_'+str(h)+'.npz', eps=eps, num=num, MAX_NEIGHBOURS=MAX_NEIGHBOURS,INTER_NEIGHBOURS=INTER_NEIGHBOURS,AGENTS_TRANSITION=AGENTS_TRANSITION,PRIORITY_VOTES_PAR=PRIORITY_VOTES_PAR,
             NORM_MAX_NEIGHBOURS=NORM_MAX_NEIGHBOURS,NORM_INTER_NEIGHBOURS=NORM_INTER_NEIGHBOURS,NORM_AGENTS_TRANSITION=NORM_AGENTS_TRANSITION)                       
    
    Plurality(h,x0, Mat_inv, y, 'Plurality')
    
    Ranked_choice_voting(h,x0, Mat_inv, y, 'RCV')
    
    Two_Round_system(h,x0,Mat_inv,y, 'TRS')
    
# Function to compute the effort needed to change the election outcome in favor of first runner-up
#INPUT: h (variable to determine the parameters used, i.e x0, \epsilon)
#       x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
#       Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where L, D, and \mathbb{I} is the laplacian, degree matrix and identity matrix resp.)
#       y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
#       VOTING_TYPE (type of electoral system to be employed, Plurality system)

#OUTPUT : output will be saved in a temporary file to be extracted later 
#       Effort (effort required to change the election outcome)
#       Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
#       NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)
def Plurality(h,x0,Mat_inv,y, VOTING_TYPE):
    
    win,Num_votes,N_V=winner(y, VOTING_TYPE) 
    
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    WParg=np.argsort(Num_votes)
    party=Num_party-2
    
    for j in range(1):  # Change the range to 'Num_party-1' to compute the effort needed to change the election's outcome for all the runner-ups 
        wp=int(WParg[party])
        Val_w0=STRG.Strategies(Num_party,wp)
        Effort[j],NUM_AGENTS_INFLUENCED[j]=effort(Num_party,x0,Mat_inv,y,wp, Val_w0, VOTING_TYPE)
        party=party-1
    
    np.savez('MULTIPARTY/Multipleparty_plurality'+str(h)+'.npz', Effort=Effort,Num_votes=Num_votes,NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED )                       

# Function to compute the effort needed to change the election outcome in favor of first runner-up
#INPUT: h (variable to determine the parameters used, i.e x0, \epsilon)
#       x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
#       Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where L, D, and \mathbb{I} is the laplacian, degree matrix and identity matrix resp.)
#       y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
#       VOTING_TYPE (type of electoral system to be employed, Ranked choice voting system system)

#OUTPUT : output will be saved in a temporary file to be extracted later 
#       Effort (effort required to change the election outcome)
#       Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
#       NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)
#       N_V (Number of votes after each elimination step)
def Ranked_choice_voting(h,x0,Mat_inv,y,VOTING_TYPE):
    
    win,Num_votes,N_V=winner(y, VOTING_TYPE)
    
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    WParg=np.argsort(Num_votes)
    party=Num_party-2
    
    #print(VOTING_TYPE,Num_votes,'Winner', int(WParg[party+1]), 'New winner', int(WParg[party]) )
    
    for j in range(1):   # Change the range to 'Num_party-1' to compute the effort needed to change the election's outcome for all the runner-ups 
        wp=int(WParg[party])
        Val_w0=STRG.Strategies(Num_party,wp)
        Effort[j],NUM_AGENTS_INFLUENCED[j]=effort(Num_party,x0,Mat_inv,y,wp,Val_w0,VOTING_TYPE)
        party=party-1
        
    np.savez('MULTIPARTY/Multipleparty_RCV'+str(h)+'.npz', Effort=Effort,Num_votes=Num_votes,NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED,N_V=N_V)                       

# Function to compute the effort needed to change the election outcome in favor of first runner-up
#INPUT: h (variable to determine the parameters used, i.e x0, \epsilon)
#       x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
#       Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where L, D, and \mathbb{I} is the laplacian, degree matrix and identity matrix resp.)
#       y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
#       VOTING_TYPE (type of electoral system to be employed, Two-Round system)

#OUTPUT : output will be saved in a temporary file to be extracted later 
#       Effort (effort required to change the election outcome)
#       Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
#       NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)

def Two_Round_system(h,x0,Mat_inv,y,VOTING_TYPE):
        
    win,Num_votes,N_V=winner(y, VOTING_TYPE)
    
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    WParg=np.argsort(Num_votes)
    party=Num_party-2 
   
    for j in range(1):   # Change the range to 'Num_party-1' to compute the effort needed to change the election's outcome for all the runner-ups 
        wp=int(WParg[party])
        Val_w0=STRG.Strategies(Num_party,wp)
        Effort[j],NUM_AGENTS_INFLUENCED[j]=effort(Num_party,x0,Mat_inv,y,wp, Val_w0, VOTING_TYPE)
        party=party-1
    
    #print(Num_votes_new, WParg, wp, Effort)
    np.savez('MULTIPARTY/Multipleparty_TRS'+str(h)+'.npz', Effort=Effort,Num_votes=Num_votes,NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED )                       


if __name__ == "__main__": 
    Num_simulation=1000 # Number of simulation
    Num_ppl=2001  # Number of people or agents
    neps=201 # Number of epsilon to be considered
    e_range=np.linspace(0,2,neps)  # Confidence bound range
    construction_norm=1  # Norm used to construct the network
    
    Par_start=3 # Party start
    Par_end=8 # Party end
    M = [(j, k) for j, k in itertools.product(range(0, Num_simulation, 1),range(0, neps, 1))]  
    
    #Creating a folder to save natural opinions
    os.mkdir('MULTIPARTY_NO')
    
    # 'for loop' to generate the natural opinion starting from Par_start to Par_end
    for Num_party in range(Par_start,Par_end):
        Natural_opinion()  

    
    for Num_party in range(Par_start,Par_end):  
        #Creating a folder to save intermediate results to be aggregated into a single file later
        os.mkdir('MULTIPARTY')
        
        # Parallelizing the computation of outcome of election and effort needed to change the election in favour of runner-up
        pool=mp.Pool(mp.cpu_count())
        pool.map(outcome,range(len(M)))
        pool.close() 
        
        #Aggregating the necessary data values into a matrix format
        MAX_NEIGHBOURS=np.zeros((Num_simulation,neps,Num_party,Num_party))
        INTER_NEIGHBOURS=np.zeros((Num_simulation,neps,Num_party,Num_party))
        AGENTS_TRANSITION=np.zeros((Num_simulation,neps,Num_party,Num_party))
        
        NORM_MAX_NEIGHBOURS=np.zeros((Num_simulation,neps,Num_party,Num_party))
        NORM_INTER_NEIGHBOURS=np.zeros((Num_simulation,neps,Num_party,Num_party))
        NORM_AGENTS_TRANSITION=np.zeros((Num_simulation,neps,Num_party,Num_party))
        
        PRIORITY_VOTES_PAR=np.zeros((Num_simulation,neps,Num_party,Num_party,Num_party))
        EFF_PLURALITY=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_VOTES_PLURALITY=np.zeros((Num_simulation,neps,Num_party))
        NUM_AGENTS_INFLUENCED_PLURALITY=np.zeros((Num_simulation,neps,Num_party-1))
        
        EFF_RCV=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_AGENTS_INFLUENCED_RCV=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_VOTES_RCV=np.zeros((Num_simulation,neps,Num_party))
        NUM_NV=np.zeros((Num_simulation,neps, Num_party-1, Num_party))
     
        EFF_TRS=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_AGENTS_INFLUENCED_TRS=np.zeros((Num_simulation,neps,Num_party-1))
        NUM_VOTES_TRS=np.zeros((Num_simulation,neps,Num_party))
        
        for h in range(Num_simulation*neps):
            num,eps=int(M[h][0]),int(M[h][1])
            
            data=np.load('MULTIPARTY/Multipleparty_'+str(h)+'.npz')
            
            MAX_NEIGHBOURS[num,eps]=data['MAX_NEIGHBOURS']
            INTER_NEIGHBOURS[num,eps]=data['INTER_NEIGHBOURS']
            AGENTS_TRANSITION[num,eps]=data['AGENTS_TRANSITION']
            PRIORITY_VOTES_PAR[num,eps]=data['PRIORITY_VOTES_PAR']
            
            NORM_MAX_NEIGHBOURS[num,eps]=data['NORM_MAX_NEIGHBOURS']
            NORM_INTER_NEIGHBOURS[num,eps]=data['NORM_INTER_NEIGHBOURS']
            NORM_AGENTS_TRANSITION[num,eps]=data['NORM_AGENTS_TRANSITION']
            
            data=np.load('MULTIPARTY/Multipleparty_plurality'+str(h)+'.npz')
            num,eps=int(M[h][0]),int(M[h][1])
            EFF_PLURALITY[num,eps]=data['Effort']
            NUM_AGENTS_INFLUENCED_PLURALITY[num,eps]=data['NUM_AGENTS_INFLUENCED']
            NUM_VOTES_PLURALITY[num,eps,:]=data['Num_votes']
            
            data=np.load('MULTIPARTY/Multipleparty_RCV'+str(h)+'.npz')
            EFF_RCV[num,eps]=data['Effort']
            NUM_AGENTS_INFLUENCED_RCV[num,eps]=data['NUM_AGENTS_INFLUENCED']
            NUM_VOTES_RCV[num,eps]=data['Num_votes']
            NUM_NV[num,eps]=data['N_V']
            
            data=np.load('MULTIPARTY/Multipleparty_TRS'+str(h)+'.npz')
            EFF_TRS[num,eps]=data['Effort']
            NUM_AGENTS_INFLUENCED_TRS[num,eps]=data['NUM_AGENTS_INFLUENCED']
            NUM_VOTES_TRS[num,eps]=data['Num_votes']
            
        #Saving into a '.npz' file in matrix format
        np.savez('EPS_CRITICAL/Ord_all_with_eq_area_'+str(Num_party)+'.npz',Num_simulation=Num_simulation,EFF_PLURALITY=EFF_PLURALITY,e_range=e_range,NUM_VOTES_PLURALITY=NUM_VOTES_PLURALITY,NUM_AGENTS_INFLUENCED_PLURALITY=NUM_AGENTS_INFLUENCED_PLURALITY,
        EFF_RCV=EFF_RCV, NUM_AGENTS_INFLUENCED_RCV=NUM_AGENTS_INFLUENCED_RCV, NUM_VOTES_RCV=NUM_VOTES_RCV, NUM_NV=NUM_NV, EFF_TRS=EFF_TRS, NUM_AGENTS_INFLUENCED_TRS=NUM_AGENTS_INFLUENCED_TRS, NUM_VOTES_TRS=NUM_VOTES_TRS,
        MAX_NEIGHBOURS=MAX_NEIGHBOURS,INTER_NEIGHBOURS=INTER_NEIGHBOURS,AGENTS_TRANSITION=AGENTS_TRANSITION, PRIORITY_VOTES_PAR=PRIORITY_VOTES_PAR,
        NORM_MAX_NEIGHBOURS=NORM_MAX_NEIGHBOURS,NORM_INTER_NEIGHBOURS=NORM_INTER_NEIGHBOURS,NORM_AGENTS_TRANSITION=NORM_AGENTS_TRANSITION)
        
        shutil.rmtree('MULTIPARTY')
   
    shutil.rmtree('MULTIPARTY_NO')   

stop = timeit.default_timer()   
print('Time:',(stop-start)/(60*60))  

