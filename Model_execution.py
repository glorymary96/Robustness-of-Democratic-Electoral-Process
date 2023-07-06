#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:00:05 2023

@author: glory
"""

import Nat_opn_generator as OPN_GEN

import numpy as np

def type_natural_opinion(Num_party, NAT_TYPE = 'Uniform'):
    if Num_party == 2 and NAT_TYPE == 'D1':
        x0=OPN_GEN.Nat_opn_D1(bi_mean, sd, Num_ppl, R=50) # Change the value of R to impose the bias
    elif Num_party == 2 and NAT_TYPE == 'D2':
        x0=OPN_GEN.Nat_opn_D2(bi_mean, sd, Num_ppl, R=50)  # Change the value of R to impose the bias
    elif Num_party == 2 and NAT_TYPE == 'D3':
        x0=OPN_GEN.Nat_opn_D3(mean, sd, Num_ppl, R=50)  # Change the value of R to impose the bias
    else:
         x0=OPN_GEN.Random_generation(Num_ppl, Num_party) #Generating natural opinion by uniform distribution
    return x0
        
        

def outcome(Num_ppl, Num_party, NAT_TYPE, epsilon,W):
    
    x0=type_natural_opinion(Num_party,NAT_TYPE)
    
    A=np.zeros((Num_ppl,Num_ppl))
    for i in range(Num_ppl):
        A[i,:]=(np.linalg.norm(np.tile(x0[i],(Num_ppl,1))-x0,ord=construction_norm,axis=1)<=epsilon)*1
    A=A-np.identity(Num_ppl) # Adjaceny Matrix
   
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    
    y=np.matmul(Mat_inv,x0+W) 
    
    return x0,y
 
if __name__ == "__main__": 
    
    construction_norm=1
    bi_mean=0.25
    mean=0
    sd=0.2 #Standard deviation
    
    Num_ppl= int(input("Enter number of agents: "))
    epsilon = float(input("Enter confidence bound value between 0 to 2 : "))
    
    while epsilon>2 or epsilon<0:
        epsilon = float(input("Invalid entry! Enter confidence bound value between 0 to 2 : "))
    
    Num_party=int(input("Enter number of parties: "))
    if Num_party==2:
        NAT_TYPE=input("Enter the natural opinion type 'D1', 'D2', 'D3', or 'Uniform' :")
        while NAT_TYPE!='D1' and NAT_TYPE!='D2' and NAT_TYPE!='D3' and NAT_TYPE!='Uniform':
            NAT_TYPE=input("Invalid type! Enter the natural opinion type 'D1', 'D2', 'D3', 'Uniform' :") 
         
    else:
        NAT_TYPE='Uniform'
    
    W=np.zeros((Num_ppl,Num_party)) # External influence, change the components of W to add external influence
    
    x0,y=outcome(Num_ppl, Num_party, NAT_TYPE, epsilon,W) # Computing the outcome of the election
    
    Num_votes=np.sum(np.argsort(x0).argsort()==Num_party-1,axis=0)
    print('Number of votes acc. to natural opinion:', Num_votes)
    
    Num_votes_final=np.sum(np.argsort(y).argsort()==Num_party-1,axis=0)
    print('Number of votes acc. to final opinion:', Num_votes_final)
    
        
    
