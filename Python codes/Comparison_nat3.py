#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:14:25 2021

@author: glory
"""

# Minimum majority vs Minimum people  vs As a single unit for different parameter values of bigaussian distribution
# We consider a synthetic country with 7 states, with varying population in each state

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import numpy as np
import multiprocessing as mp
import timeit
import itertools
import math
import random
import os
import shutil

start = timeit.default_timer()

#Generating natural opinion with different parameter values
def Natural_opinion(num_type, mean1,mean2):
    if num_type==0:
        for nos in range(NOS):
            N=np.array(random.sample(range(400,500), Num_states)) #Number of agents in each state
            N[N%2==0]+=1
            for m in range(ndelta):                
                X=[] #Natural opinions
                for g in range(Num_states):
                    Nn=math.floor(N[g]/2)  # Number of people with negative opinion
                    Np=N[g]-Nn             # Number of people with positive opinion
                    y1=np.random.normal(mean1[m],sd,Nn)
                    y2=np.random.normal(mean2[m],sd,Np)
                    x0=np.concatenate((y1,y2))
                    X.append(x0)
                    np.savez('DATA/Comp_bigauss_'+str(nos)+'_'+str(m)+'.npz',X=X)                  
    elif num_type==1:
        mean=0.25
        for nos in range(NOS):
            N=np.array(random.sample(range(400,500), Num_states))  #Number of agents in each state
            N[N%2==0]+=1
            for m in range(num_negative_bias):
                X=[]
                for j in range(Num_states):
                    Nn=int(negative_votes[m]*N[j])  # Number of people with negative opinion
                    Np=N[j]-Nn       # Number of people with positive opinion
                    y1=np.random.normal(-mean,sd,Nn)
                    y2=np.random.normal(mean,sd,Np)
                    x0=np.concatenate((y1,y2))
                    X.append(x0)
                np.savez('DATA/Comp_bigauss_'+str(nos)+'_'+str(m)+'.npz',X=X)    

# Minimal effort needed to change the election's outcome in an electoral unit                
def minimum(x0,e):
    N1=len(x0)
    D=np.tile(x0,(len(x0),1))
    A=(abs(D-np.transpose(D))<e)*1-np.identity(len(x0)) 
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    y=np.matmul(Mat_inv,x0)
    Pn = np.sum(y>0)    
    Nn = N1-Pn

    if Nn>Pn:
       x0=-x0
       Pn,Nn=Nn,Pn
       y=-y
    
    z = np.argsort(x0 + 1e2 * (x0 < 0)*(-x0))
    iter=0
    f=1
    p=Pn.copy()
    n=Nn.copy()
    while f==1 and iter<10:
        f=0
        b = 0
        while p > n and b!=N1-1:
            y=y+val_w0*Mat_inv[:,z[b]]
            p = np.sum(y > 0)
            n = N1-p               
            b=b+1
            if b==N1-1:
                f=1
                iter=iter+1   
    E_min = (iter*len(x0))+b  

    res= [Pn,Nn,E_min]
    return res  
              
#Finding the election's outcome and determining the effort needed to change the election's outcome following minimum majority and minimum people strategies.     
def calc(h):
    delt,num,eps=int(M[h][0]),int(M[h][1]),int(M[h][2])
    data=np.load('DATA/Comp_bigauss_'+str(num)+'_'+str(delt)+'.npz',allow_pickle=True)
    X=data['X']
    N=np.zeros((Num_states))
    e=e_range[eps]
    P_ini=np.zeros((Num_states))
    N_ini=np.zeros((Num_states))
    PN=np.zeros((Num_states))
    
    for g in range(Num_states):
        x0=np.array(X[g])
        N[g]=len(x0)
        D=np.tile(x0,(len(x0),1))
        A=(abs(D-np.transpose(D))<e)*1-np.identity(len(x0)) 
        Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
        y=np.matmul(Mat_inv,x0)
        P_ini[g] = np.sum(y>0)    
        N_ini[g] = N[g]-P_ini[g]     
        PN[g] = P_ini[g]-N_ini[g]
                     
    z_minppl=np.argsort(N) 
    z_minmaj=np.argsort(abs(PN)) 
    pp = np.sum(PN>0)
    nn = np.sum(PN<0)
    maximum=max(pp,nn)
    if maximum == pp:
        s=np.where(PN>0)[0]
    else:
        s=np.where(PN<0)[0]
  
# Influencing the states with minimum majority of change in outcome  
    s1=[]
    for i in range(len(z_minppl)):
          if z_minppl[i] in s:
             s1.append(z_minppl[i])
             
    nrti=Num_states-len(s)         
    NRTI=math.ceil(abs(nrti-len(s))/2) # Number of states to be influenced
    Eff=np.zeros((int(NRTI)))
    for i in range(len(Eff)):
        R=minimum(X[s1[i]],e)
        Eff[i]=R[2]
    Effort=sum(Eff) 
    PEffort=(Effort/np.sum(N))*100
    
# Influencing the states with minimum majority of change in outcome                      
    s2=[]
    for i in range(len(z_minmaj)):
          if z_minmaj[i] in s:
             s2.append(z_minmaj[i])
             
    nrti=Num_states-len(s)         
    NRTI=math.ceil(abs(nrti-len(s))/2) # Number of states to be influenced   
    Eff=np.zeros((int(NRTI)))
    for i in range(len(Eff)):
        R1=minimum(X[s2[i]],e)
        Eff[i]=R1[2]
    Effort1=sum(Eff)  
    PEffort1=(Effort1/np.sum(N))*100
    

    np.savez('DATA/Comp_bigauss_res_'+str(h)+'.npz', delt=delt, num=num,eps=eps,P_ini=P_ini,N_ini=N_ini,PN=PN,Eff_MM=Effort1,Eff_MP=Effort,PEff_MM=PEffort1, PEff_MP=PEffort)

#Effort needed to change the election's outcome given that the election was conducted as a single unit instead of multiple unit
def AS_SINGLE_ELECTION(h):
    row=M[h]
    delt,num,eps=row 
    delt,num,eps=int(delt),int(num),int(eps)
    e=e_range[eps]
    data=np.load('DATA/Comp_bigauss_'+str(num)+'_'+str(delt)+'.npz',allow_pickle=True)
    
    X=data['X']
    X=np.array(X)
    x0=[]
    for i in range(Num_states):
        x0=np.concatenate((x0,X[i]))
        
    D=np.tile(x0,(len(x0),1))
    A=(abs(D-np.transpose(D))<e)*1-np.identity(len(x0)) 
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    y=np.matmul(Mat_inv,x0)
    Pn = np.sum(y>=0)    
    Nn = len(x0)-Pn
    
    flag=0
    if Nn>Pn:
       flag=1
       x0=-x0
       Pn,Nn=Nn,Pn
       y=-y
    
    z = np.argsort(x0 + 1e2 * (x0 < 0)*(-x0))
    iter=0
    f=1
    p=Pn.copy()
    n=Nn.copy()
    while f==1 and iter<10:
        f=0
        b = 0
        while p > n and b!=len(x0)-1:
            y=y+val_w0*Mat_inv[:,z[b]]
            p = np.sum(y>=0)
            n = len(x0)-p               
            b=b+1
            if b==len(x0)-1:
                f=1
                iter=iter+1   
    E_min = (iter*len(x0))+b  
    N_Emin=(E_min/len(x0))*100
    if flag==1:
        p,n=n,p
    
    np.savez('DATA/Comp_bigauss_res_single_'+str(h)+'.npz', delt=delt, num=num,eps=eps,P_ini=Pn,N_ini=Nn,Effort=E_min, N_effort=N_Emin)    

#For different values of mean and polarization
def Polarization_with_shift():
    num_type=0
    for shift in range(len(Shift)):
        os.mkdir('DATA')
        mean1=Shift[shift]-delta
        mean2=Shift[shift]+delta
        
        Natural_opinion(num_type,mean1,mean2)  # Natural opinion 
        
        pool=mp.Pool(mp.cpu_count())
        pool.map(calc,range(len(M)))
        pool.close()  
        
        P_INI=np.zeros((ndelta,NOS,neps))
        N_INI=np.zeros((ndelta,NOS,neps))
        E_MM=np.zeros((ndelta,NOS,neps))
        E_MP=np.zeros((ndelta,NOS,neps))
        PE_MM=np.zeros((ndelta,NOS,neps))
        PE_MP=np.zeros((ndelta,NOS,neps))
              
        for h in range(ndelta*NOS*neps):
            data=np.load('DATA/Comp_bigauss_res_'+str(h)+'.npz')
            delt=data['delt']
            num=data['num']
            eps=data['eps']
            P_ini=data['P_ini']
            N_ini=data['N_ini']
            Eff_MM=data['Eff_MM']
            Eff_MP=data['Eff_MP']
            P_INI[delt,num,eps]=np.sum(P_ini)
            N_INI[delt,num,eps]=np.sum(N_ini)
            E_MM[delt,num,eps]=Eff_MM
            E_MP[delt,num,eps]=Eff_MP
            PE_MM[delt,num,eps]=data['PEff_MM']
            PE_MP[delt,num,eps]=data['PEff_MP']
            
        np.savez('OD1/Comp_polarization_'+str(Shift[shift])+'.npz',e_range=e_range,delta=delta, P_INI=P_INI,N_INI=N_INI,E_MM=E_MM,E_MP=E_MP, PE_MM=PE_MM, PE_MP=PE_MP)
        
        
        pool=mp.Pool(mp.cpu_count())
        pool.map(AS_SINGLE_ELECTION,range(len(M)))
        pool.close()  
        
        P_INI=np.zeros((ndelta,NOS,neps))
        N_INI=np.zeros((ndelta,NOS,neps))
        E_Single=np.zeros((ndelta,NOS,neps))
        P_Esingle=np.zeros((ndelta,NOS,neps))
           
        for h in range(ndelta*NOS*neps):
            data=np.load('DATA/Comp_bigauss_res_single_'+str(h)+'.npz')
            delt=data['delt']
            num=data['num']
            eps=data['eps']
            P_INI[delt,num,eps]=data['P_ini']
            N_INI[delt,num,eps]=data['N_ini']
            E_Single[delt,num,eps]=data['Effort']
            P_Esingle[delt,num,eps]=data['N_effort']
            
        np.savez('OD1/Comp_polarization_single_'+str(Shift[shift])+'.npz',e_range=e_range,delta=delta, P_INI=P_INI,N_INI=N_INI,E_Single=E_Single,P_ESingle=P_Esingle)
        
        shutil.rmtree('DATA')

# For different values of 'p', changing the weight of gaussian 
def Negative_votes_bias():  
    num_type=1
    mean=0.25
    os.mkdir('DATA') 
    Natural_opinion(num_type,mean,mean)  # Natural opinion 
    
    pool=mp.Pool(mp.cpu_count())
    pool.map(calc,range(len(M)))
    pool.close()  #Effort needed to change the election's outcome given that the election was conducted as a single unit instead of multiple unit


    P_INI=np.zeros((num_negative_bias,NOS,neps))
    N_INI=np.zeros((num_negative_bias,NOS,neps))
    E_MM=np.zeros((num_negative_bias,NOS,neps))
    E_MP=np.zeros((num_negative_bias,NOS,neps))
    PE_MM=np.zeros((num_negative_bias,NOS,neps))
    PE_MP=np.zeros((num_negative_bias,NOS,neps))
        
    for h in range(num_negative_bias*NOS*neps):
        data=np.load('DATA/Comp_bigauss_res_'+str(h)+'.npz')
        delt=data['delt']
        num=data['num']
        eps=data['eps']
        P_ini=data['P_ini']
        N_ini=data['N_ini']
        Eff_MM=data['Eff_MM']
        Eff_MP=data['Eff_MP']
        P_INI[delt,num,eps]=np.sum(P_ini)
        N_INI[delt,num,eps]=np.sum(N_ini)
        E_MM[delt,num,eps]=Eff_MM
        E_MP[delt,num,eps]=Eff_MP
        PE_MM[delt,num,eps]=data['PEff_MM']
        PE_MP[delt,num,eps]=data['PEff_MP']
            
    np.savez('OD1/Comp_proportion.npz',e_range=e_range,negative_votes_prop=negative_votes, P_INI=P_INI,N_INI=N_INI,E_MM=E_MM,E_MP=E_MP,PE_MM=PE_MM, PE_MP=PE_MP)
    
    pool=mp.Pool(mp.cpu_count())
    pool.map(AS_SINGLE_ELECTION,range(len(M)))
    pool.close()  
    
    P_INI=np.zeros((num_negative_bias,NOS,neps))
    N_INI=np.zeros((num_negative_bias,NOS,neps))
    E_Single=np.zeros((num_negative_bias,NOS,neps))
    P_Esingle=np.zeros((num_negative_bias,NOS,neps))
    
    for h in range(num_negative_bias*NOS*neps):
        data=np.load('DATA/Comp_bigauss_res_single_'+str(h)+'.npz')
        delt=data['delt']
        num=data['num']
        eps=data['eps']
        P_INI[delt,num,eps]=data['P_ini']
        N_INI[delt,num,eps]=data['N_ini']
        E_Single[delt,num,eps]=data['Effort']
        P_Esingle[delt,num,eps]=data['N_effort']
    np.savez('OD1/Comp_proportion_single.npz',e_range=e_range,negative_votes_prop=negative_votes, P_INI=P_INI,N_INI=N_INI,E_Single=E_Single, P_Esingle=P_Esingle)
    
    shutil.rmtree('DATA')
      
if __name__ == "__main__":
    NOS=500 # Number of simulations    
    Num_states=7      # Number of states
    neps=51
    e_range=np.linspace(0,1.5,neps) # Range of confidence bound
    e_num=np.arange(0,neps,1)
    val_w0=-0.1
    
    ndelta=3
    delta_start=0
    delta_fin=0.3
    delta=np.linspace(delta_start,delta_fin,ndelta) # Polarization parameter values
    
    num_negative_bias=3
    negative_bias_start=0.5
    negative_bias_fin=0.524
    negative_votes=np.linspace(negative_bias_start,negative_bias_fin,num_negative_bias) # Different values of 'p'
    
    sd=0.2
    M=[]
    Shift=np.linspace(0,0.05,2)
    M = [(i, j, k) for i, j, k in itertools.product(range(0,ndelta,1),range(0, NOS, 1),range(0, neps, 1))]    
    Polarization_with_shift() # Source code for different polarization values with varying mean values
    
    M=[]
    M = [(i, j, k) for i, j, k in itertools.product(range(0,num_negative_bias,1),range(0, NOS, 1),range(0, neps, 1))]    
    Negative_votes_bias() # Source code for different values of 'p'
    
    
stop = timeit.default_timer()
print('Time: ', (stop - start)/(60*60), 'hours')  
