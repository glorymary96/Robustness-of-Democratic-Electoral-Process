#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:52:57 2023

@author: glory
"""
# In this code we have included simulations for bipartite system of election for 
# synthetic countries: SYNTHETIC_COUNTRIES()
# case study on US: CASE_STUDY_US()
# create a country: DEFINE_SYNTHETIC_COUNTRY()


# Robustness of electoral systems in a bi-partite system
# Three types of distribution for natural opinion (bi-gaussian with different parameters)
# Three different types of electoral systems (Single representatives (SR), Winner-takes-all representatives(WTAR) and Proportional representatives(PR))
 

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
import shutil
import pandas as pd

start = timeit.default_timer()

# Generating D1 distribution of natural opinion

def Nat_type1(R):
    y=[]
    for j in range(len(R)):
        flag=1
        flag_count=0
        while flag==1 and flag_count<100:
            r1=int(np.round(R[j]*0.01*NP_district))
            r2=int(NP_district-r1)
            y1=np.random.RandomState().normal(-bi_mean,sd,r1)
            while np.sum(y1<-1) or np.sum(y1>1):
                y1[y1<-1]=np.random.RandomState().normal(-bi_mean,sd,np.sum(y1<-1))
                y1[y1>1]=np.random.RandomState().normal(-bi_mean,sd,np.sum(y1>1))
            y2=np.random.normal(bi_mean,sd,r2)
            while np.sum(y2>1) or np.sum(y2<-1):
                y2[y2>1]=np.random.RandomState().normal(bi_mean,sd,np.sum(y2>1))
                y2[y2<-1]=np.random.RandomState().normal(bi_mean,sd,np.sum(y2<-1))
            x=np.concatenate((y1,y2))
            flag_count+=1
            if abs(R[j]-(np.sum(x<0)/NP_district)*100)<1:
                flag=0
        y=np.concatenate((y,x))         
    if len(R)%2==0:
        x=np.random.RandomState().normal(-bi_mean+bi_mean,sd,1)
        while x>1 or x<-1:
            x=np.random.RandomState().normal(-bi_mean+bi_mean,sd,1)
        y=np.concatenate((y,x))
    y=np.sort(y) 
    return y

# Generating D2 distribution of natural opinion
def Nat_type2(R,nop1):
    y=[]
    for j in range(len(R)):
        r1=int(np.round(R[j]*0.01*NP_district))
        y1=np.random.RandomState().normal(-bi_mean,sd,nop1)
        y2=np.random.RandomState().normal(bi_mean,sd,NP_district-nop1)
        x=np.concatenate((y1,y2))
        x=np.sort(x)
        if r1==0:
            delta=-x[r1]
        elif r1==NP_district:
            delta=-x[r1-1]
        else:               
            delta=-(x[r1-1]+x[r1])/2
        x=x+delta
        y=np.concatenate((y,x)) 
    if len(R)%2==0:
        x=np.random.RandomState().normal(mean,sd,1)
        y=np.concatenate((y,x))
    y=np.sort(y)    
    return y

# Generating D3 distribution of natural opinion
def Nat_type3(R):
    y=[]
    for j in range(len(R)):
        r1=int(np.round(R[j]*0.01*NP_district))
        x=np.random.RandomState().normal(mean,sd,NP_district)
        x=np.sort(x)
        if r1==0:
            delta=-x[r1]
        elif r1==NP_district:
            delta=-x[r1-1]
        else:               
            delta=-(x[r1-1]+x[r1])/2
        x=x+delta
        y=np.concatenate((y,x)) 
    if len(R)%2==0:
        x=np.random.RandomState().normal(mean,sd,1)
        y=np.concatenate((y,x))  
    y=np.sort(y)    
    return y   

# Generating natural opinions for one simulation using D1 distribution
def NAT1(nos):
    data=np.load('US/Republicans.npz',allow_pickle=True)
    REPUB,Num_states=data['REPUB'],data['Num_states']
    Repub=REPUB[nos]
    X=[]
    for num_state in range(Num_states):      
        y=Nat_type1(Repub[num_state])
        X.append(y)   
    np.savez('D_DATA/NO_'+str(nos)+'.npz',X=X)  

# Generating natural opinions for one simulation using D2 distribution
def NAT2(nos):
    data=np.load('US/Republicans.npz',allow_pickle=True)
    REPUB,Num_states=data['REPUB'],data['Num_states']
    Repub=REPUB[nos]
    X=[]
    for num_state in range(Num_states):
        y=Nat_type2(Repub[num_state], int(np.ceil(NP_district/2)))
        X.append(y) 
    np.savez('D_DATA/NO_'+str(nos)+'.npz',X=X) 

# Generating natural opinions for one simulation using D3 distribution
def NAT3(nos):
    data=np.load('US/Republicans.npz',allow_pickle=True)
    REPUB,Num_states=data['REPUB'],data['Num_states']
    Repub=REPUB[nos]
    X=[]
    for num_state in range(Num_states):
        y=Nat_type3(Repub[num_state])
        X.append(y) 
    np.savez('D_DATA/NO_'+str(nos)+'.npz',X=X) 

       
# Generating natural opinions in parallel
def natural_opinion(num_type):
    if num_type==0:
        pool=mp.Pool(mp.cpu_count())
        pool.map(NAT1,range(NOS))
        pool.close()
    elif num_type==1: 
        pool=mp.Pool(mp.cpu_count())
        pool.map(NAT2,range(NOS))
        pool.close()
    elif num_type==2:
        pool=mp.Pool(mp.cpu_count())
        pool.map(NAT3,range(NOS))
        pool.close()
        
#Determining the SR seats corresponding to the parties  
def single_representative(PN,Num_states):
    P1=np.sum(PN>=0)
    N1=Num_states-P1
    return P1,N1

#Determining the WTAR seats corresponding to the parties
def WTA_representative(PN,Num_districts_perstate,Num_districts):
    P1=np.sum((PN>=0)*Num_districts_perstate)
    N1=Num_districts-P1
    return P1,N1

#Determining the PR seats corresponding to the parties
def Prop_representative(P_ini,N_ini,Proportion,Num_districts):
    P1=np.sum((P_ini*Proportion).round(decimals=0))
    N1=Num_districts-P1
    res=[P1,N1]
    return res

#Determining the winner of the election   
def winner(P,N):
        if P>=N:
            f=1
        else:
            f=0
        return f 

#Determining the minimal effort necessary to change the election in each electoral unit
def minimum(x0,eps):
    e=e_range[eps]
    N1=len(x0)
    D=np.tile(x0,(len(x0),1))
    A=(abs(D-np.transpose(D))<e)*1-np.identity(len(x0))
    Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
    y=np.matmul(Mat_inv,x0)
    Pn = np.sum(y>=0)    
    Nn = N1-Pn
    
    flag=0
    if Nn>Pn:
       flag=1
       x0=-x0
       Pn,Nn=Nn,Pn
       y=-y
    
    z = np.argsort(x0 + 1e2 * (x0 < 0)*(-x0))
    iteration=0
    f=1
    p=Pn.copy()
    n=Nn.copy()
    E_min=0
    while f==1 and iteration<10:
        f=0
        b = 0
        while p > n and b!=N1-1:
            y=y+val_w0*Mat_inv[:,z[b]]
            E_min+=1
            p = np.sum(y>=0)
            n = N1-p               
            b=b+1
            if b==N1-1:
                f=1
                iteration=iteration+1   
    if flag==1:
        p,n=n,p
        
    return E_min,p,n

#Determining the election outcome (number of votes per party) in each electoral unit
def calc(h):
    num,eps=int(M[h][0]),int(M[h][1])
    data=np.load('D_DATA/NO_'+str(num)+'.npz',allow_pickle=True)
    X=np.array(data['X'])
    e=e_range[eps]
    Num_states=len(X)
    P_ini=np.zeros((Num_states))
    N_ini=np.zeros((Num_states))
    for g in range(Num_states):
        x0=np.array(X[g])
        D=np.tile(x0,(len(x0),1))
        A=(abs(D-np.transpose(D))<e)*1-np.identity(len(x0))
        Mat_inv=np.linalg.inv(np.matmul(np.diag(np.array([1/i if i!=0 else 0 for i in np.sum(A,axis=1)])),np.diag(np.sum(A,axis=1))-A)+np.identity(len(x0)))
        y=np.matmul(Mat_inv,np.array(X[g]))
        P_ini[g] = np.sum(y>=0)   
        N_ini[g] = len(x0)-P_ini[g]
        
    PN=P_ini-N_ini  
    np.savez('D_DATA/RDres_mid_'+str(h)+'.npz',P_ini=P_ini,N_ini=N_ini,X=X,PN=PN)

# Determining minimal effort needed to change the election's outcome following SR representative electoral system
def Single_representative(h):    
    eps=int(M[h][1])  
    data=np.load('D_DATA/RDres_mid_'+str(h)+'.npz',allow_pickle=True)    
    P_ini,N_ini,PN=data['P_ini'],data['N_ini'],data['PN']
    X=data['X']
    Num_states=len(X)
    
    z_minmaj=np.argsort(abs(P_ini-N_ini))
    P,N=single_representative(P_ini-N_ini,Num_states)
    if max(P,N)==P:
        s=np.where(P_ini-N_ini>0)[0]
    else:
        s=np.where(P_ini-N_ini<0)[0]
    s2=[]    
    for i in range(len(z_minmaj)):
          if z_minmaj[i] in s:
             s2.append(z_minmaj[i]) 
    
    Eff=np.zeros(len(s2))
    P_New_SR=P_ini.copy()
    N_New_SR=N_ini.copy()
    P,N=single_representative(PN,Num_states)
    f1=winner(P,N)
    
    for i in range(len(s2)):
        pos=int(s2[i])
        Eff[i],P_New_SR[pos],N_New_SR[pos]=minimum(X[pos],eps)
        P1,N1=single_representative(P_New_SR-N_New_SR,Num_states)
        f=winner(P1,N1)
        if f!=f1:
            break  
    Eff_SR=sum(Eff)  

    np.savez('DATA/RDres_mid_SR_'+str(h)+'.npz', Eff_SR=Eff_SR)

# Determining minimal effort needed to change the election's outcome following WTAR representative electoral system
def WTAR_representative(h): 
    data=np.load('US/Republicans.npz', allow_pickle=True)
    Num_districts_perstate=data['Num_districts_perstate']
    Num_districts=data['Num_districts']
    
    eps=int(M[h][1])  
    data=np.load('D_DATA/RDres_mid_'+str(h)+'.npz',allow_pickle=True)    
    P_ini,N_ini,PN=data['P_ini'],data['N_ini'],data['PN']
    X=data['X']
    
    z_minmaj=np.argsort(abs(P_ini-N_ini))
    P,N=WTA_representative(P_ini-N_ini,Num_districts_perstate,Num_districts)
    if max(P,N)==P:
        s=np.where(P_ini-N_ini>0)[0]
    else:
        s=np.where(P_ini-N_ini<0)[0]
    s2=[]    
    for i in range(len(z_minmaj)):
          if z_minmaj[i] in s:
             s2.append(z_minmaj[i]) 
    
    Eff=np.zeros(len(s2))
    P_New_WTA=P_ini.copy()
    N_New_WTA=N_ini.copy()
    P,N=WTA_representative(PN,Num_districts_perstate,Num_districts)
    f1=winner(P,N)
   
    for i in range(len(s2)):
        pos=int(s2[i]) 
        Eff[i],P_New_WTA[pos],N_New_WTA[pos]=minimum(X[pos],eps)
        P1,N1=WTA_representative(P_New_WTA-N_New_WTA,Num_districts_perstate,Num_districts)
        f=winner(P1,N1)
        if f!=f1:
            break
    Eff_WTAR=sum(Eff) 
    
    np.savez('DATA/RDres_mid_WTAR_'+str(h)+'.npz', Eff_WTAR=Eff_WTAR)

# Determining minimal effort needed to change the election's outcome following PR representative electoral system
def Proportional_representative(h):  
    data=np.load('US/Republicans.npz', allow_pickle=True)
    Proportion=data['Proportion']
    Num_districts=data['Num_districts']
    
    eps=int(M[h][1])
    data=np.load('D_DATA/RDres_mid_'+str(h)+'.npz',allow_pickle=True)   
    P_ini,N_ini=data['P_ini'], data['N_ini']
    X=data['X']   
    
    z_minmaj=np.argsort(abs(P_ini-N_ini))
    P,N=Prop_representative(P_ini,N_ini,Proportion,Num_districts)
    if max(P,N)==P:
        s=np.where(P_ini-N_ini>0)[0]
    else:
        s=np.where(P_ini-N_ini<0)[0]
    s2=[]    
    for i in range(len(z_minmaj)):
          if z_minmaj[i] in s:
             s2.append(z_minmaj[i]) 
   
    Eff=np.zeros(len(s2))
    P_New_PR=P_ini.copy()
    N_New_PR=N_ini.copy()
    P,N=Prop_representative(P_New_PR,N_New_PR,Proportion,Num_districts)
    
    f1=winner(P,N)
    
    for i in range(len(s2)):
        pos=int(s2[i])
        Eff[i],P_New_PR[pos],N_New_PR[pos]=minimum(X[pos],eps)
        P1,N1=Prop_representative(P_New_PR,N_New_PR,Proportion,Num_districts)
        f=winner(P1,N1)
        if f!=f1:
            break
    Eff_PR=sum(Eff)
    np.savez('DATA/RDres_mid_PR_'+str(h)+'.npz', Eff_PR=Eff_PR)

#Parallel computing the election's outcome and minimum effort needed to change the election's outcome             
def ELECTIONS(N_S, num_type,Num_states, Num_districts, Proportion, REPUB): 
    
    pool=mp.Pool(mp.cpu_count())
    pool.map(calc,range(len(M)))
    pool.close()
   
    pool=mp.Pool(mp.cpu_count())
    pool.map(Proportional_representative,range(len(M)))
    pool.close()
    
    pool=mp.Pool(mp.cpu_count())
    pool.map(Single_representative,range(len(M)))
    pool.close()
   
    pool=mp.Pool(mp.cpu_count())
    pool.map(WTAR_representative,range(len(M)))
    pool.close()
   
    P_INI=np.zeros((NOS,neps,Num_states))
    N_INI=np.zeros((NOS,neps,Num_states))
    
    E_SR=np.zeros((NOS,neps))
    E_WTAR=np.zeros((NOS,neps))
    E_PR=np.zeros((NOS,neps))
   
    for h in range(NOS*neps):
        data=np.load('D_DATA/RDres_mid_'+str(h)+'.npz')
        num,eps=int(M[h][0]),int(M[h][1])
        P_INI[num,eps,:]=data['P_ini']
        N_INI[num,eps,:]=data['N_ini']
           
        data=np.load('DATA/RDres_mid_SR_'+str(h)+'.npz')
        E_SR[num,eps]=data['Eff_SR']
        
        data=np.load('DATA/RDres_mid_WTAR_'+str(h)+'.npz')
        E_WTAR[num,eps]=data['Eff_WTAR']
       
        data=np.load('DATA/RDres_mid_PR_'+str(h)+'.npz')
        E_PR[num,eps]=data['Eff_PR']
       
    np.savez('US/RealDout_'+ str(num_type+1) +'_robustness_'+str(N_S)+'.npz', P_INI=P_INI,N_INI=N_INI,E_SR=E_SR,E_WTAR=E_WTAR,E_PR=E_PR)

# Generating average percentage of votes per district, with the maximum difference between them two parties as 10%
def REPUB_CALC(Num_states, Num_districts_perstate):
    REPUB=[]
    for nos in range(NOS): 
        Repub=[]
        for num_state in range(Num_states):
            Repub.append(np.random.uniform(low=45,high=55, size=Num_districts_perstate[num_state]))
        REPUB.append(Repub)  
    return REPUB    

def SYNTHETIC_COUNTRIES():
    #Loading the specification of synthetic countries (i.e number of states and seats distribution among the states)
    data=np.load('Multi_States/Initial_considerations.npz',allow_pickle=True)
    NUM_STATES=data['NUM_STATES']
    NUM_SEATS_PER_STATE=data['NUM_SEATS_PER_STATE']
    
    for N_S in range(len(NUM_STATES)):
        Num_states=NUM_STATES[N_S] #Number of states in the synthetic country
        Num_districts_perstate=NUM_SEATS_PER_STATE[N_S] #Seat distribution in the states
        Num_districts=np.sum(Num_districts_perstate).astype('int') #Total number of seats/districts
        N=Num_districts_perstate*NP_district
        N[N%2==0]+=1 #Number of agents in each state determined by the number of seats allotted to the corresponding state
        Proportion=(Num_districts_perstate/N)  
        
        # Generating average percentage of votes per district 
        REPUB=REPUB_CALC(Num_states, Num_districts_perstate)
        
        np.savez('US/Republicans.npz', REPUB=REPUB,Num_states=Num_states, Num_districts_perstate=Num_districts_perstate,Num_districts=Num_districts,N=N, Proportion=Proportion)     
        
        # Source code in order to generate the natural opinion and for simulations corresponding to election's outcome and effort to change the election's outcome
        os.mkdir('D_DATA')
        for num_type in range(Num_type):
            natural_opinion(num_type) 
            
            os.mkdir('DATA')
            ELECTIONS(N_S, num_type,Num_states, Num_districts, Proportion, REPUB)   # Elections in states
            shutil.rmtree('DATA')
               
        shutil.rmtree('D_DATA')  

def CASE_STUDY_US():
    
    # Loading the number of districts in each state and computing the number of agents in each state accordingly
    df_state_district_distribution = pd.read_csv("HOR_DATA/state_district_distribution.csv")
    Num_districts_perstate=np.array(df_state_district_distribution['Num_districts'],dtype=int)
    N=Num_districts_perstate*NP_district
    N[N%2==0]+=1
    Proportion=(Num_districts_perstate/N)  
    Num_states=len(Num_districts_perstate)
    Num_districts=np.sum(Num_districts_perstate)
    
    # Source code inorder to generate the natural opinion following the election results in the year 2012-2020
    for year in range(2012,2022,2):
        df=pd.read_csv("HOR_DATA/Election"+ str(year)+".csv", converters={'ID': str, 'CD_NUM': str, 'STATE_FP': str})
        df_state_district_distribution = pd.read_csv("HOR_DATA/state_district_distribution.csv")     
        Repub=[]  # For election with different outcomes depending on state distribution
        i,j=0,0
        for dist in df_state_district_distribution['Num_districts']:
            num_district=int(dist)
            Num_districts_perstate[j]=num_district
            arr=np.zeros((num_district))
            for dists in range(num_district):
                arr[dists]=df['Republican'][i]
                i=i+1
            j=j+1
            Repub.append(arr)
        
        #Source code for generating natural opinion and simulations corresponding to election's outcome and effort to change the election's outcome   
        REPUB=[]
        for nos in range(NOS):
            REPUB.append(Repub)
            
        np.savez('US/Republicans.npz', REPUB=REPUB,Num_states=Num_states, Num_districts_perstate=Num_districts_perstate,Num_districts=Num_districts,N=N, Proportion=Proportion)     
        
        os.mkdir('D_DATA')  
        for num_type in range(Num_type):
            natural_opinion(num_type)  
            os.mkdir('DATA')
            ELECTIONS(year, num_type,Num_states, Num_districts, Proportion, REPUB)   # Elections in states
            shutil.rmtree('DATA') 
            
        shutil.rmtree('D_DATA')   
        
def DEFINE_SYNTHETIC_COUNTRY():
    
    N_S= 0
    Num_states=int(input('Enter the number of states: '))
    Num_districts_perstate=np.zeros((Num_states),dtype=int)
    for num_state in range(Num_states):
        Num_districts_perstate[num_state]=int(input('Enter number of seats in district '+str(num_state+1)+': '))
    NP_district=int(input('Enter the number of agents corresponding to each seat: '))
    N=Num_districts_perstate*NP_district
    N[N%2==0]+=1
    Proportion=(Num_districts_perstate/N)  
    Num_states=len(Num_districts_perstate)
    Num_districts=np.sum(Num_districts_perstate)
    
    # Generating average percentage of votes per district 
    REPUB=REPUB_CALC(Num_states, Num_districts_perstate)
    
    np.savez('US/Republicans.npz', REPUB=REPUB,Num_states=Num_states, Num_districts_perstate=Num_districts_perstate,Num_districts=Num_districts,N=N, Proportion=Proportion)     
    
    # Source code in order to generate the natural opinion and for simulations corresponding to election's outcome and effort to change the election's outcome
    os.mkdir('D_DATA')
    for num_type in range(Num_type):
        natural_opinion(num_type) 
        
        os.mkdir('DATA')
        ELECTIONS(N_S, num_type,Num_states, Num_districts, Proportion, REPUB)   # Elections in states
        shutil.rmtree('DATA')
           
    shutil.rmtree('D_DATA')  



if __name__ == "__main__":
    
    NOS=100 # Number of simulations   
    neps=51
    e_range=np.linspace(0,1.5,neps).round(decimals=3) # confidence bound parameter range
    e_num=np.arange(0,neps,1)
    val_w0=-0.1  # Influence strength 
    bi_mean=0.25 # Mean value of the gaussian distributions
    mean=0 # Total mean
    sd=0.2  #Standard deviation
    M=[]
    NP_district=101  # Number of people in each district
    Num_party=2
    Num_type=3
    
    M = [(j, k) for j, k in itertools.product(range(0, NOS, 1),range(0, neps, 1))] 
    
    # SYNTHETIC COUNTRIES
    # Robustness of electoral system to external attack for 15 synthetic countries with [3,15] seats with an average of 9 seats per state.
    # The maximum difference in votes between two parties is 10%.

    SYNTHETIC_COUNTRIES()
    
    # CASE STUDY WITH HISTORIC DATA
    # Robustness of electoral system to external attack for US House of Representative elections
    # The elections results from the US historic data is stored in the folder 'HOR_DATA' with names 'Election[year].csv'

    CASE_STUDY_US()
    
    np.savez('US/Assumptions.npz', Number_of_simulation=NOS, e_range=e_range, Influence_strength=val_w0, Delta=2*bi_mean, mean=mean, std_dev=sd, NP_district = NP_district)
    
    
      
stop = timeit.default_timer()
print('Time: ', (stop - start)/(60*60), 'hours')      
