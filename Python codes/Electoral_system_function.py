#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:40:55 2023

@author: glory
"""

import numpy as np

import Strategies_w0 as STRG

#Function to determine the winner
def winner(Num_votes): 
    f1=np.argmax(Num_votes)
    return f1     
 
#Function to compute the effort needed to change the election's outcome
def effort(Num_party,x0,Mat_inv,Y,Num_votes,wp,w0):
    x0_copy=x0.copy()
    trial=(np.argsort(x0).argsort()[:,wp]==Num_party-1)*1*np.arange(1,len(x0)+1,1)-1
    indices=np.delete(trial,trial==-1)
    x0_copy[indices]+=100
    c=1/Num_party
    z=np.argsort(np.linalg.norm(c-x0_copy,axis=1))
    VECTOR1=x0+(0.1*w0)
    VECTOR=np.zeros((len(x0),Num_party))+0.1*w0
    count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
    w0_count=0
    while count>0 and w0_count<10:
        w0=0.5*w0
        VECTOR1[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=x0[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]+w0
        VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]=0.5*VECTOR[np.where(np.sum(VECTOR1<0,axis=1)>0)[0]]
        count=len(np.where(np.sum(VECTOR1<0,axis=1)>0)[0])
        w0_count+=1
    
    iter=0    
    f1=winner(Num_votes)
    f=1
    E_min=0
    while f==1 and iter<15:
        f=0
        b = 0
        while f1!=wp and b!=len(x0)-1:
            T=np.zeros((len(x0),Num_party)) 
            for i in range(Num_party):
                T[:,i]=VECTOR[z[b],i]*Mat_inv[:,z[b]]
            Y=Y+T
            E_min+=np.sum(abs(VECTOR[z[b]]))
            Num_votes1=np.sum((np.argsort(Y).argsort()==Num_party-1)*1,axis=0)
            f1=winner(Num_votes1) 
            b=b+1
            if b==len(x0)-1:
                f=1
                iter=iter+1
    Num_agents_influenced = (iter*len(x0))+b    
    
    return [Num_agents_influenced,E_min,Num_votes1,Y]

#Function to determining the states to be influenced first following minimum majority strategy
def States_to_be_influenced(Num_party, Num_states, NUM_VOTES,party,winner_to_be):
    args=np.setdiff1d(np.arange(0,Num_party,1),winner_to_be) # Parties excluding the winner_to_be
    Arr=np.zeros((Num_party-1,Num_states))
    
    #Difference of votes between the winner_to_be and other parties and remove the states where winner_t_be is already the winner
    for i in range(Num_party-1):
        Arr[i]=NUM_VOTES[:,winner_to_be]-NUM_VOTES[:,args[i]]
        Arr[i][Arr[i]>0]=0 
    
    #Finding the states with minimum difference in majority
    MIN=Arr[0]
    for j in range(1,Num_party-1):
        for num_state in range(Num_states):
            if Arr[j,num_state]==0 or MIN[num_state]==0:
                MIN[num_state]=min(MIN[num_state],Arr[j,num_state])
            else:
                MIN[num_state]=max(MIN[num_state],Arr[j,num_state])
    num_zeros=np.sum(MIN==0)    
    # Ordering in descending order and removing the states where winner_to_be is already the winner
    Arg_Sorted_Array=np.argsort(MIN)[::-1]    
    States_to_influence=np.delete(Arg_Sorted_Array,np.arange(0,num_zeros,1))
    
    return States_to_influence

#Function to determine the effort needed to change the election's outcome following a single representative system of election
def Single_representative(h,MAT_INV):
    data=np.load('MULTIPARTY_RES/Inital_res_'+str(h)+'_.npz', allow_pickle=True)
    Num_party=data['Num_party']
    Num_states=data['Num_states']
    X0=data['X0']
    Result=data['Result']
    NUM_VOTES=data['NUM_VOTES']
    
    NUM_SEATS=np.zeros((Num_states, Num_party))
    for i in range(Num_states):
        NUM_SEATS[i,np.argmax(NUM_VOTES[i])]=1
    Num_states_won=np.sum((np.argsort(NUM_VOTES).argsort()==Num_party-1)*1,axis=0)
    #Election_winner=winner(Num_states_won)
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    Tot_NUM_VOTES_INF=np.zeros((Num_party-1,Num_states,Num_party))
    Tot_NUM_SEATS_INF=np.zeros((Num_party-1,Num_states,Num_party))
    INF_STATES=[]
    
    party=Num_party-2
    for num_party in range(1):
        NUM_VOTES_INF=NUM_VOTES.copy()
        winner_to_be=np.argsort(Num_states_won)[party]
        States_to_influence=States_to_be_influenced(Num_party,Num_states, NUM_VOTES,party,winner_to_be)
        E=[]
        INFLUENCED_STATES=[]
        for state in range(len(States_to_influence)):
            state_num=States_to_influence[state]
            w0=STRG.Strategies(Num_party,winner_to_be)
            Num_agents_inf,Eff,Num_votes_inf,Y=effort(Num_party,X0[state_num],MAT_INV[state_num], Result[state_num],NUM_VOTES[state_num], winner_to_be,w0)
            NUM_AGENTS_INFLUENCED[num_party]+=Num_agents_inf
            Effort[num_party]+=Eff
            E.append(Eff)
            NUM_VOTES_INF[state_num]=Num_votes_inf
            INFLUENCED_STATES.append(state_num)
            Num_states_won_inf=np.sum((np.argsort(NUM_VOTES_INF).argsort()==Num_party-1)*1,axis=0)
            Election_winner_inf=winner(Num_states_won_inf)
            if Election_winner_inf==winner_to_be:
                break
        party-=1
        Tot_NUM_VOTES_INF[num_party]=NUM_VOTES_INF
        NUM_SEATS_INF=np.zeros((Num_states,Num_party))
        for i in range(Num_states):
            NUM_SEATS_INF[i,np.argmax(NUM_VOTES_INF[i])]=1
            
        Tot_NUM_SEATS_INF[num_party]=NUM_SEATS_INF    
        INF_STATES.append(INFLUENCED_STATES)
        #print('SR',States_to_influence,winner_to_be,Effort,E,Num_states_won,Num_states_won_inf)
    
    np.savez('MULTIPARTY_SR/Multipleparty_'+str(h)+'.npz',NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED, Effort=Effort,
            NUM_VOTES=NUM_VOTES, Tot_NUM_VOTES_INF=Tot_NUM_VOTES_INF, INF_STATES=INF_STATES, NUM_SEATS=NUM_SEATS, Tot_NUM_SEATS_INF=Tot_NUM_SEATS_INF )
   
#Function to determine the seats distribution according to the proportional representative system of election
def Prop_seats(Num_party, Num_states, Num_seats_per_state,Num_ppl_states,NUM_VOTES):
    Temp=((NUM_VOTES/Num_ppl_states.reshape(Num_states,1))*Num_seats_per_state.reshape(Num_states,1))-((NUM_VOTES/Num_ppl_states.reshape(Num_states,1))*Num_seats_per_state.reshape(Num_states,1)).astype(int)
    Additional_seats=np.zeros((Num_states,Num_party))
    for i in range(Num_states):
        #Seats_available=int(np.sum(Temp[i]).round(decimals=0))
        Seats_available=int(np.sum(Temp[i]))
        for j in range(Seats_available):
            Additional_seats[i,np.argmax(Temp[i])]=1
            Temp[i,np.argmax(Temp[i])]=0
    Num_seats=(NUM_VOTES/Num_ppl_states.reshape(Num_states,1)*Num_seats_per_state.reshape(Num_states,1)).astype(int)+Additional_seats  
    Num_seats_won=np.sum(Num_seats,axis=0)
    return Num_seats_won,Num_seats
    
#Function to determine the effort needed to change the election's outcome following a proportional representative system of election
def Prop_representative(h,MAT_INV):
    data=np.load('MULTIPARTY_RES/Inital_res_'+str(h)+'_.npz', allow_pickle=True)
    Num_party=data['Num_party']
    Num_states=data['Num_states']
    Num_seats_per_state=data['Num_seats_per_state']
    Num_ppl_states=data['Num_ppl_states']
    X0=data['X0']
    Result=data['Result']
    NUM_VOTES=data['NUM_VOTES']
    
    Num_seats_won,NUM_SEATS=Prop_seats(Num_party, Num_states, Num_seats_per_state,Num_ppl_states,NUM_VOTES)
    #Election_winner=winner(Num_seats_won)
    
    #print('Winner', Num_seats_won, Election_winner)
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    Tot_NUM_VOTES_INF=np.zeros((Num_party-1,Num_states,Num_party))
    Tot_NUM_SEATS_INF=np.zeros((Num_party-1,Num_states,Num_party))
    INF_STATES=[]
    
    party=Num_party-2
    for num_party in range(1):
        NUM_VOTES_INF=NUM_VOTES.copy()
        winner_to_be=np.argsort(Num_seats_won)[party]
        States_to_influence=States_to_be_influenced(Num_party,Num_states,NUM_VOTES,party,winner_to_be)
        E=[]
        INFLUENCED_STATES=[]
        for state in range(len(States_to_influence)):
            state_num=States_to_influence[state]
            w0=STRG.Strategies(Num_party,winner_to_be)
            Num_agents_inf,Eff,Num_votes_inf,Y=effort(Num_party,X0[state_num],MAT_INV[state_num], Result[state_num],NUM_VOTES[state_num], winner_to_be,w0)
            NUM_AGENTS_INFLUENCED[num_party]+=Num_agents_inf
            Effort[num_party]+=Eff
            E.append(Eff)
            NUM_VOTES_INF[state_num]=Num_votes_inf
            INFLUENCED_STATES.append(state_num)
            Num_seats_won_inf,Num_seats_inf=Prop_seats(Num_party, Num_states, Num_seats_per_state,Num_ppl_states, NUM_VOTES_INF)
            Election_winner_inf=winner(Num_seats_won_inf)
            if Election_winner_inf==winner_to_be:
                break
        party-=1
        Tot_NUM_VOTES_INF[num_party]=NUM_VOTES_INF
        Tot_NUM_SEATS_INF[num_party]=Num_seats_inf
        INF_STATES.append(INFLUENCED_STATES)
        #print('PR',States_to_influence,winner_to_be,Effort,E,Num_seats_won,Num_seats_won_inf)
        #print(Num_seats_won_inf,Election_winner_inf,winner_to_be)
    np.savez('MULTIPARTY_PR/Multipleparty_'+str(h)+'.npz',NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED, Effort=Effort,
            NUM_VOTES=NUM_VOTES, Tot_NUM_VOTES_INF=Tot_NUM_VOTES_INF, INF_STATES=INF_STATES, NUM_SEATS=NUM_SEATS, Tot_NUM_SEATS_INF=Tot_NUM_SEATS_INF )

    
#Function to determine the seats distribution according to the winner takes all representative system of election
def WTAR_seats(Num_party, Num_states, Num_seats_per_state, NUM_VOTES):
    Temp=np.zeros((Num_states,Num_party))
    for i in range(Num_states):
        Temp[i,np.argmax(NUM_VOTES,axis=1)[i]]=1
    Num_seats=Temp*Num_seats_per_state.reshape(Num_states,1)
    Num_seats_won=np.sum(Num_seats,axis=0)     
    return Num_seats_won, Num_seats

#Function to determine the effort needed to change the election's outcome following a winner takes all representative system of election
def WTAR_representative(h,MAT_INV):
    data=np.load('MULTIPARTY_RES/Inital_res_'+str(h)+'_.npz',allow_pickle=True)
    Num_party=data['Num_party']
    Num_states=data['Num_states']
    Num_seats_per_state=data['Num_seats_per_state']
    X0=data['X0']
    Result=data['Result']
    NUM_VOTES=data['NUM_VOTES']
    
    Num_seats_won,NUM_SEATS=WTAR_seats(Num_party, Num_states, Num_seats_per_state, NUM_VOTES)
    #Election_winner=winner(Num_seats_won)
    #print('Winner', Num_seats_won, Election_winner)
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    Tot_NUM_VOTES_INF=np.zeros((Num_party-1,Num_states,Num_party))
    Tot_NUM_SEATS_INF=np.zeros((Num_party-1,Num_states,Num_party))
    INF_STATES=[]
    
    party=Num_party-2
    for num_party in range(1):
        NUM_VOTES_INF=NUM_VOTES.copy()
        winner_to_be=np.argsort(Num_seats_won)[party]
        States_to_influence=States_to_be_influenced(Num_party,Num_states, NUM_VOTES,party,winner_to_be)
        E=[]
        INFLUENCED_STATES=[]
        for state in range(len(States_to_influence)):
            state_num=States_to_influence[state]
            w0=STRG.Strategies(Num_party,winner_to_be)
            Num_agents_inf,Eff,Num_votes_inf,Y=effort(Num_party,X0[state_num],MAT_INV[state_num], Result[state_num],NUM_VOTES[state_num], winner_to_be,w0)
            NUM_AGENTS_INFLUENCED[num_party]+=Num_agents_inf
            Effort[num_party]+=Eff
            E.append(Eff)
            NUM_VOTES_INF[state_num]=Num_votes_inf
            INFLUENCED_STATES.append(state_num)
            Num_seats_won_inf,Num_seats_inf=WTAR_seats(Num_party, Num_states, Num_seats_per_state, NUM_VOTES_INF)
            Election_winner_inf=winner(Num_seats_won_inf)
            if Election_winner_inf==winner_to_be:
                break
        party-=1
        Tot_NUM_VOTES_INF[num_party]=NUM_VOTES_INF
        Tot_NUM_SEATS_INF[num_party]=Num_seats_inf
        INF_STATES.append(INFLUENCED_STATES)
        #print('WTAR',States_to_influence,winner_to_be,Effort,E,Num_seats_won,Num_seats_won_inf)
        #print(Num_seats_won_inf,Election_winner_inf,winner_to_be)
    np.savez('MULTIPARTY_WTAR/Multipleparty_'+str(h)+'.npz',NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED, Effort=Effort,
            NUM_VOTES=NUM_VOTES, Tot_NUM_VOTES_INF=Tot_NUM_VOTES_INF, INF_STATES=INF_STATES, NUM_SEATS=NUM_SEATS, Tot_NUM_SEATS_INF=Tot_NUM_SEATS_INF )
    

def RCV_seats(Num_party, NUM_VOTES,Num_states, Num_seats_per_state,Result):
    NUM_SEATS=np.zeros((Num_states,Num_party))
    for num_state in range(Num_states):
        Threshold = 100/(Num_seats_per_state[num_state]+1)
        if Num_seats_per_state[num_state] > Num_party:
            NUM_SEATS[num_state]=np.floor((100*NUM_VOTES[num_state]/np.sum(NUM_VOTES[num_state]))/Threshold)
        else:
            NUM_SEATS[num_state]=0    
    
        Seats_RCV = int(Num_seats_per_state[num_state]-np.sum(NUM_SEATS[num_state]))
        y_temp=Result[num_state].copy()
        Num_votes=np.sum(np.argsort(Result[num_state]).argsort()==Num_party-1,axis=0)
    
        New_threshold=100/(Num_party+1)
    
        Diff=100*(Num_votes/np.sum(Num_votes))-New_threshold
        
        if Seats_RCV>1:
            #print('Seats_RCV_greater_condition', Seats_RCV,NUM_SEATS[num_state])
            Num_votes_new=Num_votes.copy()
            S_RCV=int(Num_seats_per_state[num_state]-np.sum(NUM_SEATS[num_state]))
            while S_RCV > 0:
                if np.max(Diff)>1:
                    pos_party=np.argmax(Diff) #position of the party with surplus votes
                    NUM_SEATS[num_state,pos_party]+=1
                    S_RCV-=1
                    
                    number_of_surplus_votes=int(0.01*int(np.max(Diff))*len(Result[num_state]))
                    
                    #Indices of party with surplus votes
                    Max=np.argsort(Result[num_state]).argsort()==Num_party-1
                    indices_pos_party=Max[:,pos_party]*np.arange(1,len(Result[num_state])+1,1)-1
                    indices_pos_party=np.delete(indices_pos_party,indices_pos_party==-1)
                    
                    Temp_surplus=Result[num_state][indices_pos_party]
                    Temp_surplus[:,pos_party]=0
                    Proportion=np.sum((np.argsort(Temp_surplus).argsort()==Num_party-1)*1,axis=0)/len(Temp_surplus)
                    
                    Surplus_prop=(np.floor(Proportion)*number_of_surplus_votes).astype(int)
                    while np.sum(Surplus_prop)!=number_of_surplus_votes:
                        Temp_arr=(Proportion*number_of_surplus_votes)-Surplus_prop
                        Surplus_prop[np.argmax(Temp_arr)]+=1
                    
                    #Surplus distributin to other parties
                    Max=np.argsort(Temp_surplus).argsort()==Num_party-1
                    INDICES=[] #Agents within the party
                    for num_party in range(Num_party):
                        indices=Max[:,num_party]*np.arange(1,len(Temp_surplus)+1,1)-1
                        indices=np.delete(indices,indices==-1)
                        INDICES.append(indices)
                    
                    for num_party in range(Num_party):
                        random_selection=np.random.choice(INDICES[num_party],size=Surplus_prop[num_party])
                        y_temp[random_selection,pos_party]=0
                        
                    Num_votes_new=np.sum(np.argsort(y_temp).argsort()==Num_party-1,axis=0)
                    Diff=100*(Num_votes_new/np.sum(Num_votes_new))-New_threshold
                else:   
                    pos_party=np.argmin(Num_votes_new)
                    y_temp[:,pos_party]=0
                    Num_votes_new=np.sum(np.argsort(y_temp).argsort()==Num_party-1,axis=0)
                    Diff=100*(Num_votes_new/np.sum(Num_votes_new))-New_threshold
            #print(Num_seats_per_state[num_state], np.sum(NUM_SEATS[num_state]),NUM_SEATS[num_state])        
        elif Seats_RCV==1:
            #print('Seats_RCV_equal_condition', Seats_RCV)
            NUM_SEATS[num_state,np.argmax(Diff)]+=1    
            #print(Num_seats_per_state[num_state], np.sum(NUM_SEATS[num_state]))     
        else:
            #print('Seats_RCV_zero_condition', Seats_RCV)
            NUM_SEATS[num_state]=NUM_SEATS[num_state]
            #print(Num_seats_per_state[num_state], np.sum(NUM_SEATS[num_state]))
    Num_seats_won=np.sum(NUM_SEATS,axis=0)
    
    return Num_seats_won, NUM_SEATS 



def RCV_system(h,MAT_INV):
    data=np.load('MULTIPARTY_RES/Inital_res_'+str(h)+'_.npz',allow_pickle=True)
    Num_party=data['Num_party']
    Num_states=data['Num_states']
    Num_seats_per_state=data['Num_seats_per_state']
    X0=data['X0']
    Result=data['Result']
    NUM_VOTES=data['NUM_VOTES']
    
    Num_seats_won,NUM_SEATS=RCV_seats(Num_party,NUM_VOTES,Num_states, Num_seats_per_state, Result)
    #Election_winner=winner(Num_seats_won)
    #print('Winner', Num_seats_won, Election_winner)
    Effort=np.zeros((Num_party-1))
    NUM_AGENTS_INFLUENCED=np.zeros((Num_party-1))
    Tot_NUM_VOTES_INF=np.zeros((Num_party-1,Num_states,Num_party))
    Tot_NUM_SEATS_INF=np.zeros((Num_party-1,Num_states,Num_party))
    INF_STATES=[]
    
    party=Num_party-2
    New_Result=Result.copy()
    for num_party in range(1):
        NUM_VOTES_INF=NUM_VOTES.copy()
        winner_to_be=np.argsort(Num_seats_won)[party]
        States_to_influence=States_to_be_influenced(Num_party, Num_states, NUM_VOTES,party,winner_to_be)
        E=[]
        INFLUENCED_STATES=[]
        
        for state in range(len(States_to_influence)):
            state_num=States_to_influence[state]
            w0=STRG.Strategies(Num_party,winner_to_be)
            Num_agents_inf,Eff,Num_votes_inf,New_Result[state_num]=effort(Num_party,X0[state_num],MAT_INV[state_num], New_Result[state_num],NUM_VOTES[state_num], winner_to_be,w0)
            NUM_AGENTS_INFLUENCED[num_party]+=Num_agents_inf
            Effort[num_party]+=Eff
            E.append(Eff)
            NUM_VOTES_INF[state_num]=Num_votes_inf
            INFLUENCED_STATES.append(state_num)
            Num_seats_won_inf,Num_seats_inf=RCV_seats(Num_party, NUM_VOTES_INF,Num_states, Num_seats_per_state,New_Result)
            Election_winner_inf=winner(Num_seats_won_inf)
            if Election_winner_inf==winner_to_be:
                break
        party-=1
        Tot_NUM_VOTES_INF[num_party]=NUM_VOTES_INF
        Tot_NUM_SEATS_INF[num_party]=Num_seats_inf
        INF_STATES.append(INFLUENCED_STATES)
        #print('WTAR',States_to_influence,winner_to_be,Effort,E,Num_seats_won,Num_seats_won_inf)
        #print( 'Votes',Num_seats_won, 'After inf',Num_seats_won_inf,'winner',np.argsort(Num_seats_won)[Num_party-1], Election_winner_inf,winner_to_be)
    np.savez('MULTIPARTY_RCV/Multipleparty_'+str(h)+'.npz',NUM_AGENTS_INFLUENCED=NUM_AGENTS_INFLUENCED, Effort=Effort,
            NUM_VOTES=NUM_VOTES, Tot_NUM_VOTES_INF=Tot_NUM_VOTES_INF, INF_STATES=INF_STATES, NUM_SEATS=NUM_SEATS, Tot_NUM_SEATS_INF=Tot_NUM_SEATS_INF )
    