# [Multiparty_robustness_single_DES.py](#Multiparty-robustness-single-DES)

Program to implement the robustness of the electoral system in a single electoral unit for different electoral systems and different values of confidence bound parameter.

Include the Python file 'Nat_opn_generator.py' and 'Strategies_w0.py' in the same folder of this program execution.

Nat_opn_generator.py : Generates the natural opinion in the simplex with equal density and volume for each party.

Strategies_w0.py : Defines the influence vector.

Create a folder with the name 'EPS_CRITICAL' to save the results.

## Functions in Multiparty_robustness_single_DES.py

1. [Nat_type](#Nat-type)
2. [Natural_opinion](#Natural-opinion)
3. [winner](#winner)
4. [effort](#effort)
5. [outcome](#outcome)
6. [Plurality](#Plurality)
7. [Ranked_choice_voting](#Ranked-choice-voting)
8. [Two_Round_system](#Two-Round-system)
9. [if __name__=="__main__"](#main)

<a id="Multiparty-robustness-single-DES"></a>
## Detailed description of the functions :

<a id ="Nat-type"></a>
### Nat_type(num)

Function to generate natural opinion with almost the same number of agents in each party.

INPUT

simulation number of natural opinion

OUTPUT

The natural opinion of agents is saved in 'NO_'+str(Num_party)+'_'+str(num)+'.npz' file, where 'Num_party' denotes the number of parties involved in the election, and 'num' denotes the simulation number. The file contains the natural opinion of the agents (x0, a matrix of dimensions (number of agents, number of parties)).

<a id="Natural-opinion"></a>
### Natural_opinion()
Function to parallelize the generation of natural opinion

<a id="winner"></a>
### winner(y, VOTING_TYPE)
Function to determine the winner of the election
INPUT

1. y : Final opinion of agents (a matrix of dimension (number of agents, number of parties))
2. VOTING_TYPE: electoral system (Plurality, Ranked Choice Voting(RCV), TRS (Two-round system))

OUTPUT 

1. f1 : the winner party.
2. Num_votes (An array with number of votes per party).
3. N_V (number of votes as per the RCV system following the elimination of parties).

<a id="effort"></a>
### effort(Num_party,x0, Mat_inv, Y, wp, w0, VOTING_TYPE)
Function to compute the effort needed to change the election's outcome

INPUT 

1. Num_party : Number of parties involved in the election.
2. x0 : Natural opinion, matrix of dimension (Number of agents, Number of parties).
3. Mat_inv : Mat_inv = ${(D^{-1}L + \mathbb{I})}^{-1}$, matrix of dimension (Number of agents, Number of agents).
4. Y : Final outcome, matrix of dimension (Number of agents, Number of parties).
5. wp : Next winner of the election after the influence.
6. w0 : Influence strength vector in support of 'wp'-th party.
7. VOTING_TYPE: electoral system (Plurality, Ranked Choice Voting(RCV), TRS (Two-round system))
        
OUTPUT 

1. E_min : sum of the influence vector
2. Num_agents_influenced : Number of agents influenced to change the election's outcome

<a id="outcome"></a>
### outcome(h)

Function to compute the election outcome and to determine the winner and runner-ups.

INPUT

1. h: an integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and epsilon values.

OUTPUT

The necessary computation results are temporarily saved in a file named 'Multipleparty_'+str(h)+'.npz'. The data saved in the file includes,
1. eps : the index of $\epsilon$ in the range of $\epsilon$-s.
2. num : simulation number of the natural opinion of agents.
3. MAX_NEIGHBOURS : number of agents having more neighbours in their own party or other parties (matrix of dimension (number of parties, number of parties)).
4. INTER_NEIGHBOURS : number of agents with more neighbours in other party than its own party (matrix of dimension (number of parties, number of parties)).
5. AGENTS_TRANSITION : number of agents that changed their opinion from one party to another (matrix of dimension (number of parties, number of parties)).
6. PRIORITY_VOTES_PAR : the ranking of each party (number of parties, number of parties)).
7. NORM_MAX_NEIGHBOURS : Normalized 'MAX_NEIGHBOURS' over the number of agents in each party. 
8. NORM_INTER_NEIGHBOURS : Normalized 'INTER_NEIGHBOURS' over the number of agents in each party. 
9. NORM_AGNETS_TRANSITION : Normalized 'AGENTS_TRANSITION' over the number of agents in each party. 

<a id="Plurality"></a>
### Plurality(h,x0,Mat_inv,y, VOTING_TYPE )
Function to compute the effort needed to change the election outcome in favor of the first runner-up using the Plurality system.

INPUT
1. h (variable to determine the parameters used, i.e $x0$, $\epsilon$)
2. x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
3. Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where $L$, $D$, and $\mathbb{I}$ is the laplacian, degree matrix and identity matrix resp.)
4. y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
5. VOTING_TYPE (type of electoral system to be employed, Plurality system)

OUTPUT
The output will be saved in a 'Multipleparty_plurality'+str(h)+'.npz' file in a temporary folder named 'MULTIPARTY' and will be extracted later 

1. Effort (effort required to change the election outcome)
2. Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
3. NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)

<a id="Ranked-choice-voting"></a>
### Ranked_choice_voting(h,x0,Mat_inv,y, VOTING_TYPE )
Function to compute the effort needed to change the election outcome in favor of the first runner-up using 'Ranked-choice-voting' system.

INPUT
1. h (variable to determine the parameters used, i.e. $x0$, $\epsilon$)
2. x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
3. Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where $L$, $D$, and $\mathbb{I}$ is the laplacian, degree matrix and identity matrix resp.)
4. y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
5. VOTING_TYPE (type of electoral system to be employed, Plurality system)

OUTPUT
The output will be saved in a 'Multipleparty_plurality'+str(h)+'.npz' file in a temporary folder named 'MULTIPARTY' and will be extracted later. 

1. Effort (effort required to change the election outcome)
2. Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
3. NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)

<a id="Two-Round-system"></a>
### Two_Round_system(h,x0,Mat_inv,y, VOTING_TYPE )
Function to compute the effort needed to change the election outcome in favor of the first runner-up using 'Ranked-choice-voting' system.

INPUT
1. h (variable to determine the parameters used, i.e. $x0$, $\epsilon$)
2. x0 (Natural opinion, matrix of dimension (Num_ppl, Num_party))
3. Mat_inv ($Mat_inv = {(D^{-1}L + \mathbb{I})}^{-1}$, where $L$, $D$, and $\mathbb{I}$ is the laplacian, degree matrix and identity matrix resp.)
4. y (Final opinion of agents, matrix of dimension (Num_ppl, Num_party))
5. VOTING_TYPE (type of electoral system to be employed, Plurality system)

OUTPUT
The output will be saved in a 'Multipleparty_plurality'+str(h)+'.npz' file in a temporary folder named 'MULTIPARTY' and will be extracted later. 

1. Effort (effort required to change the election outcome)
2. Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
3. NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)



### if __name__ == "__main__" 

The main function for all the initialization and function calls to initiate the computations. Two temporary folders, 'MULTIPARTY_NO' and 'MULTIPARTY', will be created to save files in between the computations and will be deleted after aggregation of results from these files.


## Final files generated after the execution of the program:

The final results are saved in a folder named 'EPS_CRITICAL'.

1. 'Ord_all_with_eq_area_'+str(Num_party)+'.npz' : The file with all necessary details, named as per the number of parties involved in the election. The files contain,
   1. Num_simulation : Number of simulations
   2. e_range : $\epsilon$ values (a vector of size number of $\epsilon$ values considered)
   3. EFF_PLURALITY : the effort needed to change the election outcome using Plurality system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   4. NUM_VOTES_PLURALITY : number of votes of each party without any external influence using Plurality system(matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties))
   5. NUM_AGENTS_INFLUENCED_PLURALITY : the number of agents influenced needed to change the election outcome using the Plurality system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   6. EFF_RCV : the effort needed to change the election outcome using RCV system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   7. NUM_AGENTS_INFLUENCED_RCV : the number of agents influenced needed to change the election outcome using the RCV system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   8. NUM_VOTES_RCV : number of votes of each party without any external influence using RCV system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties))
   9. NUM_NV : number of votes as per the RCV system following elimination of parties (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1, number of parties).
   10. EFF_TRS : the effort needed to change the election outcome using TRS system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   11. NUM_AGENTS_INFLUENCED_TRS : the number of agents influenced needed to change the election outcome using the TRS system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1))
   12. NUM_VOTES_TRS : number of votes of each party without any external influence using TRS system(matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties))
   13. MAX_NEIGHBOURS : number of agents having more neighbours in their own party or other parties (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   14. INTER_NEIGHBOURS : number of agents with more neighbours in other party than its own party (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   15. AGENTS_TRANSITION : number of agents that changed their opinion from one party to another (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   16. PRIORITY_VOTES_PAR : the ranking of each party (a matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties))
   17. NORM_MAX_NEIGHBOURS : Normalized 'MAX_NEIGHBOURS' over the number of agents in each party (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   18. NORM_INTER_NEIGHBOURS : Normalized 'INTER_NEIGHBOURS' over the number of agents in each party (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   19. NORM_AGENTS_TRANSITION : Normalized 'AGENTS_TRANSITION' over the number of agents in each party  (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties, number of parties)).
   