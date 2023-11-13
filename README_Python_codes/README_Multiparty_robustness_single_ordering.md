# [Multiparty_robustness_single_ordering.py](#Multiparty-robustness-single-ordering)

Program to implement the robustness of the electoral system in a single electoral unit for Plurality system with just ordering (without reduction of volume of parties at the center) and different values of confidence bound parameter.

Include the Python file 'Strategies_w0.py' in the same folder of this program execution.

Strategies_w0.py : Defines the influence vector.

Create a folder with the name 'EPS_CRITICAL' to save the results.

## Functions in Multiparty_robustness_single_DES.py

1. [ordering](#ordering)
2. [x0_spec_gen](#x0-spec-gen)
3. [Nat_type](#Nat-type)
4. [Natural_opinion](#Natural-opinion)
5. [winner](#winner)
6. [effort](#effort)
7. [outcome](#outcome)
8. [if __name__=="__main__"](#main)

<a id="Multiparty-robustness-single-ordering"></a>
## Detailed description of the functions :

<a id ="ordering"></a>
### ordering(x, Num_ppl_party)
Function to order the parties

INPUT

x: the natural opinions of agents before ordering.
Num_ppl_party : number of agents supporting a specific party.

OUTPUT

x: the natural opinions of agents after ordering.

### x0_spec_gen(num_party,Num_ppl_party)
Function to generate the natural opinion of agents uniformly in a simplex.

INPUT 

num_party : position of the party.
Num_ppl_party : number of agents supporting 'num_party'.

OUTPUT

x0 : natural opinion of agents after ordering supporting 'num_party'.

<a id ="Nat-type"></a>
### Nat_type(num)

Function to generate natural opinion with almost the same number of agents in each party.

INPUT

simulation number of natural opinion

OUTPUT

The natural opinion of agents is saved in 'NO_'+str(Num_party)+'_'+str(num)+'.npz' file, where 'Num_party' denotes the number of parties involved in the election, and 'num' denotes the simulation number. The file contains the natural opinion of the agents (x0, a matrix of dimensions (number of agents, number of parties)).

<a id="Natural-opinion"></a>
### Natural_opinion()
Function to parallelize the generation of natural opinion of agents.

<a id="winner"></a>
### winner(Num_votes)
Function to determine the winner of the election.
INPUT

1. Num_votes (An array with number of votes per party).

OUTPUT 

1. f1 : the winner party.

<a id="effort"></a>
### effort(Num_party,x0, Mat_inv, Y, wp, w0, VOTING_TYPE)
Function to compute the effort needed to change the election's outcome

INPUT 

1. x0 : Natural opinion, matrix of dimension (Number of agents, Number of parties).
2. Mat_inv : Mat_inv = ${(D^{-1}L + \mathbb{I})}^{-1}$, matrix of dimension (Number of agents, Number of agents).
3. Y : Final outcome, matrix of dimension (Number of agents, Number of parties).
4. Num_votes (An array with number of votes per party).
5. wp : Next winner of the election after the influence.
6. w0 : Influence strength vector in support of 'wp'-th party.
   
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
3. Effort (effort required to change the election outcome)
4. Num_votes (number of votes for each party in a 1-D vector of size 'Num_party')
5. NUM_AGENTS_INFLUENCED (number of agents influenced to change the election outcome)


### if __name__ == "__main__" 

The main function for all the initialization and function calls to initiate the computations. Two temporary folders, 'MULTIPARTY_NO' and 'MULTIPARTY', will be created to save files in between the computations and will be deleted after aggregation of results from these files.


## Final files generated after the execution of the program:

The final results are saved in a folder named 'MULT_ORD'.

1. Multiparty_ordering_'+str(Num_party)+'.npz' : The file with all necessary details, named as per the number of parties involved in the election. The files contain,
   1. Effort : the effort needed to change the election outcome using Plurality system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1)).
   2. e_range : $\epsilon$ values (a vector of size number of $\epsilon$ values considered).
   3. NUM_VOTES : number of votes of each party without any external influence using Plurality system(matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties)).
   4. NUM_AGENTS_INFLUENCED : the number of agents influenced needed to change the election outcome using the Plurality system (matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1)).
  