# [Robustness_multiparty_single_electoral unit.py](#Robustness-multiparty-single-electoral-unit)

Program to implement the robustness of the electoral system in a single electoral unit for different values of confidence bound parameter.

Include the Python file 'Nat_opn_generator.py' and 'Strategies_w0.py' in the same folder of this program execution.

Nat_opn_generator.py : Generates the natural opinion in the simplex with equal density and volume for each party.

Strategies_w0.py : Defines the influence vector.

Create a folder with the name 'EPS_CRITICAL' to save the results.


## Functions in Robustness_multiparty_single_electoral unit.py

1. [Nat_type](#Nat-type)
2. [Natural_opinion](#Natural-opinion)
3. [winner](#winner)
4. [effort](#effort)
5. [outcome](#outcome)
6. [if __name__ == "__main__"](#main)


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
### winner(Num_votes)
Function to determine the winner of the election
INPUT

1. Num_votes (An array with number of votes per party).

OUTPUT 

1. f1 : the winner party

<a id="effort"></a>
### effort(x0, Mat_inv, Y, Num_votes, wp, w0)
Function to compute the effort needed to change the election's outcome

INPUT 

1. x0 : Natural opinion, matrix of dimension (Number of agents, Number of parties)
2. Mat_inv : Mat_inv = ${(D^{-1}L + \mathbb{I})}^{-1}$, matrix of dimension (Number of agents, Number of agents)
3. Y : Final outcome, matrix of dimension (Number of agents, Number of parties)
4. Num_votes : An array with the number of votes of each party
5. wp : Next winner of the election after the influence
6. w0 : Influence strength vector in support of 'wp'-th party 
        
OUTPUT 

1. E_min : sum of the influence vector
2. Num_agents_influenced : Number of agents influenced to change the election's outcome

<a id="outcome"></a>
### outcome(h)

Function to compute the election outcome and to determine the winner and runner-ups.

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

OUTPUT

The necessary computation results are temporarily saved in a file named 'Multipleparty_'+str(h)+'.npz'. The data saved in the file includes,
1. eps : the index of $\epsilon$ in the range of $\epsilon$-s.
2. num : simulation number of the natural opinion of agents.
3. Effort : effort needed to change the election outcome, normalized over the influence strength.
4. Num_votes : An array with the number of votes of each party
5. NUM_AGENTS_INFLUENCED : Number of agents influenced to change the election outcome.

    
### if __name__ == "__main__" 

The main function for all the initialization and function calls to initiate the computations. Two temporary folders, 'MULTIPARTY_NO' and 'MULTIPARTY', will be created to save files in between the computations and will be deleted after aggregation of results from these files.


## Final files generated after the execution of the program:

The final results are saved in a folder named 'EPS_CRITICAL'.

1. 'Ord_all_with_eq_area_'+str(Num_party)+'.npz' : The file with all necessary details, named as per the number of parties involved in the election. The files contain the effort needed to change the election outcome (Effort, matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties-1)), the range of $\epsilon$ values (e_range, a vector of size number of $\epsilon$ values considered), number of votes of each party without any external influence (NUM_VOTES, matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties)), number of agents influenced to change the election outcome (NUM_AGENTS_INFLUENCED, matrix of dimension (Number of simulations, number of $\epsilon$ values, number of parties)).