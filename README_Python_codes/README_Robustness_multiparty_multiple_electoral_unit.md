# [Robustness_multiparty_multiple_electoral_unit.py](#Robustness-multiparty-multiple-electoral-unit)

Program to evaluate the robustness of electoral system of multi-partite system to external attack for 15 synthetic countries with $[3,15]$ seats with an average of $9$ seats per state. We consider four different types of electoral systems (Single representatives (SR), Winner-takes-all representatives(WTAR), Proportional representatives(PR), and Proportional Ranked-Choice Voting System (PRCV)). The maximum difference in votes between any two parties is 10%. Here, we consider that the elections are conducted in states, and the results are aggregated at the country level to determine the winner. 

Include the python file 'Nat_opn_generator.py', 'Strategies_w0.py', and 'Electoral_system_function.py' in the same folder of this program execution.

Nat_opn_generator.py : Generates the natural opinion in the simplex with equal density and volume for each party.

Electoral_system_function.py: To compute the effort needed to change the election outcome in favor of first runner-ups as per the electoral systems.

The results are saved in a folder named 'Multi_States' and have a file named 'Initial_considerations.npz' with the details regarding the number of states in the synthetic countries and the corresponding seat distribution among the states.

## Functions in Robustness_multiparty_multiple_electoral_unit.py

1. [Nat_type](#Nat-type)
2. [NATURALOPINION](#NATURALOPINION)
3. [outcome](#outcome)
4. [ELECTION_NEC](#ELECTION-NEC)
5. [if __name__=="__main__"](#main)

<a id="obustness-multiparty-multiple-electoral-unit"></a>
## Detailed description of the functions :

<a id ="Nat-type"></a>
### Nat_type(Num_party,num_state)
Function to generate natural opinions of agents for a given state.

INPUT

1. Num_party : Number of parties involved in the election.
2. num_state : index of the state number within the specified synthetic country

OUTPUT

1. x0 : natural opinion of agents of the given state


<a id ="NATURALOPINION"></a>
### NATURALOPINION(num)
Function to generate the natural opinion for a synthetic country.

INPUT

1. num : simulation number

OUTPUT

The natural opinion of agents of the country is saved as 'NO_'+str(Num_party)+'_'+str(num)+'.npz', where 'Num_party' denotes the number of parties involved in the election and 'num' denotes the simulation number. The files contain the following details,

1. x0 : natural opinion of agents in the synthetic country (list of vectors with size of number of agents of each state).
2. Num_ppl_states : Number of agents in each state of the synthetic country (a vector of size of number of states).

<a id="outcome"></a>
### outcome(h)
Function to determine the final outcome of the agents and source code for computing the robustness of different types of electoral systems (PR, SR, WTAR, PRCV) 

INPUT 

1. h : an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

This function calls other functions for computations, and the results are saved in '.npz' files within the called functions.

<a id="ELECTION-NEC"></a>
### ELECTION_NEC()

Function to initialize and parallelize the computation. After the computation, the results are sorted and saved in matrix form within this function. Five temporary folders named 'MULTIPARTY_RES', 'MULTIPARTY_SR', 'MULTIPARTY_PR', 'MULTIPARTY_WTAR', and 'MULTIPARTY_RCV' are being created to save files during the execution of this function, and will be deleted after the aggregation of results from these files.

The results are saved in 'Pol_Multiparty_diff_elec_sys_'+str(Num_party)+'_'+str(N_S)+'.npz' files, where 'Num_party' denotes the number of parties involved in elections and 'N_S' denotes the index of synthetic countries.

<a id="main"></a>
### if ___name__=="__main__"
The main function for all the initialization and function calls to initiate the computation. One temporary folder named 'MULTIPARTY_NO' is created to save the natural opinions of agents and deleted after the computations.

## Final files generated after the execution of the program:

The final results are saved in a folder named 'Multi_States'.
1. Initial_considerations.npz : Contains all the values used for initializations. The file contains the number of states in the synthetic countries (NUM_STATES) and their corresponding seat distribution among the states (NUM_SEATS_PER_STATE), the number of simulations analyzed (Num_simulation), the number of $\epsilon$ values considered (neps), 

2. Pol_Multiparty_diff_elec_sys_'+str(Num_party)+'_'+str(N_S)+'.npz' : The results of the computations are saved in these files with the corresponding number of parties involved (Num_party) and the index number of synthetic countries (N_S). The files contain the number of simulations (Num_simulation), the array of $\epsilon$ values (e_range), the effort needed to change the election using SR (EFF_SR), WTA (EFF_WTAR), PR (EFF_PR), and PRCV (EFF_RCV) electoral systems, and the number of agents influenced to change the election outcome using SR (NUM_AGENTS_INFLUENCED_SR), WTA (NUM_AGENTS_INFLUENCED_WTAR), PR (NUM_AGNETS_INFLUENCED_PR) and PRCV (NUM_AGENTS_INFLUENCED_RCV) electoral systems.

