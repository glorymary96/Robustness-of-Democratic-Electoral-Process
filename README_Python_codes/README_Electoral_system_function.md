# [Electoral_system_function.py](#Electoral-system-function)

Python code with a set of functions to calculate the robustness of different electoral systems. This program is called from the 'Robustness_multiparty_multiple_electoral_unit.py' to evaluate the robustness of different electoral systems. 

## Functions in Electoral_system_function.py

1. [winner](#winner)
2. [effort](#effort)
3. [States_to_be_influenced](#States-to_be-influenced)
4. [Single_representative](#Single-representative)
5. [Prop_seats](#Prop-seats)
6. [WTAR_seats](#WTAR-seats)
7. [WTAR_representative](#WTAR-representative)
8. [RCV_seats](#RCV-seats)
9. [RCV_system](#RCV-system)

<a id ="Electoral-system-function"></a>
## Detailed description of the functions :

<a id="winner"></a>
### winner(Num_votes)

Function to determine the winner of the election.

INPUT

1. Num_votes : An array with number of votes per party.

OUTPUT 

1. f1 : the winner party


<a id="effort"></a>
### effort(Num_party,x0,Mat_inv,Y,Num_votes,wp,w0)

Function to compute the effort needed to change the election's outcome in an electoral unit.

INPUT

1. Num_party : Number of parties involved in the election.
2. x0 : Natural opinion of agents in the electoral unit being influenced.
3. Mat_inv : Mat_inv = ${(D^{-1}L + \mathbb{I})}^{-1}$, matrix of dimension (Number of agents, Number of agents).
4. Y : Final outcome, matrix of dimension (Number of agents in the corresponding electoral unit, number of parties).
5. Num_votes : An array with number of votes of each party in the corresponding electoral unit.
6. wp : Next winner of the election after the influence in the corresponding electoral unit.
7. w0 : Influence strength vector in support of 'wp'-th party

OUTPUT

1. Num_agents_influenced : Number of agents influenced to change the election outcome
2. E_min : effort needed to change the election outcome.
3. Num_votes1 : An array with number of votes of each party after influence.
4. Y : Final outcome, matrix of dimension (number of agents in the corresponding electoral unit, number of parties)

<a id="States-to_be-influenced"></a>
### States_to_be_influenced(Num_party, Num_states, NUM_VOTES,party,winner_to_be)

Function to determine the states to be influenced following the minimum majority strategy (the states with the minimum difference in majority will be influenced first).

INPUT

1. Num_party : Number of parties involved in the election.
2. Num_states : Total number of states in the synthetic country.
3. NUM_VOTES : Number of votes of each party in each of the electoral unit (matrix of dimension (number of states, number of parties)).
4. party : an variable to determine the next winner.
5. winner_to_be : the next winner-to-be party.

OUTPUT

1. States_to_influence : An array of indices of states to influence as per the minimum majority strategy.

<a id="Single-representative"></a>
### Single_representative(h,MAT_INV)

Function to determine the effort needed to change the election outcome following the SR electoral system.

INPUT 

1. h : An integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.
2. MAT_INV : A list of ${(D^{-1}L + \mathbb{I})}^{-1}$ matrices corresponding to each electoral unit.
   

OUTPUT

The necessary details after the computation is saved in a 'Multipleparty_'+str(h)+'.npz' within the temporarily created folder named 'MULTIPARTY_SR'. The file contains the following details.

1. NUM_AGENTS_INFLUENCED : Number of agents influenced to change the election outcome.
2. Effort : effort needed to change the election outcome.
3. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).
4. Tot_NUM_VOTES_INF : Number of votes of each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).
5. INF_STATES : Index of the states influenced to change the election outcome.
6. NUM_SEATS : Number of seats allotted to each party in each electoral unit (matrix of dimension (number of states, number of parties)).
7. Tot_NUM_SEATS_INF : Number of seats allotted to each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).

<a id="Prop-seats"></a>
### Prop_seats(Num_party, Num_states, Num_seats_per_state,Num_ppl_states,NUM_VOTES)

Function to determine the number of seats allotted to each party depending on their percentage of votes.

INPUT 

1. Num_party : Number of parties involved in the election (an int).
2. Num_states : Total number of states in the synthetic country (an int).
3. Num_seats_per_state : Number of seats in each of the states in the synthetic country (an array of size of the number of states).
4. Num_ppl_states : Number of agents in each of the states in the synthetic country (an array of size of the number of states).
5. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).

OUTPUT

1. Num_seats_won : Number of seats won by each party in the entire synthetic country (an array of size of the number of states).
2. Num_seats : Number of seats won by each party in each electoral unit (matrix of dimension (number of states, number of parties)).


<a id="Prop-representative"></a>
### Prop_representative(h,MAT_INV)

Function to determine the effort needed to change the election outcome following the PR electoral system.

INPUT 

1. h : An integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.
2. MAT_INV : A list of ${(D^{-1}L + \mathbb{I})}^{-1}$ matrices corresponding to each electoral unit.
   

OUTPUT

The necessary details after the computation are saved in a 'Multipleparty_'+str(h)+'.npz' within the temporarily created folder named 'MULTIPARTY_PR'. The file contains the following details.

1. NUM_AGENTS_INFLUENCED : Number of agents influenced to change the election outcome.
2. Effort : effort needed to change the election outcome.
3. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).
4. Tot_NUM_VOTES_INF : Number of votes of each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).
5. INF_STATES : Index of the states influenced to change the election outcome.
6. NUM_SEATS : Number of seats allotted to each party in each electoral unit (matrix of dimension (number of states, number of parties)).
7. Tot_NUM_SEATS_INF : Number of seats allotted to each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).


<a id="WTAR-seats"></a>
### WTAR_seats(Num_party, Num_states, Num_seats_per_state, NUM_VOTES)

Function to determine the number of seats allotted to each party depending on their percentage of votes.

INPUT 

1. Num_party : Number of parties involved in the election (an int).
2. Num_states : Total number of states in the synthetic country (an int).
3. Num_seats_per_state : Number of seats in each of the states in the synthetic country (an array of size of the number of states).
4. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).

OUTPUT

1. Num_seats_won : Number of seats won by each party in the entire synthetic country (an array of size of the number of states).
2. Num_seats : Number of seats won by each party in each electoral unit (matrix of dimension (number of states, number of parties)).


<a id="WTAR-representative"></a>
### WTAR_representative(h,MAT_INV)

Function to determine the effort needed to change the election outcome following the WTAR electoral system.

INPUT 

1. h : An integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.
2. MAT_INV : A list of ${(D^{-1}L + \mathbb{I})}^{-1}$ matrices corresponding to each electoral unit.
   

OUTPUT

The necessary details after the computation are saved in a 'Multipleparty_'+str(h)+'.npz' within the temporarily created folder named 'MULTIPARTY_WTAR'. The file contains the following details.

1. NUM_AGENTS_INFLUENCED : Number of agents influenced to change the election outcome.
2. Effort : effort needed to change the election outcome.
3. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).
4. Tot_NUM_VOTES_INF : Number of votes of each party in each electoral unit after influecne (matrix of dimension (number of states, number of parties)).
5. INF_STATES : Index of the states influenced to change the election outcome.
6. NUM_SEATS : Number of seats allotted to each party in each electoral unit (matrix of dimension (number of states, number of parties)).
7. Tot_NUM_SEATS_INF : Number of seats allotted to each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).


<a id="RCV-seats"></a>
### RCV_seats(Num_party, NUM_VOTES, Num_states, Num_seats_per_state, Result)

Function to determine the number of seats allotted to each party depending on their percentage of votes.

INPUT 

1. Num_party : Number of parties involved in the election (an int).
2. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).
3. Num_states : Total number of states in the synthetic country (an int).
4. Num_seats_per_state : Number of seats in each of the states in the synthetic country (an array of size of the number of states).
5. Results : Final opinion of agents in each electoral unit (list of matrices of dimension (number of states, number of parties)).

OUTPUT

1. Num_seats_won : Number of seats won by each party in the entire synthetic country (an array of size of the number of states).
2. NUM_SEATS : Number of seats won by each party in each electoral unit (matrix of dimension (number of states, number of parties)).

<a id="RCV-system"></a>
### RCV_system(h,MAT_INV)

Function to determine the effort needed to change the election outcome following the PRCV electoral system.

INPUT 

1. h : An integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.
2. MAT_INV : A list of ${(D^{-1}L + \mathbb{I})}^{-1}$ matrices corresponding to each electoral unit.
   

OUTPUT

The necessary details after the computation are saved in a 'Multipleparty_'+str(h)+'.npz' within the temporarily created folder named 'MULTIPARTY_WTAR'. The file contains the following details.

1. NUM_AGENTS_INFLUENCED : Number of agents influenced to change the election outcome.
2. Effort : effort needed to change the election outcome.
3. NUM_VOTES : Number of votes of each party in each electoral unit (matrix of dimension (number of states, number of parties)).
4. Tot_NUM_VOTES_INF : Number of votes of each party in each electoral unit after influecne (matrix of dimension (number of states, number of parties)).
5. INF_STATES : Index of the states influenced to change the election outcome.
6. NUM_SEATS : Number of seats allotted to each party in each electoral unit (matrix of dimension (number of states, number of parties)).
7. Tot_NUM_SEATS_INF : Number of seats allotted to each party in each electoral unit after influence (matrix of dimension (number of states, number of parties)).

