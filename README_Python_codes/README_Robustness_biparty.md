# [Robustness_biparty.py](#Robustness-biparty)

Program to evaluate the robustness of different electoral system to external influences for a two-party electoral system for a generic case and case study on US House of Representative elections.

The folder in which this python code is executed should contain the following folders

1. 'Multi_States' with an 'Initial_consideration.npz' file. The file should contain,
    NUM_STATES: A vector with 'number of states' in each of the 15 synthetic countries.
    NUM_SEATS_PER_STATE: List of vectors, each element of vector gives the number of seats in each state of the synthetic country

2. 'HOR_DATA' with the following files,
    1. 'state_district_disribution.csv': data of seat distribution of United States.
    2. 'Election_'+str(year)+'.csv': data of percentage of votes of Republicans and Democrats from the historic data of House of Representative election from 2012-2020

Create a folder with name 'US' to save the results.

## Functions in Robustness_biparty.py

1. [Nat_type1](#Nat-type-1)
2. [Nat_type2](#Nat-type-2)
3. [Nat_type3](#Nat-type-3)
4. [NAT1](#Nat1)
5. [NAT2](#Nat1)
6. [NAT3](#Nat1)
7. [natural_opinion](#natural-opinion)
8. [single_representative](#single-representative)
9. [WTA_representative](#wta-representative)
10. [Prop_representative](#prop-representative)
11. [winner](#winner)
12. [minimum](#minimum)
13. [calc](#calc)
14. [Single_representative](#Single-representative)
15. [WTAR_representatuve](#WTAR-representative)
16. [Proportional_representative](#Proportional-representative)
17. [ELECTIONS](#elections)
18. [REPUB_CALC](#repub-calc)
19. [SYNTHETIC_COUNTRIES](#synthetic-countries)
20. [CASE_STUDY_US](#case-study-us)
21. [DEFINE_SYNTHETIC_COUNTRY](#define-synthetic-country)

<a id = "Robustness-biparty"></a>
## Detailed description of the functions : 'Robustness_biparty.py'

<a id="Nat-type-1"></a>
### Nat_type1(R)

To generate the natural opinion of the agents following D1 distribution, i.e. bias introduced by changing the weights of the gaussians.

INPUT

R : Array of percentage of votes of one party in one electoral unit 

OUTPUT

y: Natural opinion of agents in a state (vector of number of agents in the corresponding state)


<a id="Nat-type-2"></a>
### Nat_type2(R,nop1)

To generate the natural opinion of the agents following D2 distribution, i.e. bias introduced by changing the mean of the distribution.

INPUT

R : Array of percentage of votes of one party 
nop1: Half of the agents 

OUTPUT

y: Natural opinion of agents in a state (vector of number of agents in the corresponding state)


<a id="Nat-type-3"></a>
### Nat_type3(R)

To generate the natural opinion of the agents following D3 distribution, i.e. bias introduced by changing the mean of the distribution.


INPUT

R : Array of percentage of votes of one party 

OUTPUT

y: Natural opinion of agents in a state (vector of number of agents in the corresponding state)


<a id="Nat1"></a>
### NAT1(nos), NAT2(nos), NAT3(nos)

To generate natural opinion corresponding to a country with multiple electoral unit with varying seats following D1: NAT1(nos), D2: NAT2(nos), and D3: NAT3(nos) distribution.
This function aggregates the natural opinion of different electoral units (for example, opinion of agents in a state) into a list of vectors.


INPUT

1.nos (realization number)

OUTPUT

The data will be saved into an 'NO_'+str(nos)+'.npz' file named with the corresponding simulation number (nos), and can be loaded whenever necessary.

<a id="natural-opinion"></a>
### natural_opinion(num_type)

To parallelize the generation of natural opinion according to the distribution of natural opinion.

INPUT

1. num_type (the type of distribution, it can be either 0, 1 or 2)

OUTPUT

The natural opinions following different types of distribution will be saved in 'NO_'+str(nos)+'.npz' files from the function called within this function.

<a id="single-representative"></a>
### single_representative(PN, Num_states)

Function to calculate the number of seats acquired by both parties, to determine the winner

INPUT

1. PN: Difference between the seats, if PN > 0, then the '+1' party has more votes, and vice-versa.
2. Num_states: Number of states in the country.

OUTPUT

1. P1: Number of seats for the '+1' party.
2. N1: Number of seats for the '-1' party.

<a id="wta-representative"></a>
### WTA_representative(PN, Num_districts_perstate, Num_districts)

Function to calculate the number of seats acquired by both parties, to determine the winner

INPUT

1. PN: Difference between the seats, if PN > 0, then the '+1' party has more votes, and vice-versa.
2. Num_districts_perstate: Number of districts per state, which is equal to the number of states.
3. Num_districts: Number of districts in the whole country, i.e. again equal to the total number of seats.
    
OUTPUT

1. P1: Number of seats for the '+1' party.
2. N1: Number of seats for the '-1' party.

<a id="prop-representative"></a>
### Prop_representative(P_ini, N_ini, Proportion, Num_districts)

Function to calculate the number of seats acquired by both parties, to determine the winner

INPUT

1. P_ini: vector of votes of '+1' party, vector of size 'Num_states'.
2. N_ini: vector of votes of '-1' party, vector of size 'Num_states'.
3. Proportion: Number of seats per state/ Num_of agents in the corresponding state, vector of size 'Num_states'.
4. Num_districts: Number of districts in the whole country, i.e. again equal to the total number of seats.
    
OUTPUT

1. P1: Number of seats for the '+1' party.
2. N1: Number of seats for the '-1' party.

<a id="winner"></a>
### winner(P,N)

Function to determine the winner of the election

INPUT

1. P: number of '+1' seats acquired
2. N: number of '-1' seats acquired

OUTPUT

1. f (f=1, implies '+1' party is the winner and f=0, implies '-1' party is the winner)

<a id="minimum"></a>
### minimum(x0, eps)

Function to determine the effort needed to change the election outcome in a single electoral unit

INPUT

1. x0: vector of natural opinions
2. eps: index of epsilon value

OUTPUT

1. E_min: effort needed to change the outcome of the election following 'Minimum strategy' explained in the paper.
2. p: '+1' votes after the outcome is changed.
3. n: '-1' votes after the outcome is changed.

<a id="calc"></a>
### calc(h)

Function to calculate the number of votes of each party within each electoral unit

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

OUTPUT

Saves the neccessary details to an 'RDres_mid_'+str(h)+'.npz' files, which then can retrieved later whenever necessary instead of recalculating. We save the following details,

1. P_ini, N_ini : Final votes of each party within each electoral unit before external influence (vector of size of number of states).
2. X: the natural opinions of all agents in the country (list of vectors with each vector of size number of agents in each electoral unit).
3. PN: the difference between the final votes of each electoral unit (vector of size of number of states).

<a id="Single-representative"></a>
### Single_representative(h)

Function to determine the minimal effort needed to change the election's outcome following SR representative electoral system

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

OUTPUT

Saves the effort needed to change the election outcome following SR representative electoral system into 'RDres_mid_SR_'+str(h)+'.npz' files, with the name 'Eff_SR'.

<a id="WTAR-representative"></a>
### WTAR_representative(h)

Function to determine the minimal effort needed to change the election's outcome following WTA representative electoral system

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

OUTPUT

Saves the effort needed to change the election outcome following WTA representative electoral system into 'RDres_mid_WTAR_'+str(h)+'.npz' files, with the name 'Eff_WTAR'.

<a id="Proportional-representative"></a>
### Proportional_representative(h)

Function to determine the minimal effort needed to change the election's outcome following PR representative electoral system

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and epsilon values.

OUTPUT

Saves the effort needed to change the election outcome following PR representative electoral system into 'RDres_mid_PR_'+str(h)+'.npz' files, with the name 'Eff_PR'.

<a id="elections"></a>
### ELECTIONS(N_S, num_type, Num_states, Num_districts, Proportion, REPUB)

Function to parallelize the computations and to aggregate the results into a single files based on the type of natural opinion distribution and synthetic country/ year of election in case of case study of US

INPUT

1. N_S: index of the synthetic country (int with values from 0 to 14).
2. num_type: type of natural opinion distribution (int with values from 0 to 2).
3. Num_states: Number of states in the synthetic country or in the US (in case of case study of US) (int with values from 16 to 20 or 50 in case of US).
4. Num_districts: Number of districts in the synthetic country or in the US (in case of case study of US) (int with values equal sum of seats in the synthetic countries or 435 in case of US).
5. Proportion: Number of seats per state/ Num_of agents in the corresponding state, vector of size 'Num_states'.
6. REPUB: vectors indicating the percentage of votes of one party. (list of vectors, each vector indicates an electoral unit, each element of the vector indicate the percentage of votes of one party say '-1').

OUTPUT

Saves the neccessary details for our analysis into a 'RealDout_'+ str(num_type+1) +'_robustness_'+str(N_S)+'.npz' file, named as per the 'num_type' and index of the synthetic country. The file contains the following details,

1. P_INI: final votes of '+1' party in each electoral unit (matrix of size (number of simulation, number of epsilon values, number of states)).
2. N_INI: final votes of '+1' party in each electoral unit (matrix of size (number of simulation, number of epsilon values, number of states)).
3. E_SR: effort needed to change the election outcome follow SR representative electoral system (matrix of dimension (number of simulation, number of epsilon values)).
4. E_WTAR: effort needed to change the election outcome follow WTA representative electoral system (matrix of dimension (number of simulation, number of epsilon values)).
5. E_PR: effort needed to change the election outcome follow PR representative electoral system (matrix of dimension (number of simulation, number of epsilon values)).

<a id="repub-calc"></a>
### REPUB_CALC(Num_states, Num_districts_perstate)

Function to determine the ppercentage of votes of one of party (say '-1'), such that the max difference between the two parties is 10%.

INPUT 

1. Num_states: Number of states in the country (i.e. the number of electoral units)
2. Num_districts_perstate: Number of seats in each of the state.

OUTPUT

1. REPUB: vectors indicating the percentage of votes of one party. (list of vectors, each vector indicates an electoral unit, each element of the vector indicate the percentage of votes of one party say '-1').

<a id="synthetic-countries"></a>
### SYNTHETIC_COUNTRIES()
Function to initiate the computation of the robustness of electoral system on a set of 15 synthetic countries.

<a id = "case-study-us"></a>
### CASE_STUDY_US()
Function to initiate the computation of the robustness of the US using the historical data of US House of representative elections.

<a id = "define-synthetic-country"></a>
### DEFINE_SYNTHETIC_COUNTRY()
Function to initiate the computation of the robustness of electoral system by defining your own synthetic country. By running this function, you will be asked for each of the initialization required, and by providing them, you can find the robustness of electoral system using the specified synthetic country. You can as well give the specification of an exisiting country, like Switzerland, Germany and so on.


<a id = "main"></a>
### if __name__ == "__main__"
Main function for all the initialization and function calls to initiate the computations.


## Final files generated after the execution of the program:

Two folders named 'D_DATA' and 'DATA' are created during the execution of the program to save temporary files, and deleted after aggregating the required results.
All the files are saved in a folder named 'US'

1. Assumptions.npz : contains all the initializations used, i.e Number of simulations (NOS), epsilon range (e_range), strength of influence (Influence_strength), Polarization value (Delta), mean of the total distribution (mean), standard deviation of the gaussain distribution (std_dev), number of people considered in each district of the state (NP_district).

2.  Republicans.npz : contains the vote percentages of the electoral unit (REPUB), Number of states (Num_states), Number of districts per state (Num_districts_perstate), Number of districts (Num_districts), Total number of agents in the whle country (N), Proportion of seats to the number of agents in the electoral unit (Proportion)

3.  RealDout_'+ str(num_type+1) +'_robustness_'+str(N_S)+'.npz : Files with the data from the computations. The file is named as per the type of natural opinion distribution and index of synthetic country or the year of historic data of US House of Representative election in case of the case study.
The file contains the final votes of '+1' party in each electoral unit (P_INI: matrix of size (number of simulation, number of epsilon values, number of states)), final votes of '+1' party in each electoral unit (N_INI: matrix of size (number of simulation, number of epsilon values, number of states)), effort needed to change the election outcome follow SR representative electoral system (E_SR: matrix of size (number of simulation, number of epsilon values)), effort needed to change the election outcome follow WTA representative electoral system (E_WTAR: matrix of size (number of simulation, number of epsilon values)), effort needed to change the election outcome follow PR representative electoral system (E_PR: matrix of size (number of simulation, number of epsilon values)). 
