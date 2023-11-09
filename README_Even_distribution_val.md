# [Even_distribution_val.py](#Even-distribution-val)

Program to validate the model using the historical data of the US House of Representative election from 2012-2012.

The folder in which this Python code is executed should contain the following folders.

1. 'HOR_DATA' with the following files,
    1. 'state_district_disribution.csv': data of seat distribution of United States.
    2. 'Election_'+str(year)+'.csv': data of the percentage of votes of Republicans and Democrats from the historical data of House of Representative election from 2012-2020

Create a folder with the name 'US_CONST_EVEN' to save the results.

## Functions in Even_distribution_val.py

1. [Nat_type1](#Nat-type-1)
2. [Nat_type2](#Nat-type-2)
3. [Nat_type3](#Nat-type-3)
4. [natural_opinion](#natural-opinion)
5. [winner](#winner)
6. [influence](#influence)
7. [calc_HOR](#calc-HOR)
8. [HOR_ELECTION](#HOR-ELECTION)
9. [if __name__ == "__main__"](#main)

<a id = "Even-distribution-val"></a>
## Detailed description of the functions : 'Even_distribution_val.py'

<a id="Nat-type-1"></a>
### Nat_type1(R)

To generate the natural opinion of the agents following D1 distribution, i.e. bias introduced by changing the weights of the Gaussians.

INPUT

R : Array of percentage of votes of one party in one electoral unit 

<a id="Nat-type-2"></a>
### Nat_type2(R)

To generate the natural opinion of the agents following D2 distribution, i.e. bias introduced by changing the mean of the distribution.

INPUT

R : Array of the percentage of votes of one party 
nop1: Half of the agents 

<a id="Nat-type-3"></a>
### Nat_type3(R)

To generate the natural opinion of the agents following D3 distribution, i.e. bias introduced by changing the mean of the distribution.

INPUT

R : Array of the percentage of votes of one party 

<a id="natural-opinion"></a>
### natural_opinion(type)

To parallelize the generation of natural opinion according to the distribution of natural opinion.

INPUT

1. type (the type of distribution, it can be either 0, 1 or 2)

OUTPUT
The natural opinions following different types of distribution will be saved in 'NO_'+str(nos)+'.npz' files, names as per the simulation number and can be loaded whenever necessary. The file contains the natural opinion of agents (X0, a list of vectors of the size of the number of agents we consider in each district.)

<a id="winner"></a>
### winner(P,N)

Function to determine the winner of the election

INPUT

1. P: number of '+1' seats acquired
2. N: number of '-1' seats acquired

OUTPUT

1. f (f=1, implies the '+1' party is the winner, and f=0, implies the '-1' party is the winner)

<a id="influence"></a>
### influence(x0,y, Mat_inv, num_agents_to_inf, Pn, Nn)

Function to impose external influence on the agents of the population.

INPUT

1. x0: natural opinion of agents (vector of size number of agents in an electoral unit)
2. y: Final opinion of agents before external influence (vector of size number of agents in an electoral unit)
3. Mat_inv : $(D^{-1}L + I)^{-1}$, where $D, L, I$ are degree, laplacian and identity matrix of the interaction network.
4. num_agents_to_inf : the number of agents to be influenced.
5. Pn: number of '+1' votes within the electoral unit.
6. Nn: number of '-1' votes within the electoral unit.

OUTPUT

1. p: number of '+1' votes within the electoral unit.
2. n: number of '-1' votes within the electoral unit.

<a id="calc-HOR"></a>
### calc_HOR(h)

Function to calculate the votes in each electoral district and initiate the influence on agents defined by the percentage of agents to be influenced.

INPUT

1. h: an integer from which we determine the natural opinion and epsilon to be used from the set of natural opinions and $\epsilon$ values.

OUTPUT

The results are saved in an 'USdatares_'+str(h)+'.npz' files with the following details,
 
1. num : number of the simulation of natural opinion
2. eps : index of the epsilon range array to choose the epsilon value
3. success : matrix of success of influence (matrix of dimension (number of influence percentage considered, number of districts of US))

<a id="HOR-ELECTION"></a>
### HOR_ELECTION()

Function to parallelize and initiate the computation.

OUTPUT

The results from the computation are aggregated and saved in a '.npz' file with the following details,

1. e_range: Array of range of epsilon values.
2. SUCCESS: matrix of success of influence (matrix of dimension (number of simulations, number of epsilon values, number of influence percentages, number of districts in the US)).
3. Influence_percentage_array : Array of influence percentages considered.

<a id="main"></a>
### if __name__ == "__main__"
The main function for all the initialization and function calls to initiate the computations.


## Final files generated after the execution of the program:

All the files are saved in a folder named 'US_CONST_EVEN', with the following file names,

1. usrepub_'+str(num_type+1)+'_'+ str(year)+'.npz' : files with all the necessary details named as per the type of natural opinion distribution and the year of US House of Representative election historical data. It contains an array of epsilon values (e_range), a matrix of success using external influence (SUCCESS), and an array of influence percentages considered (Influence_percentage_array).

   

   