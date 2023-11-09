# [Comparison_nat3.py](#Comparison-nat3)

Program to evaluate which strategy among the 'minimum majority' and 'minimum population' gives the minimum effort needed to change the election outcome.

Create a folder with the name 'OD1' to save the results.

## Functions in Comparison_nat3.py

1. [Natural_opinion](#Natural-opinion)
2. [minimum](#minimum)
3. [calc](#calc)
4. [AS_SINGLE_ELECTION](#AS-SINGLE-ELECTION)
5. [Polarization_with_shift](#Polarization-with-shift)
6. [Negative_votes_bias](#Negative-votes-bias)
7. [if __name__=="__main__"](#main)

<a id ="Comparison-nat3"></a>
## Detailed description of the functions : 'Polarization_bias.py'

<a id="Natural-opinion"></a>
### Natural_opinion(num_type, mean1, mean2)

Function to generate the natural opinion of the agents as per the parameters given.

INPUT

1. num_type : takes the value 0 (for different values of mean and polarization) or 1 (for different values of 'p', changing the weight of two Gaussians).
2. mean1 : an array of mean values for negative Gaussians.
3. mean2 : an array of mean values for positive Gaussians.

OUTPUT

The files are in a temporary folder 'DATA' named 'Comp_bigauss_'+str(nos)+'_'+str(m)+'.npz', where 'nos' denotes the simulation number and 'm' denotes the index of polarization parameter $\Delta$ or change in weight parameter $p$.

<a id="minimum"></a>
### minimum(x0,e)

Function to calculate the minimum effort needed to change the election outcome in an electoral unit.

INPUT

1. x0 : Natural opinion of agents in the electoral unit.
2. e : the $\epsilon$ value for constructing the interaction network.
   
OUTPUT

A list named 'res' is returned with the following details.

1. Pn : Number of '+1' votes in the electoral unit.
2. Nn : Number of '-1' votes in the electoral unit.
3. E_min : the effort needed to change the election outcome in the corresponding electoral unit.

<a id="calc"></a>
### calc(h)

Function to calculate the election outcome in each of the electoral units.

INPUT

1. h: an integer from which we determine polarization parameter $\Delta$ or change in weight $p$ according to 'num_type', the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.


OUTPUT

The details after the computation are saved in 'Comp_bigauss_res_'+str(h)+'.npz' file within the temporary folder 'DATA'. The folder is deleted after the aggregation of necessary results. The file contains the following details,

1. delt : an integer to determine polarization parameter $\Delta$ or change in weight $p$ according to 'num_type'.
2. num : an integer to determine the simulation number from the set of natural opinions.
3. eps : an integer to determine the $\epsilon$ value.
4. P_ini : An array with the number of '+1' votes of the size of the number of states in the synthetic country.
5. N_ini : An array with the number of '-1' votes of the size of the number of states in the synthetic country.
6. PN : An array with the difference in '+1' and '-1' votes (WLG, we considered P_ini-N_ini).
7. Eff_MM : The effort needed to change the election outcome using 'minimum majority' strategy.
8. Eff_MP : The effort needed to change the election outcome using 'minimum population' strategy.
9. PEff_MM : The percentage of agents influenced to change the election outcome using the 'minimum majority' strategy.
10. PEff_MP : The percentage of agents influenced to change the election outcome using the 'minimum population' strategy.


<a id="AS-SINGLE-ELECTION"></a>
### AS_SINGLE_ELECTION(h)

Function to determine the effort needed to change the election's outcome given that the election was conducted as a single unit instead of multiple electoral units.

INPUT

1. h: an integer from which we determine polarization parameter $\Delta$ or change in weight $p$ according to 'num_type', the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.


OUTPUT

The details after the computation are saved in 'Comp_bigauss_res_single_'+str(h)+'.npz' file within the temporary folder 'DATA'. The folder is deleted after the aggregation of necessary results. The file contains the following details,

1. delt : an integer to determine polarization parameter $\Delta$ or change in weight $p$ according to 'num_type'.
2. num : an integer to determine the simulation number from the set of natural opinions.
3. eps : an integer to determine the $\epsilon$ value.
4. P_ini : An array with the number of '+1' votes of the size of the number of states in the synthetic country.
5. N_ini : An array with the number of '-1' votes of the size of the number of states in the synthetic country.
6. Effort : The effort needed to change the election outcome using the 'minimum' strategy.
7. N_effort : The percentage of agents influenced to change the election outcome using the 'minimum' strategy.

<a id="Polarization-with-shift"></a>
### Polarization_with_shift()

Function to parallelize the computation of the effort needed to change the election outcome for different values of mean and polarization and to aggregate the results into a single file. 

OUTPUT

The details of the computation are saved in a 'Comp_polarization_'+str(Shift[shift])+'.npz' file for election in multiple electoral units and 'Comp_polarization_single_'+str(Shift[shift])+'.npz' file for election in a single electoral unit within the folder 'OD1' along with the overall shift of the distribution $\mu$ ('Shift[shift]'). The 'Comp_polarization_'+str(Shift[shift])+'.npz' file contains the following details,

1. e_range : the range of $\epsilon$ values (a vector).
2. delta : the range of polarization parameter $\Delta$ values (a vector).
3. P_INI : the number of votes of the '+1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
4. N_INI : the number of votes of the '-1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
5. E_MM : the effort needed to change the election outcome using the 'minimum majority' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
6. E_MP : the effort needed to change the election outcome using the 'minimum population' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
7. PE_MM : the percentage of agents influenced to change the election outcome using the 'minimum majority' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
8. PE_MP :  the percentage of agents influenced to change the election outcome using the 'minimum population' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).

The 'Comp_polarization_single_'+str(Shift[shift])+'.npz' file contains the following details,

1. e_range : the range of $\epsilon$ values (a vector).
2. delta : the range of polarization parameter $\Delta$ values (a vector).
3. P_INI : the number of votes of the '+1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
4. N_INI : the number of votes of the '-1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
5. E_Single : the effort needed to change the election outcome using 'minimum strategy' (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
6. P_ESingle : the percentage of the number of agents influenced to change the election outcome using 'minimum strategy' (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).


<a id="Negative-votes-bias"></a>
### Negative_votes_bias()

Function to parallelize the computation of the effort needed to change the election outcome for different values of 'p', changing the weight of Gaussians, and aggregate the results into a single file. 

OUTPUT

The details of the computation are saved in a 'Comp_proportion.npz' file for election in multiple electoral units and 'Comp_proportion_single.npz' file for election in a single electoral unit within the folder 'OD1'. The 'Comp_proportion.npz' file contains the following details,

1. e_range : the range of $\epsilon$ values (a vector).
2. negative_votes_prop : the range of different values of 'p', changing the weight of gaussians (a vector).
3. P_INI : the number of votes of the '+1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
4. N_INI : the number of votes of the '-1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
5. E_MM : the effort needed to change the election outcome using the 'minimum majority' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
6. E_MP : the effort needed to change the election outcome using the 'minimum population' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
7. PE_MM : the percentage of agents influenced to change the election outcome using the 'minimum majority' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
8. PE_MP :  the percentage of agents influenced to change the election outcome using the 'minimum population' strategy (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).

The 'Comp_proportion_single.npz' file contains the following details,

1. e_range : the range of $\epsilon$ values (a vector).
2. delta : the range of polarization parameter $\Delta$ values (a vector).
3. P_INI : the number of votes of the '+1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
4. N_INI : the number of votes of the '-1' party (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
5. E_Single : the effort needed to change the election outcome using 'minimum strategy' (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).
6. P_ESingle : the percentage of the number of agents influenced to change the election outcome using 'minimum strategy' (a matrix of dimension (number of polarization parameter $\Delta$ considered, total number of simulations, number of $\epsilon$ values)).


<a id="main"></a>
### if __name__=="__main__"

The main function for all the initialization and function calls to initiate the computations.

## Final files generated after the execution of the program:

A folder named 'DATA' is created during the program's execution to temporarily save the intermediate results and delete them after the aggregation of necessary results.

1. Comp_polarization_'+str(Shift[shift])+'.npz
2. Comp_polarization_single_'+str(Shift[shift])+'.npz
3. Comp_proportion.npz
4. Comp_proportion_single.npz

The details saved in these files are mentioned above.
