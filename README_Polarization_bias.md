# [Polarization_bias.py](#Polarization-bias)

Program to evaluate the robustness of the electoral system in a single electoral unit with varying polarization ($\Delta$) and bias($\mu, p$).

Create a folder named 'VAR_FILES' to save the results.

## Functions in Polarization_bias.py :

1. [Natural_opinion](#Natural-opinion)
2. [outcomme](#outcome)
3. [Polarization](#Polarization)
4. [Proportion](#Proportion)
5. [if __name__=="__main__"](#main)


## Detailed description of the functions : 'Polarization_bias.py'

<a id="Natural-opinion"></a>
### Natural_opinion(num_type)

Function to generate natural opinion of agents according to different parameters.
1. num_type = 0: Imposing bias with $\mu$
2. num_type = 1: Imposing bias with $p$ (change in weights)

<a id="outcome"></a>
### outcome(h)

Function to compute the effort needed to change the election outcome following the Random and Minimum strategy defined in the paper.

INPUT

1. h: an integer from which we determine the natural opinion and $\epsilon$ to be used from the set of natural opinions and $\epsilon$ values.

OUTPUT

The necessary results of the computations are saved in a 'Res_'+str(h)+'.npz' file within the temporarily created folder 'DATA'. The files contain 

1. pos : index of the array of mean values.
2. nos : simulation number of natural opinions.
3. eps : index of the array of epsilon values.
4. Num_agents_inf: Number of agents influenced to change the election outcome using Minimum strategy.
5. P_ini : number of '+1' votes before influence.
6. N_ini : number of '-1' votes before influence.
7. mu : average number of agents influenced to change the election outcome over 30 independent choices of Random strategy of influence.
8. sigma : standard deviation of the number of agents influenced to change the election outcome over 30 independent choices of Random strategy of influence.

<a id="Polarization"></a>
### Polarization()

Function to initiate and parallelize the computation of the effort needed to change the election outcome.

OUTPUT

The necessary results of the computations are saved in a 'Min_Polarization_'+str(sd_num)+'_'+str(bi_mean_num)+'.npz' file, along with index of standard deviation ('sd_num') and polarization parameter ('bi_mean_num') within the temporarily created folder named 'FILES'. The files contain
1. NOS : number of simulations
2. SD : vector of standard deviation values
3. MU_VAL : vector different values of $\mu$ considered
4. BI_MEAN : vector of different values of $\Delta$ considered
5. e_range : vector of range of $\epsilon$ values
6. NUM_AGENTS_INF : Number of agents influenced to change the election outcome using minimum strategy (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
7. P_INI : Initial number of votes of '+1' party (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
8. N_INI : Initial number of votes of '-1' party (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
9. MU : Average number of agents influenced to change the election outcome using random strategy over 30 independent choices of random selection of agents selected in priority (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
10. SIGMA : Standard deviation of the number of agents influenced to change the election outcome using random strategy over 30 independent choices of a random selection of agents selected in priority (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))

<a id="Proportion"></a>
### Proportion()

Function to initiate and parallelize the computation of the effort needed to change the election outcome.

OUTPUT

The necessary results of the computations are saved in a 'Min_Diffprop_'+str(sd_num)+'_'+str(bi_mean_num)+'.npz' file, along with index of standard deviation ('sd_num') and polarization parameter ('bi_mean_num') within the temporarily created folder named 'FILES'. The files contains
1. NOS : number of simulations
2. SD : vector of standard deviation values
3. BI_MEAN : vector of different values of $\Delta$ considered
4. Array_pos: vector of different numbers of '+1' voters considered in the natural opinions
5. Array_neg: vector of different numbers of '-1' voters considered in the natural opinions
6. e_range : vector of range of $\epsilon$ values
7. NUM_AGENTS_INF : Number of agents influenced to change the election outcome using minimum strategy (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
8. P_INI : Initial number of votes of '+1' party (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
9. N_INI : Initial number of votes of '-1' party (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
10. MU : Average number of agents influenced to change the election outcome using random strategy over 30 independent choices of random selection of agents selected in priority (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))
11. SIGMA : Standard deviation of the number of agents influenced to change the election outcome using random strategy over 30 independent choices of a random selection of agents selected in priority (matrix of dimension (number of $\mu$ values, number of simulations, number of $\epsilon$))

<a id="main"></a>
### if __name__=="__main__"

The main function for all the initialization and function calls to initiate the computations.

## Final files generated after the execution of the program:

Two folders named 'DATA' and 'FILES' are created during the execution of the program to temporarily save the intermediate results and delete them after the aggregation of necessary results.

1. Polarization_delta.npz : The file generated after the computation of the effort needed to change the election outcome by varying the polarization parameter $\Delta$. The file contains the number of agents in the system (Num_ppl), different values of $\epsilon$ considered (e_range), the strength of external influence applied (strength_w0), the standard deviation of the natural opinion distribution (SD), different values of $\Delta$ considered (DELTA_vals), different values of $\mu$ considered (MEAN_vals), the effort needed to change the election outcome using minimum strategy(EFFORT,  with the normalization of strength of influence), number of agents influenced to change the election outcome using minimum strategy (NUM_AGENTS_INFLUENCED), average effort needed to change the election outcome using random strategy over 30 independent choices of random selection of agents selected in priority (EFFORT_RAND, with the normalization of strength of influence).

2. Mean_variation.npz : The file generated after the computation of the effort needed to change the election outcome by varying the shift of the distribution $\mu$. The file contains the number of agents in the system (Num_ppl), different values of $\epsilon$ considered (e_range), the strength of external influence applied (strength_w0), the standard deviation of the natural opinion distribution (SD), different values of $\Delta$ considered (DELTA_vals), different values of $\mu$ considered (MEAN_vals), the effort needed to change the election outcome using minimum strategy(EFFORT,  with the normalization of strength of influence), number of agents influenced to change the election outcome using minimum strategy (NUM_AGENTS_INFLUENCED), average effort needed to change the election outcome using random strategy over 30 independent choices of random selection of agents selected in priority (EFFORT_RAND,  with the normalization of strength of influence).

3. Proportion_variation.npz : The file generated after the computation of the effort needed to change the election outcome by varying the proportion of agents in each party $p$. The file contains the number of agents in the system (Num_ppl), different values of $\epsilon$ considered (e_range), the strength of external influence applied (strength_w0), standard deviation of the natural opinion distribution (SD), different values of $\Delta$ considered (DELTA_vals), different values of $\mu$ considered (MEAN_vals), effort needed to change the election outcome using minimum strategy(EFFORT,  with the normalization of strength of influence), number of agents influenced to change the election outcome using minimum strategy (NUM_AGENTS_INFLUENCED), the number of agents supporting '+1' party (Array_pos), the number of agents supporting '-1' party (Array_neg), average effort needed to change the election outcome using  random strategy over 30 independent choices of random selection of agents selected in priority (EFFORT_RAND,  with the normalization of strength of influence).