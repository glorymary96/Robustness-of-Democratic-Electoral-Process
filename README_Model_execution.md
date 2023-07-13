### Manual for the code 'Model_execution.py'

Running this script runs an example of the opinion formation simulation with choice of parameters.

## Functions in Model_execution.py
    1. outcome
    2. type_natural_opinion

===============================================================================
# Detailed description of the functions:

1. outcome(x0, epsilon, W)*
	Computes the final opinions from the natural opinion x0, the communication distance epsilon, and the influence W. 
	INPUT:
		1. x0 (vector, matrix): natural opinions
		2. epsilon (float): communication distance
		3. W (vector, matrix): influence vector/matrix
	OUTPUT:
		1. y (vector, matrix): final opinions

===============================================================================
2. type_natural_opinion(Num_ppl, Num_party, NAT_TYPE)*
	Generates the natural opinions by calling the appropriate function. If the opinion is "Bigaussian", the mean, Delta, sd parameters have to be specified beforehand in the local environment. If Num_party is 2, NAT_TYPE can be 'Bigaussian' or 'Uniform', otherwise it can only be 'Uniform'. 'Uniform' draws the opinions from the appropriate simplex described in the manuscript's SI. 
	INPUT:
		1. Num_ppl (int): number of agents
		2. Num_party (int): number of parties
		3. NAT_TYPE (string): type of distribution from which the opinions will be drawn 
	OUTPUT: 
		x0 (vector, matrix): natural opinions

===============================================================================
===============================================================================