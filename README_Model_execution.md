## Manual for the code 'Model_execution.py'

Model_execution.py
        outcome
        type_natural_opinion

===============================================================================
===============================================================================
**Model_execution.py**
Running this script runs an example of the opinion formation simulation with choice of parameters. 
===============================================================================
===============================================================================
*outcome(x0, epsilon, W)*
Computes the final opinions from the natural opinion x0, the communication distance epsilon, and the influence W. 
INPUT:
	x0 (vector, matrix): natural opinions
	epsilon (float): communication distance
	W (vector, matrix): influence vector/matrix
OUTPUT:
	y (vector, matrix): final opinions

===============================================================================
*type_natural_opinion(Num_ppl, Num_party, NAT_TYPE)*
Generates the natural opinions by calling the appropriate function. In the opinion is "Bigaussian", the mean, Delta, ds parameters have to be specified beforehand in the local environment. If Num_party is 2, NAT_TYPE can be 'Bigaussian' or 'Uniform', otherwise it can only be 'Uniform'. 'Uniform' draws the opinions from the appropriate simplex described in the manuscript's SI. 
INPUT:
	Num_ppl (int): number of agents
	Num_party (int): number of parties
	NAT_TYPE (string): type of distribution from which the opinions will be drawn 
OUTPUT: 
	x0 (vector, matrix): natural opinions

===============================================================================
===============================================================================