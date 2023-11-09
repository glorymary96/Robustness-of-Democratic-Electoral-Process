# [Nat_opn_generator.py](#Nat_opn_generator)

Program to generate the natural opinion with equal density and volume for each party for a multi-party electoral system, and as per the parameters specified for a two-party system following a bi-gaussian distribution. This Python code is generally called from the execution of other Python codes for calculating the robustness of the electoral system, and the matrix of the natural opinion of agents is returned.

## Functions in Nat_opn_generator.py
1. [admissible_summits](#admissible-summits)
3. [gen_rand_perm_vect](#gen-rand-perm-vect)
4. [Nat_opn_2](#Nat-opn-2)
5. [rand_opinion](#rand-opinion)
6. [rand_opinion_pos](#rand-opinion-pos)
7. [Random_generation](#random-generation)
8. [Random_generation_pos](#random-genertaion-pos)
9. [slide_summit_vec](#slide-summit-vec)
10. [unit_simplex_arb](#unit-simplex-arb)
11. [vertices](#vertices)
    

<a id="Nat_opn_generator"></a>
## Detailed description of the functions :'Nat_opn_generator.py'
To generate the natural opinions depending on the number of parties in the electoral system.

<a id="admissible-summits"></a>
### admissible_summits(p)
Defines the vertices of the admissible opinion space.

INPUT
1. p (int): number of parties
   
OUTPUT
1. V (matrix): vertices of the admissible opinion space

<a id="gen-rand-perm-vect"></a>
### gen_rand_perm_vect(x)
Generates a random permutation of the vector x.

INPUT
1. x (vector)
   
OUTPUT
1. p (vector)

<a id="Nat-opn-2"></a>
### Nat_opn_2(mu, Delta, sd, num_ppl, R=50)
Generates gaussian and bigaussian random natural opinion for 2 parties

INPUT
1. mu (float): mean of the whole distribution.
2. Delta (float): polarization coefficient, i.e., distance between the gaussian peaks.
3. sd (float): standard deviation of the Gaussian(s).
4. num_ppl (int): number of opinions to be generated.
5. R (float): percentage of agents in each peak.
   
OUTPUT
1. x0 (matrix): array of natural opinions (num_ppl,2)

<a id="rand-opinion"></a>
### rand_opinion(p, V)
Draws one random opinion uniformly in the admissible space. The of vertices is already given in 'V'. One can get 'V' by using the function 'admissible_summits'. 

INPUT

1. p (int): number of parties.
2. V (matrix): vertices of the admissible space.
   
OUTPUT

1. x (matrix): random opinion

<a id="rand-opinion-pos"></a>
### rand_opinion_pos(p, V, k)
Same as 'rand_opinion' with a fixed party k.

INPUT

1. p (int): number of parties.
2. V (matrix): vertices of the admissible space.
3. k (int): index of the party where the opinion is drawn.

OUTPUT
1. x (matrix): random opinion
   
<a id="random-generation"></a>
### Random_generation(num_ppl, p)
Generates 'num_ppl' random opinions with p parties.

INPUT
1. num_ppl (int): number of agents.
2. p (int): number of parties.
   
OUTPUT

1. X (matrix): random opinions

<a id="random-genertaion-pos"></a>
### Random_generation_pos(num_ppl, p, k)
Generates 'num_ppl' random opinions in party k among p parties.

INPUT
1. num_ppl (int): number of agents.
2. p (int): number of parties.
3. k (int): index of the party.
   
OUTPUT
1. X (matrix): random opinions

<a id="slide-summit-vec"></a>
### slide_summit_vec(x, a)
Slides the point x (typically in the simplex) towards the barycenter of the unitary p-simplex. For a = 1, the summit does not move, and for a = 0, the summit reaches the barycenter.

INPUT

1. x (vector): point to be moved towards the barycenter of the simplex.
2. a (float): factor by which the point is slided.
   
OUTPUT

1. y (vector): moved point

<a id="unit-simplex-arb"></a>
### unit_simplex_arb(S)
Draws a random point uniformly from an arbitrary simplex, defined by the columns of S. Follows the idea presented in doi.org/10.13140/RG.2.1.3807.6968.

INPUT

1. S (matrix): list of vertices of the simplex.
   
OUTPUT

1. x (vector): random point in S.

<a id="vertices"></a>
### vertices(p)
Generates the list of vertices of interest in the unitary simplex with 'p' vertices.

INPUT
1. p (int): number of parties.

OUTPUT
1. vertex (matrix): list of the simplex's vertices.
