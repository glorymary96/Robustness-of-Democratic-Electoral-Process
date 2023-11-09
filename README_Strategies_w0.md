# [Strategies_w0.py](#Strategies-w0)

Program to arbitrarily define the influence vector supporting each of the parties. 

## Functions in Strategies_w0.py

1. [Strategies](#strategies)

<a id ="Strategies-w0"></a>
## Detailed description of the functions :

<a id ="strategies"></a>
### Strategies(Num_party, winner_to_be)

Function to determine the influence vector depending on the next winner.

INPUT

1. Num_party : Number of parties involved in the election.
2. winner_to_be : the index of next-winner-to be party.

OUTPUT

1. w0 : the influence vector in support of the next winner (vector of size number of parties).