# Robustness-of-Democratic-Electoral-Process

There have been recent reports of systematic attempts to influence the outcomes of democratic elections externally. For example, there are reports on Russian interference in the US elections. With different electoral systems being used across democratic countries, a question that naturally arises is whether
certain electoral systems are more robust against such external influences than others. Our goal is to compare the effects of systematic external efforts to modify an agent’s opinion on different electoral systems and rank the latter according to their relative robustness.

Building on earlier works in analytical social sciences, we adapt a mathematical model of opinion dynamics, where voters are represented by agents interacting with one another. In the opinion dynamics
model, generally, the agents reconsider their opinion based on interaction with other agents. As time evolves, they forget their initial opinion to reach a common consensus, which is unrealistic in the real world. We adapted the model to prevent this unrealistic behavior by introducing a natural opinion which
is the opinion that each agent would have if they did not interact with other agents.

Agents tend to interact with other agents of similar opinion; this is parameterized as confidence bound. In general, this determines the open-mindedness of society. This idea of confidence bound
is inspired from the well-established ‘bounded confidence model’

Electoral systems are the set of rules and procedures used to select representatives for a political body. They vary widely from country to country but typically involve some form of voting by the country’s citizens. The election result is determined by an outcome function that depends on the considered electoral system, such as proportional or single representative, winner-takes-all, and so forth. We categorize our study into a two-party and multi-party electoral system.

An external attack is introduced in the system as an influence field to change the election’s outcome. We then extract the total effort the field of influence has to exert to change the electoral outcome as a function of agent polarization, the ability to interact with agents of a different opinion, and the number of existing political parties.

Any model is verified to predict the actual trend only when some historical data validate it. From the literature survey, we observed that opinion dynamics models should be validated against actual data. However, there are only a few works on validation.

Here we calibrated and validated our model against the historical data of the US House of Representative elections for different probability distributions of natural opinion. We employ the volatility (i.e., how easily a district can swing) of the districts of the US to validate our model, roughly what happens in an actual election. From the validation, we attained a good correlation between our simulation and actual data, and we concluded that the model captures the robustness of electoral systems.

We present a detailed study on the bipartite elections to evaluate the system’s robustness and extend it to a multi-partite system. From the results, we conclude that in a bipartite system, the openness of the society makes it harder to change the election’s outcome, and the system’s robustness decreases with polarization.

We extend the model to a multi-partite system, such that the opinion of agents lies in a simplex. In addition to this extension, we consider some additional conditions on the natural opinion we believe in having for an opinion of a sane agent. The additional conditions provide a genuine meaning to each agent’s opinion and significantly affect the system’s dynamics. From consensus-type dynamics, it is proved in the literature that the opinion of agents comes closer to the opinion in the middle in a steady-state configuration due to high interaction. However, this direct result is proved to be faulty without these
conditions.

The system’s robustness as a function of the openness of society varies from a bipartite to a multi-partite system. For a multi-partite system, the effort needed to change the election’s outcome gets higher when agents interact more with other agents until it saturates. Further increase in interaction among the agents results in a much easier swing from runners-up to the winner. This is a direct consequence of the position of the runners-up in the left-right spectrum of parties.

Proportional electoral systems are generally the most robust irrespective of the number of parties in the systems. The electoral system for a multi-partite system is more robust when there are moderate interaction between the agents. Thus making the electoral system more resilient to external attack from
foreign agencies or social media information.

Summary of the files

1. HOR_DATA : Contains the data of House of Representative elections (2012-2020) for validation and case study and geodata for the plots.
2. Model_execution.py : A simple execution of the model.
3. Even_distribution_val.py : Program for the validation of the model using House of Representative elections (2012-2020) of United States.
4. Polarization_bias.py : Program to study the effection of polarization and bias on the effort needed to change the election outcome.
5. Robustness_biparty.py : Program to evaluate the robustness of different electoral system to external influences for a two-party electoral system for a generic case and case study on US House of Representative elections.
6. Nat_opn_generator.py : Program to generate the natural opinion with equal density and volume for each party.
7. Strategies_w0.py : Program to arbitarily define the influence vector supporting each of the party.
8. Robustness_multiparty_single_electoral_unit.py : Program to implement the robustness of the electoral system in a single electoral unit for different values of confidence bound parameter.
9. Electoral_system_function.py : A program with functions to calculate the effort needed to change the election outcome following different electoral systems.
10. Robustness_multiparty_multiple_electoral_unit.py : Program to find the effort needed to change the election outcome following different electoral systems using different sub-programs within the program.