# SEINR
In recent years, the global epidemiology of infectious diseases has posed a significant threat to human health, with outbreaks like the new coronavirus, influenza, monkeypox, and others continuing to affect populations worldwide. While the incidence rates of many infectious diseases have decreased, they remain a major concern. Since the 1970s, numerous new infectious diseases have emerged, leading to outbreaks and epidemics that have claimed thousands of lives in a short span of time. To mitigate the spread of these diseases, this paper focuses on simulating the outbreak dynamics of infectious diseases by establishing the SEIR model and, innovatively, the SEINR model.

The SEIR model involves four states: susceptible (S), latent (E), infected (I), and recovered (R) individuals. The total population, N, is given, and an initial number of recovered individuals is set. A system of differential equations is formulated using a logistic model. However, the analytical solution is elusive and can only be obtained numerically, so our primary focus is on the SEINR model.

The SEINR model builds upon the SEIR model by introducing the state of asymptomatic infection, making it more suitable for modeling diseases like the new coronavirus. In this model, there are five states: susceptible (S), pre-symptomatic (E), asymptomatic infected (N), symptomatic infected (I), and recovered (R) individuals. Similar to the SEIR model, susceptible and pre-symptomatic individuals transition between these states after contact. Pre-symptomatic individuals progress to symptomatic infection after an incubation period, at which point they are no longer considered, being isolated and non-infectious. Apart from specifying these states, we need to set parameters like daily contact rates and cure rates.

Using this information, a set of differential equations is formulated to simulate the outbreak of infectious diseases. We then plot curves representing the number of infected individuals over time for various models, including SI, SIS, SIR, SEIR, and SEINR. We also visualize the dynamics of other states over time. Subsequently, we analyze the differences between the models and highlight the advantages of the innovative SEINR model. We offer a comprehensive explanation of the changes in the SEINR model states.

Further analysis focuses on the benefits of the SEINR model, contrasting it with other models and detailing the patterns within each state of SEINR. We then predict key epidemic milestones, such as the peak and end of the outbreak, as well as the percentage of uninfected people.

To enhance our understanding, we perform a sensitivity analysis on the SEIR model, adjusting parameters like the daily cure rate and contact rate. This allows us to assess the impact of different parameters on the SEINR model. Based on the nature of these parameters, we propose strategies for preventing the spread of infectious diseases. Finally, we conduct a thorough analysis of the model's strengths and weaknesses.
