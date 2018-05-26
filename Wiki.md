# Collaborative Design [Race Car]
This project is an ongoing research effort in the [hrc^2 Lab](hrc2.io) at Cornell University investigating collaborative design with humans and an intelligent agent. _Add information about the task here or remove if we don't actually need it in the repo_

##Environment
We use an 

##Reinforcement Learning Agent

##User Design Platform

##Server Backend

##Genetic Algorithm
To explore the vast design space, we used a python implementation of NSGAII included in [Platypus](https://github.com/Project-Platypus/Platypus). We use a parameterized version of the car as our variables and a combination of the components that the OpenAI gym environment uses to determine reward as well as some self-defined costs. 

Specifically, we maximize the standard reward we used to train the driving agent, we minimize the Fuel spent and the time spent on Grass Tiles (off track), and the overall cost of the car, which we defined.