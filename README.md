# Collaborative Design [Race Car]
This project is an ongoing research effort in the [hrc^2 Lab](hrc2.io) at Cornell University investigating collaborative design with humans and an intelligent agent. _Add information about the task here or remove if we don't actually need it in the repo_

## Installation

### Dependencies
Make sure to install:
* swig `sudo apt install swig`
* cairo and dependencies `sudo apt install libcairo2-dev pkg-config python3-dev`
* pycurl (the pip wheel is broken, to install on Ubuntu 18.04+, `sudo apt install python3-pycurl` and delete from requirements.txt)

## Environment
We use OpenAI Gym's Car-Racing-v0 Environment as the base of our environment. In the provided environment, we have augmented the Controlling and Dynamics modules to work with car configurations that we provide from either a Genetic Algorithm or from the user. To do this we had to define an intermediate car configuration, write parsing methods to transition from an image to our configuration spec, and then translate the configuration into a Car class that would interact with the provided environment. 

The new environment is called Car-Racing-v1 and can be instantiated in the same way that any OpenAI Gym environment is with the make function (provided the gym module being called is our modified version).

## Reinforcement Learning Agent
A core component of our testing is to build a solution to the Car-Racing environment. The solution used a q-learning approach with a simple Artificial Neural Network with one hidden layer which is feed in a single frame from the environment and returns an action to take (we limit the action space to help speed up training). The Neural Network acts as a q-table where instead of looking up q values, the q values are outputs from the Neural Network. Our solution performs well on simple tracks but performs poorly on some edge cases.

Training was done over 12 hours on a  2.9 GHz Intel Core i5 CPU with 16 GB of RAM. 

## User Design Platform
We made a react app to allow the user to define their own car configuration. _describe the react app here_

## Server Backend
_describe the server here_

## Genetic Algorithm
To explore the vast design space, we used a python implementation of NSGAII included in [Platypus](https://github.com/Project-Platypus/Platypus). We use a parameterized version of the car as our variables and a combination of the components that the OpenAI gym environment uses to determine reward as well as some self-defined costs. 

Specifically, we maximize the standard reward we used to train the driving agent, we minimize the Fuel spent and the time spent on Grass Tiles (off track), and the overall cost of the car, which we defined.
