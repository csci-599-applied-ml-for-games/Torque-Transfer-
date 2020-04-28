# Torque Transfer

Contains code for Torque Transfer, USC CSCI 599 Spring 2020.

Github link: [https://github.com/csci-599-applied-ml-for-games/Torque-Transfer-](https://github.com/csci-599-applied-ml-for-games/Torque-Transfer-) <br/>

## Objective- <br />
Apply Transfer Learning along with Reinforcement Learning: <br/>
The goal of our project is to demonstrate how Transfer Learning can be used across simulators and eventually in a real life environment. <br />
Using knowledge gained in one self driving environment, we train the same agent on a similar environment and we compare the learning curves for agent with transfer learning and that without. <br />

## Methodology- <br/>
### TORCS: <br/>
**Sensor Input**: MLP as a policy network using PPO that uses sensor state from the input in order to generate actions (such as throttle and steering). Uses curriculum learning from simple tasks to more complex ones. <br />
Image Input: Navigates using image result as input to a CNN, with only steering as the action. PPO is most effective but tends to prioritize “hacking” the reward by avoiding long steps and using shortcuts. <br />
**Imitation Learning**: Interfaces sensor input to image input to avoid shortcuts and predict the driver action using a CNN. Error function is MSE, masters game after 50 epochs. <br />
As the Image forward is same as Image backward. This results in state space aliasing. PPO critic only takes state as input, hence gets confused while predicting advantage. This causes fall in learning curve. <br/>

### Donkey Car Sim: <br />
To train the Donkey Car, same model was used as in TORCS but with DDQN as the RL algorithm. <br />
One model was trained entirely from ground up with the model architecture as described before. <br />
Another model was trained using a transfer learning based agent where the agent was pre-trained on TORCS. <br />
The transfer learning was implemented by replacing just the final Dense layer and allowing all layers to be trainable. <br />

## Results- <br />
The agent which used Transfer Learning from TORCS to Donkey Car Simulator performed better in the following ways: <br/>
1. Has a higher average reward. <br />
2. Takes fewer episodes to train. <br />
3. The agent using transfer learning has way more stability as compared to the agent which trained from ground up. <br />

## The following video shows the demo of our project with explanation- <br />

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/184LlwAaF-4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Here is the link to the paper- <br />
[https://drive.google.com/file/d/1szvbnzQd4vPkF-I6dM8p47iKH5mu0oIF/view?usp=sharing](https://drive.google.com/file/d/1szvbnzQd4vPkF-I6dM8p47iKH5mu0oIF/view?usp=sharing) <br/>

## Contributors: <br/>
Shashank Hegde - [https://www.linkedin.com/in/karkala-shashank-hegde/](https://www.linkedin.com/in/karkala-shashank-hegde/) <br/>
Sriram Ramaswamy - [https://www.linkedin.com/in/sriramvera/](https://www.linkedin.com/in/sriramvera/) <br/>
Sumeet Bachani - [https://www.linkedin.com/in/sumeetbachani/](https://www.linkedin.com/in/sumeetbachani/) <br/>
Tushar Kumar - [https://www.linkedin.com/in/tushartk/](https://www.linkedin.com/in/tushartk/) <br/>
