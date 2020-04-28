
# Torque Transfer

Contains code for Torque Transfer, USC CSCI 599 Spring 2020.

Github link: [https://github.com/csci-599-applied-ml-for-games/Torque-Transfer-](https://github.com/csci-599-applied-ml-for-games/Torque-Transfer-) <br/>

## Objective <br />
The main objective is to apply Transfer Learning using Reinforcement Learning at the core.<br/>
The goal is to demonstrate how Transfer Learning can be used across simulators and eventually in a real life environment. Using knowledge gained in one self driving environment, we train the same agent on a similar environment and we compare the learning curves for agent with transfer learning and that without. <br />

## Methodology <br/>
### TORCS: <br/>
1. **Sensor Input**: MLP as a policy network using PPO that uses sensor state from the input in order to generate actions (such as throttle and steering). Uses curriculum learning from simple tasks to more complex ones. <br />
2. **Image Input**: Navigates using image result as input to a CNN, with only steering as the action. PPO is most effective but tends to prioritize “hacking” the reward by avoiding long steps and using shortcuts. <br />
3. **Imitation Learning**: Interfaces sensor input to image input to avoid shortcuts and predict the driver action using a CNN. Error function is MSE, masters game after 50 epochs. <br /> As the Image forward is same as Image backward. This results in state space aliasing. PPO critic only takes state as input, hence gets confused while predicting advantage. This causes fall in learning curve. <br/>

### Donkey Car Sim: <br />
1. **Image Input**: The image dimesion is kept the same as TORCS and a similar track is also chosen to provide similar environment for training. <br />
2. **Ground up training**: One model was trained entirely from ground up with the model architecture as described before. <br />
3. **Transfer Learning based training**: Another model was trained using a transfer learning based agent where the agent was pre-trained on TORCS. The transfer learning was implemented by replacing just the final Dense layer and allowing all layers to be trainable. <br />

## Results <br />
The agent which used Transfer Learning from TORCS to Donkey Car Simulator performed better in the following ways: <br/>
1. Has a higher average reward. <br />
2. Takes fewer episodes to train. <br />
3. Has better stability in when driving in test mode.<br />

### The following video shows the demo of our project with explanation: <br />

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/184LlwAaF-4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Here is the link to the paper: <br />
[https://drive.google.com/file/d/1szvbnzQd4vPkF-I6dM8p47iKH5mu0oIF/view?usp=sharing](https://drive.google.com/file/d/1szvbnzQd4vPkF-I6dM8p47iKH5mu0oIF/view?usp=sharing) <br/>

### Here is the link to the presentation: <br />
[https://drive.google.com/file/d/1JZsqM1Tf6chaGBK5BR000iyisoCKoxu6](https://drive.google.com/file/d/1JZsqM1Tf6chaGBK5BR000iyisoCKoxu6) <br/>

## Contributors <br/>
Shashank Hegde - [https://www.linkedin.com/in/karkala-shashank-hegde/](https://www.linkedin.com/in/karkala-shashank-hegde/) <br/>
Sriram Ramaswamy - [https://www.linkedin.com/in/sriramvera/](https://www.linkedin.com/in/sriramvera/) <br/>
Sumeet Bachani - [https://www.linkedin.com/in/sumeetbachani/](https://www.linkedin.com/in/sumeetbachani/) <br/>
Tushar Kumar - [https://www.linkedin.com/in/tushartk/](https://www.linkedin.com/in/tushartk/) <br/>
