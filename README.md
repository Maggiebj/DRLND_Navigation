# DRLND_Navigation
## Project Description
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.
![banana](banana.gif)
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation
The project was in a linux server with unityagent=0.4.0 and python 3.6 installed.

1. You may need to install Anaconda and create a python 3.6 environment.
```bash
conda create -n drnd python=3.6
conda activate drnd
```
2. Clone the repository below, navigate to the python folder and install dependencies. Pay attention that the torch=0.4.0 in the requirements.txt is no longer listed in pypi.org, you may leave your current torch and remove the line torch=0.4.0
 ```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```
3. Download unity environment file  [banana](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip). This is the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.
(To watch the agent, you may follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the [environment for the Linux operating system](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip).)

5. Unzip the compressed file
6. Create the Ipython kernel:
```bash
python -m ipykernel install --user --name=drnd
```

   
## Executing 
In the notebook, be sure to change the kernel to match "drnd" by using the drop down in "Kernel" menu. Be sure the adjust the Banana file location locally.

Executing Navigation.ipynb
  
