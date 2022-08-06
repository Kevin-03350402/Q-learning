
import argparse
import numpy as np
from numpy import random
from environment import MountainCar, GridWorld
import sys
from matplotlib import pyplot as plt
"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 


        make sure that this is turned off when you submit to the autograder.



        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should 
        
        make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *not* have the bias term 
folded in.
"""

def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate

# a function that approximate like q(s,a;w), find best action
def q(state,weight):
    # find the values for all actions
    actions_value = weight@state
    # find the action with highest value
    best_action = np.argmax(actions_value)
    return best_action

# calculate the q value
def qvalue(state,weight,a):
    return state@weight[a,:]

# epsilon_greedy
def epsilon_greedy(epsilon,best_action,total_possible_actions):
    prob = np.random.uniform(low=0.0, high=1.0)
    # with 1-epsilon prob, return best action
    if prob>=epsilon:
        return best_action
    # with epsilon prob, return a random action
    else:
        # generate an uniform random in all possible actions
        randome_action = np.random.randint(env.action_space)
        return randome_action
        
   
def moving_average(x, w):
    tmp =  np.convolve(x, np.ones(w), 'valid') / w
    last = tmp[-1]
    lastavg = last*np.ones(w-1)
    res = np.concatenate((tmp, lastavg))
    return res





if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    if env_type == "mc":
        env = MountainCar(mode = mode) # Replace me!
    elif env_type == "gw":
        env = GridWorld(mode = "tile" ) # Replace me!
    else: raise Exception(f"Invalid environment type {env_type}")
    
    
    

    # initialize the weight matrix with zeros(fold bias)

        
    weight_matrix = np.zeros((env.action_space, env.state_space+1))

    rewardtxt = open(returns_out,"w+")

    total_possible_actions = env.action_space

    rewardlist = []
    for episode in range(episodes):

        # Get the initial state by calling env.reset()
        ini_states = env.reset()
        # Append an element in the front of state to take care of the bias
        cur_states = np.insert(ini_states,0,1)
        reward_total = 0

        
        

        for iteration in range(max_iterations):
            best_action = q(cur_states,weight_matrix)
            real_action = epsilon_greedy(epsilon,best_action,total_possible_actions)


            # Select an action based on the state via the epsilon-greedy strategy

            # Take a step in the environment with this action, and get the 
            # returned next state, reward, and done flag
            next_state , reward, done = env.step(real_action)
          
            reward_total += reward
 


            # Using the original state, the action, the next state, and 
            # the reward, update the parameters. 
            
            # find the gradient first
            # the gradent wrt the matrix is a zero matrix of size(a*s) with ath row replaced by s
            
            grad_wq = np.zeros((env.action_space, env.state_space+1))
            # insert a zero to cur_states to take care of the bias term
            s = cur_states
            grad_wq [real_action,:] = s
           
            # find the max_action q(sprime,aprime)
            # engineer the next state (add the bias to the start):

            next_state = np.insert(next_state,0,cur_states[0])

            next_best_action = q(next_state,weight_matrix)
            nest_best_value = qvalue(next_state,weight_matrix,next_best_action)

            # caculate the alpha() part
            
            diff = lr*(qvalue(cur_states,weight_matrix,real_action)-(reward+gamma*nest_best_value))*grad_wq
            # update weight
            weight_matrix = weight_matrix-diff
            # append the bias term to the new state to take care of the bias
            
            cur_states = next_state

            # Don't forget to update the bias term!

            # Remember to break out of this inner loop if the environment signals done!
            if done:
                break
            pass
        rewardlist.append(reward_total)
        rewardtxt.write(f'{reward_total:.18e}\n')

    epsides = np.array(list(range(0,episodes)))
    rewardlist = np.array(rewardlist)
    print(rewardlist)
    rollingavg = moving_average(rewardlist, 25)
    
    
    
  
    rewardtxt.close()
    # Save your weights and returns. The reference solution uses 
    np.savetxt(weight_out,weight_matrix, fmt="%.18e", delimiter=" ")

    plt.plot(epsides,rewardlist,label="reward (raw)")
    plt.plot(epsides,rollingavg, label="rolling avg 25")
    leg = plt.legend(loc='upper center')
    plt.show()