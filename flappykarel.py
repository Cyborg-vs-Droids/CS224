#######################################################################
# Copyright (C)                                                       #
# Niraj Rajai- nirajrajai12@gmail.com                                 #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

class FlappyKarel:
    
    def __init__(self,grid_size,negative_states,goal_states,neg_reward,goal_reward,step_reward,initial_state,discount_factor,epsilon):
        self.grid_height = grid_size[0]
        self.grid_width = grid_size[1]
        self.grid = pd.DataFrame([list(range(i*self.grid_height+1,(i+1)*self.grid_height+1)) for i in range(0,self.grid_width)]).T
        self.rewards = self.grid.copy()
        self.negative_states = negative_states
        self.goal_states = goal_states
        self.initial_state = initial_state
        self.prev_state = self.initial_state
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_actions = 2

        self.state_to_col_mapping = {val:int((val-1)/self.grid_height) for val in range(1,self.grid_height*self.grid_width+1)}

        for col in self.grid.columns:
            self.rewards.loc[self.grid[col].isin(negative_states),col] = neg_reward
            self.rewards.loc[self.grid[col].isin(goal_states),col] = goal_reward
            self.rewards.loc[~self.grid[col].isin(negative_states+goal_states),col] = step_reward
    
    def step(self,action):
        curr_col = self.state_to_col_mapping[self.prev_state]
        curr_index = self.grid.loc[self.grid[curr_col]==self.prev_state,curr_col].index[0]

        next_col = curr_col
        next_index = curr_index

        if action == 0: # move right up
            next_col = curr_col+1
            next_index = curr_index-1
            if (next_col > self.grid_width-1)|(next_index<0):
                next_col = curr_col
                next_index = curr_index + 1
                if next_index>self.grid_height-1:
                    next_index = curr_index
                
        
        elif action == 1: # move right down
            next_col = curr_col+1
            next_index = curr_index+1
            if (next_col > self.grid_width-1)|(next_index>self.grid_height-1):
                next_col = curr_col
                next_index = curr_index + 1
                if next_index>self.grid_height-1:
                    next_index = curr_index
        
        

        next_state = self.grid.loc[next_index,next_col]
        reward = self.rewards.loc[next_index,next_col]

        end_state = False

        if (next_state in self.negative_states) | (next_state in self.goal_states):
            end_state = True

        prev_state = self.prev_state
        self.prev_state = next_state

        return prev_state,next_state,reward,end_state

    def select_action(self,state):
        
        action = np.random.choice(np.array(range(0,self.num_actions)),p=self.policy[state,:])
        # if np.random.uniform()<self.epsilon:
        #     # Ties arbitrarily broken
        #     action = np.random.randint(0,self.num_actions)
        # else:
        #     all_actions = np.array(range(0,self.num_actions))
        #     action = np.random.choice(all_actions[self.policy[state,:]==self.policy[state,:].max()])
            

        return action


    def montecarlo(self,episodes=100,steps=200):

        self.policy = np.full((self.grid_height*self.grid_width+1,self.num_actions),1.0/self.num_actions)
        self.returns_dict = {state:{action:[] for action in range(self.num_actions)} for state in range(1,self.grid_height*self.grid_width+1)}
        self.q_value = {state:{action:0 for action in range(self.num_actions)} for state in range(1,self.grid_height*self.grid_width+1)}

        rewards = []

        for episode in tqdm(range(episodes)):
            
            
            self.visit_dict = {state:{action:False for action in range(self.num_actions)} for state in range(1,self.grid_height*self.grid_width+1)}

            self.prev_state = np.random.choice(list(set(range(1,self.grid_height*self.grid_width+1))-set(self.negative_states+self.goal_states)))

            state_action_reward = []

            for step in range(steps):
                action = self.select_action(self.prev_state)
                prev_state,next_state,reward,end_state = self.step(action)
                #print(prev_state,action,next_state,reward)
                state_action_reward.append([prev_state,action,reward])
                
                if end_state:
                    break
            
            expected_reward = 0

            for state,action,reward in state_action_reward[::-1]:

                expected_reward = self.discount_factor*expected_reward + reward

                if not self.visit_dict[state][action]:
                    temp_ls = self.returns_dict[state][action]
                    temp_ls.append(reward)
                    self.returns_dict[state][action] = temp_ls
                    self.q_value[state][action] = np.mean(self.returns_dict[state][action])
                    a_star = np.argmax(list(self.q_value[state].values()))

                    for a in range(self.num_actions):
                        if a==a_star:
                            self.policy[state,a] = 1 - self.epsilon + self.epsilon/self.num_actions
                        else:
                            self.policy[state,a] = self.epsilon/self.num_actions
                    
            
            rewards.append(np.sum([reward for _,_,reward in state_action_reward]))
        
        sns.lineplot(x=list(range(episodes)),y=pd.Series(rewards)).get_figure().savefig('output.png')
    
    def select_action_q(self,state,epsilon):
        if np.random.uniform()<epsilon:
            # Ties arbitrarily broken
            action = np.random.randint(0,self.num_actions)
        else:
            all_actions = np.array(range(0,self.num_actions))
            q_values_state = np.array(list(self.q_value[state].values()))
            action = np.random.choice(all_actions[q_values_state==q_values_state.max()])
        return action

    def qlearning(self,episodes,steps,step_size=0.1):

        #self.policy = np.full((self.grid_height*self.grid_width+1,self.num_actions),1.0/self.num_actions)
        self.q_value = {state:{action:0 for action in range(self.num_actions)} for state in range(1,self.grid_height*self.grid_width+1)}
        rewards = []
        for episode in tqdm(range(episodes)):
            self.prev_state = np.random.choice(list(set(range(1,self.grid_height*self.grid_width+1))-set(self.negative_states+self.goal_states)))
            
            
            for step in range(steps):
                action = self.select_action_q(self.prev_state,self.epsilon)
                prev_state,next_state,reward,end_state = self.step(action)
                #print(prev_state,action,end_state,self.q_value[prev_state])
                self.q_value[prev_state][action] = self.q_value[prev_state][action] + step_size*(reward + self.discount_factor*max(list(self.q_value[next_state].values()))-self.q_value[prev_state][action])
                
                if end_state:
                    break
            self.prev_state = self.initial_state
            episode_reward = 0
            for step in range(steps):
                action = self.select_action_q(self.prev_state,epsilon=0)
                prev_state,next_state,reward,end_state = self.step(action)
                episode_reward += reward
                if end_state:
                    break
            rewards.append(episode_reward)
        
        sns.lineplot(x=list(range(len(rewards))),y=pd.Series(rewards))
    
    def printoptimalpath(self,start_state,steps=50):
        self.prev_state = start_state
        for i in range(steps):
            action = self.select_action_q(self.prev_state,epsilon=0)
            prev_state,next_state,reward,end_state = self.step(action)
            print(prev_state,' ----> ',next_state)

            if end_state:
                break
            



