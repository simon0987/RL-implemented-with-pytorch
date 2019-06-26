import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import logging

from agent_dir.agent import Agent
from environment import Environment
from replay_memory import ReplayMemory,Transition

use_cuda = torch.cuda.is_available()
outputFile = open('dqn.csv', 'w+')

class DQN(nn.Module):
    '''
    This architecture is the one from OpenAI Baseline, with small modification.
    '''
    def __init__(self, channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.lrelu(self.fc(x.view(x.size(0), -1)))
        q = self.head(x)
        return q

class AgentDQN(Agent):
    def __init__(self, env, args):
        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        # TODO:
        # Initialize your replay buffer
        self.memory = ReplayMemory(10000)
        # build target, online network
        self.target_net = DQN(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = DQN(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('dqn')
        
        # discounted reward
        self.GAMMA = 0.99
        
        # training hyperparameters
        self.train_freq = 4 # frequency to train the online network
        self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
        self.batch_size = 32
        self.num_timesteps = 3000000 # total training steps
        self.display_freq = 10 # frequency to display training progress
        self.save_freq = 200000 # frequency to save the model
        self.target_update_freq = 1000 # frequency to update target network

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)

        self.steps = 0 # num. of passed steps. this may be useful in controlling exploration


    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        # TODO:
        # At first, you decide whether you want to explore the environemnt
        sample = random.random()
        # TODO:
        # if explore, you randomly samples one action
        # else, use your model to predict action

        eps_threshold = 0.05 + (0.9-0.05) * math.exp(-1. * self.steps / 200)
        if test == True:
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state

            return self.online_net(state).max(1)[1].view(1,1).item()
        else:
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.online_net(state).max(1)[1].view(1,1)
            else:
                return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long).cuda()

    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        one_batch = Transition(*zip(*transitions))

        state_batch = torch.cat(one_batch.state)
        action_batch = torch.cat(one_batch.action)
        reward_batch = torch.cat(one_batch.reward)
        # TODO:
        # Compute Q(s_t, a) with your model.
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          one_batch.next_state)), dtype=torch.uint8).cuda()
        non_final_next_states = torch.cat([s for s in one_batch.next_state
                                                if s is not None])
        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values = torch.zeros(self.batch_size).cuda()
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        # TODO:
        # Compute temporal difference loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        episodes_done_num = 0 # passed episodes
        total_reward = 0 # compute average reward
        loss = 0 
        while(True):
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while(not done):
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                total_reward += reward

                reward = torch.tensor([reward]).cuda()
                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                self.memory.push(state,action,next_state,reward)
                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save('dqn')

                self.steps += 1

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                if outputFile:
                    outputFile.write(str(episodes_done_num) + ","+ str(self.steps)+ "," + \
                        str(total_reward/self.display_freq) + " \n")
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save('dqn')