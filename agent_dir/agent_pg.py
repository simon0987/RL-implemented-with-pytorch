import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent_dir.agent import Agent
from environment import Environment

outputFile = open('pg.csv', 'w+')
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_num, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_num)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob

class AgentPG(Agent):
    def __init__(self, env, args):
        self.env = env
        self.model = PolicyNet(state_dim = self.env.observation_space.shape[0],
                               action_num= self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('pg.cpt')
        # discounted reward
        self.gamma = 0.99 
        
        # training hyperparameters
        # i change the episodes from 100000 -> 10000
        self.num_episodes = 10000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions ,self.saved_log_probs = [], [] ,[]
        self.step = 0
    
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        # I add a list to store log_probs
        self.rewards, self.saved_actions,self.saved_log_probs = [], [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical
        state = torch.from_numpy(state).float().unsqueeze(0)
        #state: (8)
        action_probs = self.model(state)
        m = Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        R = 0
        discounted_rewards = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0,R)
        # TODO:
        # compute loss
        loss = []
        discounted_rewards = torch.tensor(discounted_rewards)
        #normalize the rewards to 0~1
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) /\
            (discounted_rewards.std() + np.finfo(np.float32).eps)
        for log_prob, R in zip(self.saved_log_probs,discounted_rewards):
            loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()


    def train(self):
        avg_reward = None # moving average of reward
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while(not done):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)

                self.saved_actions.append(action)
                self.rewards.append(reward)
                self.step += 1
            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print('Epochs: %d/%d | Avg reward: %f '%
                       (epoch, self.num_episodes, avg_reward))
                if outputFile:
                    outputFile.write(str(self.step) + "," + \
                        str(avg_reward) + " \n")
            if avg_reward > 80: # to pass baseline, avg. reward > 50 is enough.
                self.save('pg.cpt')
                if outputFile:
                    outputFile.write(str(self.step) + "," + \
                        str(avg_reward) + " \n")
                break
    def fixSeed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False