import numpy as np
import random
from collections import namedtuple, deque
import itertools

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


import time

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4       # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # Minimum priority
        self.eps = 0.0001
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        start_time = time.time()
        self.memory.add(state, action, reward, next_state, done)
        # print("Sample add time {:.4f}".format(start_time - time.time()))
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1 
        if len(self.memory) > 0 and (self.t_step % UPDATE_EVERY == 0):
            # start_time = time.time()
            experiences = self.memory.sample()
            # print("Sample time {:.4f}".format(start_time - time.time()))
            self.learn(experiences, GAMMA, self.t_step)
            self.memory.updateBeta()
        
        if self.t_step % 1e3 == 0:
            self.memory.reorder()
            

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, t_step):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Get best actions from local network
        target_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # And use them to evaluate the target network
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, target_actions)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # TD Error
        td_error = (Q_targets - Q_expected).abs().detach().numpy() + self.eps
        start_time = time.time()
        self.memory.setPriority(indices, td_error)
        # print("Update time {:.4f}".format(start_time - time.time()))
         
        start_time = time.time()
        # Compute loss
        loss = F.mse_loss(weights * Q_expected, weights * Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("Backprop time {:.4f}".format(start_time - time.time()))
        
        

        if (t_step %UPDATE_EVERY) == 0:
        # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.max_prio = 100
        self.beta = 0.5
        self.beta_annealing = 1.000005
       
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, priority=None):
        """Add a new experience to memory."""
        if priority == None: 
            priority = self.max_prio
        e = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done, 'priority': priority}
        self.memory.appendleft(e)
        
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        N = len(self.memory)

        # Sampled indices
        depth = int(np.ceil(np.log2(N)))
        indices = []
        for k in range(depth):
            x = range(N)
            x_slice = x[2**k : 2**(k+1)]
            indices.append(np.random.choice(x_slice))
        
        experiences = [self.memory[index] for index in indices]
        
        weights = (N / np.array(indices)) ** (-self.beta)
        weights = weights / weights[-1]

        states = torch.from_numpy(np.vstack([e['state'] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e['action'] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e['reward'] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e['next_state'] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e['done'] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack([weight for weight in weights])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)

    def setPriority(self, indices, priorities):
        """
        Update the prioritiers at the `indices` with the given `priorities`
        
        Params
        ======
            indices (list of int): indices at which to update
            priorities (list of double): new priorities
        """
        for i, p in zip(indices, priorities):
            self.memory[i]['priority'] = float(p)
            
    def updateBeta(self):
        """
        Update maximum priority
        """
        self.beta *= self.beta_annealing 
        
    def reorder(self):
        """
        Reorder array
        """
        priorities = [e['priority'] for e in self.memory]
        sorted_indices = np.argsort(priorities)
        sorted_experiences = deque(maxlen=self.buffer_size)
        for index in sorted_indices:
            sorted_experiences.appendleft(self.memory[index])
        self.memory = sorted_experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)