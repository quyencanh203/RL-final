
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # dung luong 
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    # def sample(self, batch_size):
    #     assert batch_size
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )   
        
    # ham lay dung luong bo nho
    def __len__(self):
        return len(self.memory)
    

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, eps=1e-6):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.eps = eps
        self.priorities = []

    def _get_priority(self, error):
        return (np.abs(error) + self.eps) ** self.alpha

    def push(self, state, action, reward, next_state, done, error):
        priority = self._get_priority(error)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        weights = (len(self.memory) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
            torch.tensor(np.array(weights), dtype=torch.float32),
            indices
        )

    def update_priorities(self, batch_indices, errors):
        for idx, error in zip(batch_indices, errors):
            priority = self._get_priority(error)
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

    
