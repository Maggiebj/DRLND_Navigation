import numpy as np
import random
from collections import namedtuple, deque

from p1_dqn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
BETA_START = 0.4        # hyper parameter for importance sampling weight
BETA_END = 1.0          # 
BETA_INCREASE = 8e-3    # beta increases to maximal 1.0

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

        # Priority Replay memory
        self.prioritybuffer = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.prioritybuffer.add(state, action, reward, next_state, done)
        
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get priority sample subset and learn
            if len(self.prioritybuffer.memory) > BATCH_SIZE:
                experiences = self.prioritybuffer.sample() #batch size of prioritized experience tuples
                self.learn(experiences, GAMMA)

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
            action_values = self.qnetwork_local(state)#forward
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, indice, weight) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)# dimension 1 is Q value
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions) #gather(input,dim,index) ,gathers values(input) along an dim specified by index.(actions)
       

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) * weights
        prios = loss + 1e-5
        loss = loss.mean()
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.prioritybuffer.update_priorities(indices, prios.data.cpu().numpy()) #update experiences priority
        self.optimizer.step()#update local network

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   #soft update  from local to target             

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



class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        """Initialize a PriorityReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha(float): hyperparameter using to calculate priority
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.priority = np.ones((buffer_size,),dtype=np.float32) #initialized to 1
        self.beta =BETA_START #increasingly up to 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):
        """Prioritized sample a batch of experiences from memory."""
        
        if len(self.memory)==len(self.priority):
            
            prios = self.priority
        else:
            prios = self.priority[:len(self.memory)]
        probs = prios ** self.alpha
        probs /= probs.sum()
      
        indices = np.random.choice(len(self.memory),self.batch_size,p=probs)
        samples = [self.memory[idx] for idx in indices]
       
        total = len(self.memory)
        weights = (total*probs[indices])**(-self.beta)
        weights /= weights.max()                       #normalized to [0,1]
        weights = torch.from_numpy(np.array(weights,dtype=np.float32)).to(device)
        
        self.beta = min(BETA_END,self.beta+BETA_INCREASE) #increase beta
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, indices, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def update_priorities(self,batch_indices,batch_priorities):
        for idx, prio in zip(batch_indices,batch_priorities):
            self.priority[idx] = prio