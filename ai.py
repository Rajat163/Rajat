# AI for Self Driving Car

# Step 1 Importing the libraries

import numpy as np #for the arrays 
import random # we'll have to select random experiences in experience replay 
import os # To load the previously saved brains and save newly trained brains. line 100
import torch #We are using pytorch
import torch.nn as nn
import torch.nn.functional as F #to perform functions of a neural network
import torch.optim as optim # for optimization
import torch.autograd as autograd
from torch.autograd import Variable  #we need variables containing tensors as well as gradients(a type of array ) so we import it

#Step 2 Creating Architecture of Neural Network

class Network(nn.Module): #Network is inherited from Module class. for line 103 
    
    def __init__(self, input_size, nb_action): 
        super(Network, self).__init__() #super is function of Module Class used to get the tools ofModule class
        self.input_size = input_size
        self.nb_action = nb_action  
        self.fc1 = nn.Linear(input_size, 30) #fc1 and fc2 are Full Coonections fully connected hidden layer. input_size -30 is connection between input layer and hidden layer. No of hidden layer neurons are found by  experimetns 
        self.fc2 = nn.Linear(30, nb_action)#Linear creates the linear full connections
        
    def forward(self, state):# it will activate the neurons to move forward
        x = F.relu(self.fc1(state)) # x will acticvate the required 30 hhidden neurons. It is kind of fof logic
        q_values = self.fc2(x) #fc2 will tell you the action and x will give you next state
        return q_values
    
    #Step 3 Implemetning expreience Replay
    
class ReplayMemory(object):
    
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
    
        def push(self, event):
            self.memory.append(event)
            if len(self.memory) > self.capacity:
               del self.memory[0]
    
        def sample(self, batch_size):
            samples = zip(*random.sample(self.memory, batch_size))
            return map(lambda x: Variable(torch.cat(x, 0)), samples)   # We want to return samples in a torch variable. Now cat(x,o) is used for line 81
            
        #Step 4 Implementing Deep Q-learning model
        
class Dqn():
          
    def __init__(self,input_size,nb_action,gamma):             
        self.gamma=gamma # Gamma is discount factor of Q learning equation
        self.reward_window=[]#Mean of reward of last 100 acrions wrt time
        self.model= Network(input_size,nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.001) 
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  #we have input state as vector butwe want it into a Tensor but NN only accepts input in form of Batches so we use unsqueeze to creae a fake dimension of batch
        self.last_action = 0                    
        self.last_reward = 0
            
    def select_action(self,state): #state is the i/p of NN
        probs=F.softmax(self.model(Variable(state, volatile=True))*7)#T=7. Higher the value of Temprature Higher the surity of action we will get
        action = probs.multinomial() #randomm draw of Action left,right,straight
        return action.data[0,0] 
           
    def learn(self,batch_state, batch_next_state, batch_reward, batch_action):
        outputs=self.model(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1) #gather will give the selected action only in Batch_action, Action is at index 1. As 
        next_outputs = self.model(batch_next_state).detach().max(1)[0] # for deep q learning equation
        target=self.gamma*next_outputs + batch_reward
        td_loss=F.smooth_l1_loss(outputs,target)
        self.optimizer.zero_grad()#ReInitializes at every Iteration
        td_loss.backward(retain_variables = True) #Performs Backward Propagation
        self.optimizer.step()            # Performs Stochastic Gradient toupdate weights
                
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: # 1st memory is object of Replay Class and 2nd memory is attribute of this class
             batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
             self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action             
            
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
            
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
                    
    def load(self):
        if os.path.isfile('last_brain.pth'):
           print("=> loading checkpoint... ")
           checkpoint = torch.load('last_brain.pth')
           self.model.load_state_dict(checkpoint['state_dict']) # Updating Weights of Model
           self.optimizer.load_state_dict(checkpoint['optimizer']) #same as above
           print("done !")
        else:
           print("no checkpoint found...")
