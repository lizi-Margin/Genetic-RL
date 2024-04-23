import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import net
from train_buffer import ReplayBuffer 

class QTAgent:
    def __init__(self,state_num,action_num,lr = 0.05,futuer_para = 0.8,mr = 1e-5):
        self.size = 1

        self.state_num = state_num
        self.action_num = action_num
        self.q_table = np.zeros((action_num, state_num))  # Q 表格，行为加速、左转、右转，列为前、左、右是否有目标

        self.learning_rate = lr
        self.gamma = futuer_para
        self.epsilon = 0.15  # 探索率

        self.mutation_rate = mr

        self.last_A = 0
        self.last_S = 0


    def step(self , St,Rt,train = True):
        # action decision first -> Sara(real Atp1)    or   update first -> q-learning(asuming a somehow fake Atp1)
        At =self.choose_action(St)             
        if(train):self.update_q_table(Rt,St,At)        
        
        # 加了就是Q-learning
        #At =self.choose_action(St)             

        return At


    def update_q_table(self,nowR,nowS,nowA):
        self.TD(self.last_S,self.last_A,nowR,nowS,nowA) 
        self.mutate()
        self.last_A = nowA 
        self.last_S = nowS

    def TD(self,St,At,Rtp1,Stp1,Atp1):        
        oldQ = self.q_table[At,St]
        Qastp1 = self.q_table[Atp1,Stp1]
        self.q_table[At,St] = (1-self.learning_rate)*oldQ +self.learning_rate* (Rtp1 + self.gamma*Qastp1 - oldQ)

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_num)  # 随机选择动作 - Greedy Exploration
        else:
            return np.argmax(self.q_table[:, state])  # 选择具有最高 Q 值的动作

    # def step(self, state,train:bool , reward=0):
    #     action = self.choose_action(state)
    #     if train : 
    #         self.update_q_table(state, action, reward)
        
    #     return action
    
    def mutate(self ):
        for i in range(self.q_table.shape[0]):
            for j in range(self.q_table.shape[1]):
                if np.random.uniform(0, 1) < self.mutation_rate:
                    self.q_table[i, j] = np.random.rand()*self.q_table[i, j]* 2 # 随机突变染色体中的基因值

    def crossover(self, other):
        crossover_point = np.random.randint(1, self.q_table.shape[1] - 1)  # 随机选择一个交叉点
        new_q_table1 = np.concatenate((self.q_table[:, :crossover_point], other.q_table[:, crossover_point:]), axis=1)
        new_q_table2 = np.concatenate((other.q_table[:, :crossover_point], self.q_table[:, crossover_point:]), axis=1)
        return new_q_table1, new_q_table2





    # load / save
    def get_dict(self):
        dic = {
        'state_num':
        self.state_num 
        ,'action_num':
        self.action_num         
        ,'learning_rate':
        self.learning_rate
        ,'discount_factor':
        self.gamma
        ,'epsilon':
        self.epsilon
        }        
        return dic
    def load_dict(self,dic):
        
        self.state_num =dic['state_num']
        self.action_num     = dic['action_num']    
        #self.learning_rate= dic['learning_rate']
        #self.gamma =  dic['discount_factor']
        #self.epsilon = dic ['epsilon']
    
    # def load_table(self,addr,id):         
    #     self.q_table = np.zeros((self.action_num, self.state_num))  # Q 表格，行为加速、左转、右转，列为前、左、右是否有目标
    #     for i in range (self.action_num):
    #         self.q_table[i] = np.fromfile(addr+'/FlyingCreature-'+str(id)+'-table-'+str(i)+'.bin')

    # def save_table(self,addr,id):
    #     for i in range (self.action_num):
    #         self.q_table[i].tofile(addr+'/FlyingCreature-'+str(id)+'-table-'+str(i)+'.bin')

    def load_table(self,addr,id):         
        self.q_table = np.loadtxt(addr+'/FlyingCreature-'+str(id)+'-table-.txt')

    def save_table(self,addr,id):
        np.savetxt(addr+'/FlyingCreature-'+str(id)+'-table-.txt',self.q_table )






class RandAgent:
    def __init__(self, action_num):
       self.action_num = action_num 
       self.size = 0
    def step(self , St,Rt,train = True):
        return 0
    def get_dict(self):
       return {}
    def load_dict(self,dic):
        return
    def load_table(self,addr,id):         
        return
    def save_table(self,addr,id):
        return

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=8e-4, gamma=0.95,mr = 0, train_buffer = None):
        self.size = 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = net.QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.learning_rate = lr      
        self.gamma = gamma
        self.epsilon = 0.1
        self.batch_size = 50

        if train_buffer == None:  self. train_buffer = ReplayBuffer (int (1e4)) 
        else :     self.train_buffer = train_buffer
        
        
        self.mutation_rate = mr 
        
        self.last_A = 0
        self.last_S = np.random.rand(self.state_dim)
        self.last_q_table = None

   
        
    

    def step(self , St,Rt,train = True):
        

        At =self.choose_action(St)             
        if(train):self.update_q_net(Rt,St,At)        
        #At =self.choose_action(St)             

        return At



    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_values = self.q_network.forward(torch.tensor(state, dtype=torch.float32))
            self.last_q_table = q_values.detach().tolist()
            return torch.argmax(q_values).item()

  
    def update_q_net (self, nowR,nowS,nowA):
        self.train_buffer.add(self.last_S,self.last_A , nowR, nowS,int(0))
        
        chunk_size = self.batch_size
        if (len(self.train_buffer) < self.batch_size) : chunk_size = len(self.train_buffer)
        states, actions, rewards, next_states, dones = self.train_buffer.sample(chunk_size)
        self.UN (states,actions,rewards,next_states,dones)

        self.last_A = nowA
        self.last_S = nowS 
 
    def UN(self, state, action, reward, next_state, done):
        # with gradient
        q_values = self.q_network.forward(torch.tensor(state, dtype=torch.float32))        
        action = torch.LongTensor(action).unsqueeze(dim=-1)
        q_value = q_values.gather(1,action).flatten()

        # target with no gradient
        next_q_values = self.q_network.forward(torch.tensor(next_state, dtype=torch.float32)).detach()
        reward = torch.FloatTensor(reward)
        done = torch.IntTensor(done)
        next_q_value = next_q_values.max(1)[0]        
        target = reward + (1 - done) * self.gamma * next_q_value

        # optimize 
        #print(q_value,target)
        self.optimizer.zero_grad()
        loss = self.loss_fn(q_value, (target))        
        loss.backward()
        self.optimizer.step()


    def mutate(self ):
        pass
    def crossover(self, other):
        pass



    # load / save
    def get_dict(self):
        dic = {
        'state_dim':
        self.state_dim
        ,'action_dim':
        self.action_dim         
        ,'learning_rate':
        self.learning_rate
        ,'discount_factor':
        self.gamma
        ,'epsilon':
        self.epsilon
        }        
        return dic
    def load_dict(self,dic):
        
        self.state_dim =dic['state_dim']
        self.action_dim     = dic['action_dim']    
        #self.learning_rate= dic['learning_rate']
        #self.gamma =  dic['discount_factor']
        #self.epsilon = dic ['epsilon']
    
    # def load_table(self,addr,id):         
    #     self.q_table = np.zeros((self.action_num, self.state_num))  # Q 表格，行为加速、左转、右转，列为前、左、右是否有目标
    #     for i in range (self.action_num):
    #         self.q_table[i] = np.fromfile(addr+'/FlyingCreature-'+str(id)+'-table-'+str(i)+'.bin')

    # def save_table(self,addr,id):
    #     for i in range (self.action_num):
    #         self.q_table[i].tofile(addr+'/FlyingCreature-'+str(id)+'-table-'+str(i)+'.bin')

    def load_table(self,addr,id):         
        self.load_net(addr+'/FlyingCreature-'+str(id)+'-table-.txt')

    def save_table(self,addr,id):
        self.save_net(addr+'/FlyingCreature-'+str(id)+'-table-.txt' )

    def save_net(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load_net(self,filename):
        self.q_network.load_state_dict(torch.load(filename))

