import numpy as np

class QTAgent:
    def __init__(self,state_num=4,action_num=4,lr = 0.05,futuer_para = 0.8):

        self.state_num = state_num
        self.action_num = action_num
        self.q_table = np.zeros((action_num, state_num))  # Q 表格，行为加速、左转、右转，列为前、左、右是否有目标

        self.learning_rate = lr
        self.gamma = futuer_para
        self.epsilon = 0.15  # 探索率

        self.last_A = 0
        self.last_S = 0

    # def update_q_table(self, state, action, reward):
    #     current_q_value = self.q_table[action, state]
    #     max_future_q_value = np.max(self.q_table[:, state])
    #     new_q_value = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.gamma * max_future_q_value)
    #     self.q_table[action, state] = new_q_value
    def update(self , St,At,Rt):
        self.update_q_table(Rt,St)
        self.last_A = At 
        self.last_S = St

    def update_q_table(self,nowR,nowS):
        self.TD(self.last_S,self.last_A,nowR,nowS) 

    def TD(self,St,At,Rtp1,Stp1):
        Atp1 =np.argmax(self.q_table[:, Stp1]) 
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
    
    def mutate(self, mutation_rate):
        for i in range(self.q_table.shape[0]):
            for j in range(self.q_table.shape[1]):
                if np.random.uniform(0, 1) < mutation_rate:
                    self.q_table[i, j] = np.random.rand()  # 随机突变染色体中的基因值

    def crossover(self, other):
        crossover_point = np.random.randint(1, self.q_table.shape[1] - 1)  # 随机选择一个交叉点
        new_q_table1 = np.concatenate((self.q_table[:, :crossover_point], other.q_table[:, crossover_point:]), axis=1)
        new_q_table2 = np.concatenate((other.q_table[:, :crossover_point], self.q_table[:, crossover_point:]), axis=1)
        return new_q_table1, new_q_table2

    def get_dict(self):
        dic = {
        'state_num':
        self.state_num 
        ,'action_num':
        self.action_num         
        ,'learning_rater':
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
        #self.learning_rate= dic['learning_rater']
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