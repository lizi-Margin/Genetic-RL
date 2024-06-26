import numpy as np
import math as m
from agent import QTAgent
from agent import DQNAgent
from agent import RandAgent
from utils import Vector3
from utils import Vector2
import json

# w东 h南 地    前 左 上

class FlyingCreature:
    def __init__(self,id,agent_size012 = 1,map_size = 800,train_buffer = None):
        self.id = id
        self.default_idle_v = 1.5
        self.default_thrust_v = 3.5
        self.default_drag_v = 0.2
        self.default_aim_angle = m.pi/12
        self.default_vission_angle = m.pi/4

        self.vission_range = 60
        self.mobility = 0.6
        self.size = 5
        self.position = np.random.rand(2) * map_size  # 随机初始位置 
        self.velocity = (np.random.rand(2)-0.5) 
        self.velocity = (self.velocity/np.linalg.norm(self.velocity))  * self.default_idle_v

        if agent_size012 == 0:
            self.agent = RandAgent(4) 
        elif agent_size012 == 2 : self.agent = DQNAgent(2,4,train_buffer=train_buffer)
        else :self.agent = QTAgent(4,4)
        
        self.in_vission = []
        self.in_aim_range = []

        self.this_action = None
    
    def set_agent(self,agent):
        if agent != None : self.agent = agent

    def update(self,env,train=True):
        self.syc_in_range(env)
        state = self.get_state()
        state_ind = self.get_state_index(state)   
        reward = self.get_reward()

        if self.agent.size==2 :action_index =  self.agent.step(state,reward,train= train)
        elif self.agent.size == 1:action_index =  self.agent.step(state_ind,reward,train= train)
        else:action_index =  self.agent.step(state_ind,reward,train= train)

        self.this_action = self.get_action_from_index(action_index)

        if self.agent.size == 0 : reward = 0 
        return reward
    

    def update_in_control(self,env,wasd):
        self.size = 20
        self.mobility = 2
        
        self.syc_in_range(env)
        state_ind = self.get_state_index(self.get_state())
        reward = self.get_reward()
        
        action_index = 0
        if wasd[0] : action_index =1
        elif wasd[1]:action_index = 2
        elif wasd[3]:action_index = 3
        self.this_action = self.get_action_from_index(action_index)
        return reward
        

    def syc_in_range(self,env):
        self.in_vission = []
        self.in_aim_range = []
        for c in env.creatures : 
            if self.get_range(c.position)>self.vission_range : continue
            angle ,on_left  = self.get_location_angle_l ( c.position)
            if angle> self.default_vission_angle:continue

            self.in_vission .append(c) 
            if self.get_range(c.position)<self.vission_range/2 :                 
                if (angle< self.default_aim_angle):self.in_aim_range.append(c)

    def get_location_angle_l(self,ndarr):
            to_c = ((ndarr - self.position).tolist())
            to_c = Vector2(to_c)
            my_v = Vector2(self.velocity.tolist())
            ang = to_c.get_angle(my_v)            
            to_c = to_c.get_Vector3()
            left_wing = my_v.get_Vector3().rotate_xyz_fix(0,0,m.pi/2)
            on_left = left_wing.get_prod(to_c)>0
            return ang ,on_left
    def get_velocity_angle_l(self,ndarr):
            v = ((ndarr ).tolist())
            v = Vector2(v).get_Vector3()
            my_v = Vector2(self.velocity.tolist()).get_Vector3()
            ang = my_v.get_angle(v)            
            left_wing = my_v.rotate_xyz_fix(0,0,m.pi/2)
            on_left = left_wing.get_prod(v)>0
            return ang ,on_left
       
    def get_self_xyz(self,to_c):
        to_c = Vector2(to_c)
        to_c = to_c.get_Vector3()
        v = Vector2(self.velocity.tolist()).get_Vector3()
        ang,on_left = self.get_velocity_angle_l( np.array([1,0,0]))
        if not on_left : ang = -ang
        to_c.rev_rotate_zyx_self(m.pi,0,-ang)         
        to_c = to_c.get_Vector2()
        return to_c.get_list()

    def get_fix_xyz(self,to_c):
        to_c = Vector2(to_c)
        to_c = to_c.get_Vector3()
        v = Vector2(self.velocity.tolist()).get_Vector3()
        ang = v.get_angle(Vector3([1,0,0]))
        to_c.rotate_zyx_self(m.pi,0,ang) 
        to_c = to_c.get_Vector2()
        return to_c.get_list()


    def get_reward(self):
        reward = 0
        if (len(self.in_vission)>0):
            reward += 3
        if (len(self.in_aim_range)>0):
            reward += 7
        return reward
    def get_state(self):
        state = [2,0]
        min_rng = self.vission_range
        for c in self.in_vission:
            rng = self.get_range(c.position)
            if rng < min_rng :
                to_c =  c.position -self.position
                state = self.get_self_xyz(to_c.tolist())
                state = Vector2(state)
                state = state.prod(1/state.get_module()).get_list()  
                min_rng = rng
        return state


            
    def get_state_index(self,state):
        
        on_left = state [1]>=0
        vec_state = Vector2(state).get_Vector3()
        rng = vec_state.get_module()
        ang = (vec_state).get_angle(Vector3([1,0,0]))

        state_index = 0
        if rng <=1.001 :
            if (ang < self.default_aim_angle ): state_index = 1
            elif (ang < self.default_vission_angle): 
                if on_left:
                    state_index = 2
                else :
                    state_index = 3
            
        
        return state_index

    def get_action_from_index(self,action_index:int):
        action = np.array([0,0]) # 自身坐标系, 前左上
        if (action_index == 1):
            action[0] = 1
        elif (action_index ==2) :
            action[1] = 1
        elif (action_index == 3):
            action[1] = -1
        
        ##action = action*(1/np.linalg.norm(action))
        return ((action)* self.mobility).tolist()

    def take_action(self,is_player = False):
        # action  = Vector3([action[0],action[1],0])
        # v = Vector2(self.velocity.tolist()).get_Vector3()
        # ang = v.get_angle(Vector3([1,0,0]))
        # action.rotate_zyx_self(m.pi,0,ang) 
        # ##action = action.prod(1/action.get_module())
        # mo =action.get_module()
        
        # v.add(action)         ############ problems

        action = self.this_action

        v = Vector2(self.velocity.tolist()).get_Vector3()
        v.rotate_zyx_self(0,0,-action[1]*m.pi/10)
        self.velocity = np.array(v.get_Vector2().get_list())
        self.velocity = (self.velocity)+ (self.velocity/np.linalg.norm(self.velocity)) * action[0]

        ### 阻力 ###
        self.velocity = (self.velocity)- (self.velocity/np.linalg.norm(self.velocity)) * self.default_drag_v

        ### 限速 ###
        mo = np.linalg.norm(self.velocity)
        if mo> self.default_thrust_v : self.velocity = self.velocity/mo * self.default_thrust_v
        if mo< self.default_idle_v : self.velocity = self.velocity/mo * self.default_idle_v
        
        self.position += self.velocity

    def get_range(self,ndarr):
        return np.linalg.norm(self.position - ndarr)
    def get_vertices(self):
        # 返回生物的顶点坐标，用于绘制带尖头的三角形
        size =  self.size # 生物大小
        direction = self.velocity
        angle = np.arctan2(direction[1], direction[0])
        vertices = [
            (self.position[0] + size * 1. * np.cos(angle), self.position[1] + size * 1. * np.sin(angle)),
            (self.position[0] + size * np.cos(angle + 10 * np.pi / 12), self.position[1] + size * np.sin(angle + 10 * np.pi / 12)),
            (self.position[0] + size * np.cos(angle + 14 * np.pi / 12), self.position[1] + size * np.sin(angle + 14 * np.pi / 12)),

            
        ]
        return vertices



    def save_json(self,addr :str):
        
        dic = {
        'id': self.id ,
       'default_idle_v':
        self.default_idle_v 
       ,'default_thrust_v':
        self.default_thrust_v 
      ,'default_drag_v':
        self.default_drag_v 
      ,'default_aim_angle': 
        self.default_aim_angle 
      ,'default_vission_angle':
        self.default_vission_angle
      ,'vission_range':
        self.vission_range
      ,'mobility':
        self.mobility 
      ,'size':
        self.size, 
        'agent':self.agent.get_dict()
        }

        json_str = json.dumps(dic, indent=4)

        # 将 JSON 字符串存储到文件中
        with open(addr+'/FlyingCreature-'+str(self.id)+'.json', "w") as json_file:
            json_file.write(json_str)        
        self.agent.save_table(addr,self.id) 

    def open_json(self,addr,id = None):
        id_bac = self.id
        if id != None :              
            self.id = id 

        # 将 JSON 字符串存储到文件中
        with open(addr+'/FlyingCreature-'+str(self.id)+'.json', "r") as json_file:
            dic = json.load(json_file)
        
        self.agent.load_dict(dic['agent'])



        self.id =dic['id']
       
        self.default_idle_v = dic['default_idle_v']
        self.default_thrust_v  = dic['default_thrust_v'] 
        self.default_drag_v =dic['default_drag_v']
        self.default_aim_angle =dic['default_aim_angle']
     
        self.default_vission_angle =dic['default_vission_angle']
      
        self.vission_range=dic['vission_range']
      
        self.mobility= dic['mobility'] 
      
        self.size= dic['size'] 
        

        self.agent.load_table(addr,self.id)

        self.id  = id_bac