import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from creature import FlyingCreature
from train_buffer import ReplayBuffer

class World:
    def __init__(self, num_creatures,num_creatures_rand = 0,agent_size012=2):
        self.num_creatures = num_creatures
        self.creatures  = []
        self.player_creature = None

        buff = ReplayBuffer(int(1e4))
        i = 0
        for _ in range(num_creatures):
            self.creatures.append(FlyingCreature(i,agent_size012=agent_size012,train_buffer= buff))
            i +=1
        self.num_creatures += num_creatures_rand
        for _ in range(num_creatures_rand):
            self.creatures.append(FlyingCreature(i,agent_size012=0))
            i +=1       
        self.w = 800
        self.h = 800
        
        self.step = -1
        self.total_reward_list_capacity = int(1e3)
        self.total_reward_list  = None #np.zeros(self.total_reward_list_capacity)

    def update(self,train=True):
        self.total_reward_list[self.step%self.total_reward_list_capacity] = 0
        for creature in self.creatures:
            c_reward= creature.update(self,train = train)
            self.total_reward_list[self.step%self.total_reward_list_capacity] += c_reward
            self.lock_position(creature)

    def update_player(self,wasd):        
        player_reward = self.player_creature.update_in_control(self,wasd)
        
    def action(self):
        for creature in self.creatures:
            creature.take_action()
            self.lock_position(creature)
        if self.player_creature!= None:
            self.player_creature.take_action()
            self.lock_position(self.player_creature)


    
    def lock_position(self,c):
        creature = c
        creature.position[0]= creature.position[0]  % self.w
        creature.position[1]= creature.position[1]  % self.h
    
    def render(self, screen):
        screen.fill((255, 255, 255))  # 白色背景
        for creature in self.creatures:
            pygame.draw.polygon(screen, (10, 5, 45), creature.get_vertices())  # 画出生物 
        if self.player_creature!= None : pygame.draw.polygon(screen, (255, 0, 0), self.player_creature.get_vertices())  # 画出生物
        pygame.display.flip()
    

    def execute(self,player):
        self.player_creature= player 

    def run(self,train=False,plot = False):
        
        self.step = -1
        pygame.init()
        screen = pygame.display.set_mode((self.w, self.h))
        clock = pygame.time.Clock()

        # 创建一个图表
        fig, ax = plt.subplots()
        x = np.linspace(0, self.total_reward_list_capacity ,self.total_reward_list_capacity)
        self.total_reward_list  = np.zeros(self.total_reward_list_capacity)
        
       
        
        running = True
        while running:
            wasd = [False,False,False,False]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # 检测按键
                    if event.key == pygame.K_w:
                        wasd[0] = True
                    elif event.key == pygame.K_a:
                        wasd[1] = True
                    elif event.key == pygame.K_s:
                        wasd[2] = True
                    elif event.key == pygame.K_d:
                        wasd[3] = True

            self.step+=1
            self.update(train=train)
            if self.player_creature != None :self.update_player(wasd)

            self.action()

            self.render(screen)            

            clock.tick(30)

            #if (self.step % 10 == 0): print(self.step)
            if (plot and self.step % 50 == 0):
                print(self.step ,' -- ' , self.total_reward_list[self.step%self.total_reward_list_capacity])
                print('   ',self.creatures[0].agent.last_q_table )
            if (self.step >0 and self.step % 1000 == 0):
                self.save_all('./SAVE/AUTOSAVE-GAME-MODE')
                if plot:
                    plt.plot(x,self.total_reward_list) 
                    plt.show()


        pygame.quit()
        self.save_all('./SAVE/AUTOSAVE-GAME-MODE')



    def run_no_play(self,totol_step = 5*1e3):
        self.step = -1
        # 创建一个图表
        fig, ax = plt.subplots()
        x = np.linspace(0, self.total_reward_list_capacity ,self.total_reward_list_capacity)
        self.total_reward_list  = np.zeros(self.total_reward_list_capacity)
        
       
        
        running = True
        while running and self.step<totol_step :
            self.step+=1
            self.update()
            self.action()
            if (self.step % 10 == 0):
                print(self.step ,' -- ' , self.total_reward_list[self.step%self.total_reward_list_capacity])
                print('   ',self.creatures[0].agent.last_q_table )
            if (self.step >0 and self.step % 1000 == 0):

                self.save_all('./SAVE/AUTOSAVE')
                plt.plot(x,self.total_reward_list) 
                plt.show()


        pygame.quit()
        self.save_all('./SAVE/AUTOSAVE')
    
    def save_all(self,ad): 
        for creature in self.creatures:
            creature.save_json(ad)
        print('save sccess')
    def load_all(self,ad,to_id = None):
        if id != None : 
            for creature in self.creatures:
                creature.open_json(ad,id = np.random.randint(0,to_id) )
 
        else : 
            for creature in self.creatures:
                creature.open_json(ad)
    
        print('load sccess')
    


