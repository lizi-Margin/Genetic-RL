# agent_fish shc 2024

# 集体游动： 一定的距离和方向
# 群体防御： 紧密的阵型、迅速改变方向
# 信息传递： 鱼群中的个体之间进行信息传递。这种信息交流有助于群体中的个体更好地协调行动，比如发现食物或者发现潜在的危险。
# 群体繁殖： 集体进行繁殖。协同的方式产卵和保护幼鱼

# 不完全状态： 个体只能感知到附近的状态

# state - 状态输入 ， action - 动作输出 . 采用遗传算法

import pygame
import numpy as np

class FishSimulator:
    def __init__(self, num_fish, num_predators):
        self.screen_width = 600
        self.screen_height = 600

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Fish Simulation")
        self.clock = pygame.time.Clock()

        self.size_range_fish = (5, 7)
        self.velocity_range_fish = (-1, 1)
        self.max_speed_fish = 7
        self.vision_range_fish = 80

        self.size_predator = 15
        self.velocity_range_predator = (-1, 1)
        self.max_speed_predator = 10
        self.vision_range_predator = 400



        self.fishes = []
        self.dead_fishes=[]
        i = 0
        for _ in range(num_fish):
            position = np.random.uniform(0, self.screen_width, size=2)
            velocity = np.random.uniform(-1, 1, size=2)
            size = np.random.uniform(*self.size_range_fish)
            self.fishes.append(Fish(i ,position, velocity, size, self.screen_width, self.screen_height, self.max_speed_fish, self.vision_range_fish))
            i+=1

        self.predators = []
        self.dead_predators=[]
        i = 0
        for _ in range(num_predators):
            position = np.random.uniform(0, self.screen_width, size=2)
            velocity = np.random.uniform(-1, 1, size=2)
            size = self.size_predator
            self.predators.append(Predator(i,position, velocity, size, self.screen_width, self.screen_height, self.max_speed_predator, self.vision_range_predator))
            i+=1

    def update(self):
        for this_fish in self.fishes:
            near_fishes_visible = [fish.get_visible() for fish in self.fishes if 0 <np.linalg.norm(this_fish.position - fish.position) < this_fish.vision_range]            
            near_predators_visible = [predator.get_visible() for predator in self.predators if 0 <np.linalg.norm(this_fish.position - predator.position) < this_fish.vision_range]

            # 计算周围鱼类的平均位置和速度
            near_fish_positions = [fish.position for fish in near_fishes_visible]
            near_fish_velocities = [fish.velocity for fish in near_fishes_visible]
            # 计算周围掠食者的平均位置
            near_predator_positions = [predator.position for predator in near_predators_visible] 
            near_predator_velocities = [predator.position for predator in near_predators_visible]
            this_fish.step(near_fishes_visible,near_fish_positions,near_fish_velocities,near_predators_visible,near_predator_positions,near_predator_velocities)

        for this_predator in self.predators:

            near_fishes_visible = [fish.get_visible() for fish in self.fishes if 0 <np.linalg.norm(this_predator.position - fish.position) < this_predator.vision_range]            
            near_predators_visible = [predator.get_visible() for predator in self.predators if 0 <np.linalg.norm(this_predator.position - predator.position) < this_predator.vision_range]
            # 计算周围鱼类的平均位置和速度
            near_fish_positions = [fish.position for fish in near_fishes_visible]
            near_fish_velocities = [fish.velocity for fish in near_fishes_visible]
            # 计算周围掠食者的平均位置
            near_predator_positions = [predator.position for predator in near_predators_visible] 
            near_predator_velocities = [predator.position for predator in near_predators_visible]
            
            dead_list = this_predator.step(near_fishes_visible,near_fish_positions,near_fish_velocities,near_predators_visible,near_predator_positions,near_predator_velocities)

            for id in dead_list:
                for fish in self.fishes:
                    if fish.id == id :
                        fish.dead = True
                        fish .position= np.array([-1e6,-1e6])


    def plot(self):
        self.screen.fill((255, 255, 255))
        for fish in self.fishes:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(fish.position[0]), int(fish.position[1])), int(fish.size))
        for predator in self.predators:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(predator.position[0]), int(predator.position[1])), int(predator.size))
        pygame.display.flip()

    def excute( self,Creature) : pass
    def run_simulation(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update()
            self.plot()
            self.clock.tick(60)

        pygame.quit()


class Creature :
    def __init__(self):
        self. id = 0
        self.dead= False
        self.position = None 
        self.velocity  =None
        self.size  = None
    def __init__(self,id,dead,position,velocity,size):
        self. id = id
        self.dead= dead
        self.position =  position
        self.velocity  = velocity
        self.size  = size
    @ staticmethod
    def generate_random_vector(k):
    # 生成一个长度为 2 的随机向量
        random_vector = np.random.rand(2)
        
        # 将向量的分量范围调整到 [-0.5, 0.5]
        random_vector -= 0.5
        
        # 缩放向量，使其模长为 k/2
        scaled_vector = random_vector * (k / 2) / np.linalg.norm(random_vector)
        
        return scaled_vector        
    @staticmethod
    def get_nearest_from(self_position,near_positions,max_rng=1e6):

            nearest_rng = max_rng
            nearest_pos = None
            for pos in near_positions: 
                this_rng = np.linalg.norm(self_position - pos)
                
                if this_rng < nearest_rng :
                    nearest_rng =this_rng
                    nearest_pos = pos

            return nearest_rng, nearest_pos
        
    def get_visible(self):
        return Creature(self.id,self.dead,self.position,self.velocity,self.size)
    
    def get_rng(self,your_pos):
        return np.linalg.norm(self.position-your_pos)



class Fish(Creature):
    def __init__(self,id, position, velocity, size, screen_width, screen_height, max_speed, vision_range):
        super().__init__(id,False,position,velocity,size)



        
        
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_speed = max_speed
        self.vision_range = vision_range




    def step(self,fishes, fishes_pos,fishes_v,predators,predators_pos,predators_v,ABM=True):
        if  self.dead: return
        escape_tendency = 2.9
        with_friend_v_tendency = 2.4
        to_friend_center_tendency = 0.35
        out_friend_tendency = 1.35 
        radom_swim_tendency = 1
 
        to_map_center_tendency = 0.003

        if fishes_pos and ABM:
            center = np.mean(fishes_pos, axis=0)
            avg_velocity = np.mean(fishes_v, axis=0)
            nearest_rng, nearest_pos = self.get_nearest_from(self.position,fishes_pos,max_rng=self.vision_range)
            # print(nearest_rng,nearest_pos)
            
            # 同伴速度
            self.velocity +=  with_friend_v_tendency * (avg_velocity - self.velocity)/(2*self.max_speed)
            
            # 与同伴保持距离
            k = self.vision_range*0.7
            if(self.vision_range> nearest_rng)and (nearest_rng<k): 
                to_nearest = (nearest_pos - self.position)
                self.velocity += -out_friend_tendency *(k * to_nearest /(np.linalg.norm(to_nearest))  -  to_nearest)/k 
                
            #向同伴靠拢
            k = nearest_rng*2
            neighbors_mean_pos= []
            has_neighbor = False
            for fish in fishes :
                if fish.get_rng(self.position) < k :
                    has_neighbor=True
                    neighbors_mean_pos.append(fish.position)
            neighbors_mean_pos = np.mean(neighbors_mean_pos) 
            if  has_neighbor:self.velocity += to_friend_center_tendency * (neighbors_mean_pos - self.position)/k
            else:self.velocity += to_friend_center_tendency * (center - self.position)/np.linalg((center - self.position))



        # 逃离掠食者
        k = 1
        if predators_pos:
            escape_direction = np.mean(self.position - np.array(predators_pos), axis=0) 
            self.velocity += escape_tendency *(escape_direction /(np.linalg.norm(escape_direction))  ) 

        # 随机游动
        self.velocity += self.generate_random_vector(radom_swim_tendency)

        # 阻力&min_spd
        min_k = 0.5
        self.velocity = self.velocity *0.96
        if np.linalg.norm(self.velocity)<self.max_speed*min_k:self.velocity= self.max_speed*min_k* self.velocity/np.linalg.norm(self.velocity)

        # 限制速度不能超过最大速度
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        
        # 处理地图边界
        k = 0.5
        
        wh2 =    np.array([self.screen_width/2,self.screen_height/2])
        screen_center_pos = self.position -  wh2
        if np.abs(screen_center_pos[0]) > self.screen_width*0.5 *k:
            self.velocity += to_map_center_tendency * (np.array([self.screen_width / 2, self.position[1]]) - self.position)
        if np.abs(screen_center_pos[1])>self.screen_height*0.5  *k:
            self.velocity += to_map_center_tendency * (np.array([self.position[0], self.screen_height / 2]) - self.position)



        # 更新位置
        self.position += self.velocity 

        if self.position[0] < 0 or self.position[0] > self.screen_width:
            self.velocity[0] *= -1
            self.velocity += 0.1 * (np.array([self.screen_width / 2, self.position[1]]) - self.position)
        if self.position[1] < 0 or self.position[1] > self.screen_height:
            self.velocity[1] *= -1
            self.velocity += 0.1 * (np.array([self.position[0], self.screen_height / 2]) - self.position)


        
class Predator(Creature):
    def __init__(self, id ,position, velocity, size, screen_width, screen_height, max_speed, vision_range):

        super().__init__(id,False,position,velocity,size)
        
        
        
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_speed = max_speed
        self.vision_range = vision_range

    def step(self,fishes, fishes_pos,fishes_v,predators,predators_pos,predators_v):
        out_friend_tendency = 1.2

        id_to_eat = []

        if self.dead : return

        # 寻找最近的鱼    
        min_distance ,closest_fish_pos  = self.get_nearest_from(self.position,fishes_pos,max_rng=self.vision_range)
       
        if self.vision_range>min_distance:
            # 调整速度，向最近的鱼靠近
            direction = closest_fish_pos - self.position
            self.velocity += 1.5 * direction/np.linalg.norm(direction)
            
            # 限制速度不能超过最大速度
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity *= self.max_speed / speed

        if predators_pos:
             # 与同伴保持距离
            k = 50
            nearest_rng , nearest_pos = self.get_nearest_from(self.position,predators_pos,max_rng=self.vision_range)
            if(self.vision_range> nearest_rng)and (nearest_rng<k): 
                to_nearest = (nearest_pos - self.position)
                self.velocity += -out_friend_tendency *(k * to_nearest /(np.linalg.norm(to_nearest))  -  to_nearest)/k 
        
        # 更新位置
        self.position += self.velocity
        
        # 处理地图边界
        # if self.position[0] < 0 or self.position[0] > self.screen_width:
        #     self.velocity[0] *= -1
        #     self.velocity += 0.1 * (np.array([self.screen_width / 2, self.position[1]]) - self.position)
        # if self.position[1] < 0 or self.position[1] > self.screen_height:
        #     self.velocity[1] *= -1
        #     self.velocity += 0.1 * (np.array([self.position[0], self.screen_height / 2]) - self.position)


        for fish in fishes :
            if fish.get_rng(self.position) < 6: 
                id_to_eat .append(fish.id)


        # 返回
        
        print(id_to_eat)
        return id_to_eat
    




me = 'Creature'

# 设置参数
num_fish = 100
num_predators = 1
# 创建模拟器并运行模拟
world = FishSimulator(num_fish, num_predators)
world.excute(me);
world.run_simulation()


