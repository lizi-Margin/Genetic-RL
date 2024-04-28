from env import  *
from creature import *
# 创建环境并运行
world = World(60,num_creatures_rand=140)

#world.load_all('',to_id=20)
world.run_no_play(totol_step=10*1e3)

