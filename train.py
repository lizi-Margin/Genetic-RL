from env import  *
from creature import *
# 创建环境并运行
world = World(num_creatures=100)

world.run_no_play(totol_step=10*1e3)
world.save_all('./SAVE')

