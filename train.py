<<<<<<< HEAD
from env import  *
from creature import *
# 创建环境并运行
world = World(20,num_creatures_rand=200)

world.load_all('./AUTOSAVE',to_id=20)
world.run_no_play(totol_step=10*1e3)
world.save_all('./SAVE')

=======
from env import  *
from creature import *
# 创建环境并运行
world = World(20,num_creatures_rand=200)

world.load_all('./AUTOSAVE',to_id=20)
world.run_no_play(totol_step=10*1e3)
world.save_all('./SAVE')

>>>>>>> main
