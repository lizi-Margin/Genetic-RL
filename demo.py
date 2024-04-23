<<<<<<< HEAD
from env import  *
from creature import *
# 创建环境并运行
me = FlyingCreature(-1)
world = World(100,num_creatures_rand=0)

world.execute(me)

#world.load_all('./AUTOSAVE-GAME-MODE')
world.load_all('./004-S4000',to_id=20)
world.run(train=False,plot=False)
world.save_all('./SAVE-GAME-MODE')
=======
from env import  *
from creature import *
# 创建环境并运行
me = FlyingCreature(-1)
world = World(100,num_creatures_rand=0)

world.execute(me)

#world.load_all('./AUTOSAVE-GAME-MODE')
world.load_all('./004-S4000',to_id=20)
world.run(train=False,plot=False)
world.save_all('./SAVE-GAME-MODE')
>>>>>>> main
