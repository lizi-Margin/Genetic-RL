from env import  *
from creature import *
# 创建环境并运行
me = FlyingCreature(-1)
world = World(num_creatures=100)

world.execute(me)

#world.load_all('./AUTOSAVE-GAME-MODE')
world.load_all('./001SarsaSAVE')
world.run()
world.save_all('./SAVE-GAME-MODE')
