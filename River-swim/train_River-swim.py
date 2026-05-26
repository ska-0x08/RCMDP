import torch
import time
import os
import matplotlib.pyplot as plt
from RCMDP_agent import Agent
from environment_River-swim import Env
env = Env()


current_path = os.path.dirname(os.path.abspath(__file__))
timestamp = time.strftime("%Y%m%d-%H%M%S")  



T = 200                                     ## 每次我玩多少步骤
STATE_DIM = 6                               
ACTION_DIM = 2
K = 400

agent = Agent(STATE_DIM, ACTION_DIM, BATCH_SIZE, LR_ACTOR, LR_CRITIC, GAMMA, LAMBDA, EPOCH, EPS_CLIP)

print("What is mind?")
print ("No matter.")
print ("What is matter?")
print ("Never mind.")
tmp_MAX = []
for iteration_i in range(T): # 玩多少轮


    tmp_MAX.append(MAX)
    agent.update() # TODO 跟新模型
    print (f"I'v finished {iteration_i} training")
    print (f"MAX : {MAX}")


env.close()
plt.plot(range(len(tmp_MAX)), tmp_MAX)
plt.xlabel("Training Times")
plt.ylabel("Max")
plt.savefig("line_chart_MAX.png", dpi=600, bbox_inches="tight")
plt.show()

