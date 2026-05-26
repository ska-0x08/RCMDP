import numpy as np
import time
import os
import matplotlib.pyplot as plt
from RCMDP_agent import Agent
from environment_River_swim import Env
from torch.distributions import Categorical

env = Env()


current_path = os.path.dirname(os.path.abspath(__file__))
timestamp = time.strftime("%Y%m%d-%H%M%S")  


## 初始化超参数
T = 1000                         
STATE_DIM = 6                               
ACTION_DIM = 2
Constraintion_DIM = 2

max_steps = 1000
C_KL = 0.1
alpha = 0.004
tau = 10000
tol = 1e-8
b = [0, 42]
gamma = 0.99
alpha_kle = 0.004
lamb = 10
S = STATE_DIM
A = ACTION_DIM
## 初始化策略
pi = np.ones((S, A)) / A
## 初始化初始状态分布
rho = [1, 0, 0, 0, 0, 0]
## 初始化环境table
p_0 = np.zeros((S, A, S))
# swim left  action:0
p_0[0, 0, 0] = 0.9
p_0[0, 0, 1] = 0.1
for s in range(1, 6):
    p_0[s, 0, s] = 0.6
    p_0[s, 0, s - 1] = 0.3
    p_0[s, 0, min(s + 1, 5)] += 0.1
# swim right action:1
for s in range(0, 5):
    p_0[s, 1, s] = 0.6
    p_0[s, 1, max(s - 1, 0)] += 0.1
    p_0[s, 1, s + 1] = 0.3
p_0[5, 1, 5] = 0.9
p_0[5, 1, 4] = 0.1
## 初始化c_0, c_1
c = [np.array([0.001, 0.0, 0.0, 0.0, 0.1, 1.0]), 
    np.array([0.2, 0.035, 0.0, 0.01, 0.08, 0.9])]

agent = Agent(rho, p_0, C_KL, S, A, gamma, tau, tol, alpha, lamb)



## 电子水印
print("What is mind?")
print ("No matter.")
print ("What is matter?")
print ("Never mind.")

reward_curv = []
cost_curv = []

## 主程序
for iterator in range (T):

    s = env.reset()
    reward = 0
    cost = 0
    for i in range (max_steps):
        prob = pi[s]
        dist = Categorical(prob)
        a = dist.sample()
        s, r, cost_ = env.step(s, a)
        reward += (gamma ** i) * r
        cost += (gamma ** i) * cost_
    reward_curv.append(reward) 
    cost_curv.append(cost)
    if (iterator + 1) % 10 == 0:
        print(
            f"Iteration {iterator + 1}: "
            f"reward = {reward:.4f}, cost = {cost:.4f}"
        )
    target_i = 0
    target_Q = np.zeros((S, A))
    target_J = -1145145
    for i in range (Constraintion_DIM):
        J,  Q= agent.KL_Uncertainty_Evaluator (c[i], pi, i, b[i])
        if (J > target_J):
            target_i = i
            target_Q = Q
            target_J = J


    pi = agent.update(pi, target_Q, target_i)


    ##weights = pi * np.exp(-alpha * Q)
    ##pi = weights / weights.sum(axis=1, keepdims=True)
    #sum_ = np.zeros(S)
    #for i in range (STATE_DIM):
    #    for j in range (ACTION_DIM):
    #        sum_[i] += pi[i][j] * exp(-alpha * Q[i][j])
    #for i in range (STATE_DIM):
    #    for j in range (ACTION_DIM):
    #        pi[i][j] = pi[i][j] * exp(-alpha * Q[i][j]) / sum_[i]

            
plt.figure(figsize=(7, 5))
plt.plot(reward_curv, label="RNPG")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("Objective Function")
plt.legend()
plt.grid(True)
plt.savefig("reward_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(cost_curv, label="RNPG")
plt.axhline(y=b[1], linestyle="--", label="baseline")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Constraint Function")
plt.legend()
plt.grid(True)
plt.savefig("cost_curve.png", dpi=300, bbox_inches="tight")
plt.show()



