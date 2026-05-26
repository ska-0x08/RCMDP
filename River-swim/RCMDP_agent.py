import numpy as np

class Agent():



    def __init__ (self, rho, p_0, C_KL, S, A, gamma, tau, tol, alpha, lamb):
        self.S = S
        self.A = A
        self.gamma = gamma
        self.tau = tau
        self.tol = tol
        self.C_KL = C_KL
        self.p_0 = p_0
        self.rho = np.asarray(rho, dtype=float)
        self.alpha = alpha
        self.lamb = lamb

    def Robust_Q_table (self, c_i, pi, obj_type):

        p_0 = self.p_0
        S = self.S
        A = self.A
        gamma = self.gamma
        tau = self.tau
        tol = self.tol
        C_KL = self.C_KL


        Q = np.zeros((S, A))
        V = np.zeros(S)
        if obj_type == 0 :
            sign = -1
        else :
            sign = 1
        for i in range (tau):

             #? 修改点 ---------------------------------------------------
            z = sign * V / C_KL
            z = z - np.max(z)
            exp_V = np.exp(z)
            #exp_V = np.exp(sign * (V - np.max(V)) / C_KL)   # shape: (S,)
            #? ---------------------------------------------------------
            weights = p_0 * exp_V[None, None, :]
            p_star = weights / weights.sum(axis=2, keepdims=True)

            expected_V = p_star @ V
            c_pi = c_i
            Q = c_pi[:, None] + gamma * expected_V 
            #V_new = pi @ Q
            V_new = np.sum(pi * Q, axis=1)
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new
        return Q, V

    def KL_Uncertainty_Evaluator(self, c_i, pi, obj_type, b):
        if obj_type == 0 :
            sign = -1
        else :
            sign = 1    


        c_eval = c_i.copy()
        if obj_type == 0:
            c_eval = c_eval / self.lamb


        p_0 = self.p_0
        gamma = self.gamma
        C_KL = self.C_KL        
        rho = self.rho




        Q, V = self.Robust_Q_table(c_eval, pi, obj_type)
        ##——————————————————————————————————————————————————————————————————————
        ## 为了数值稳定，减去 V.max() ？？？
        #? 修改点 ---------------------------------------------------
        z = sign * V / C_KL
        z = z - np.max(z)
        exp_V = np.exp(z)
        #exp_V = np.exp(sign * (V - np.max(V)) / C_KL)   # shape: (S,)
        #? ---------------------------------------------------------
        ## p_0 shape: (S, A, S)
        ##exp_V[None, None, :] shape: (1, 1, S)
        weights = p_0 * exp_V[None, None, :]

        ## 对 next_state 维度归一化
        p_star = weights / weights.sum(axis=2, keepdims=True)
        #weight = np.zeros ((S, A))
        #p_star = np.zeros((S, A, S))
        #for s in range (S):
        #    for a in range (A):
        #       for s_ in range (S):
        #            weight[s][a] += p_0[s][a][s_] * np.exp(V[s_] / C_KL)
        #for s in range (S):
        #    for a in range (A):
        #        for s_ in range (S):
        #            p_star[s][a][s_] = p_0[s][a][s_] * np.exp(V[s_] / C_KL) / weight[s][a]
        ##———————————————————————————————————————————————————————————————————————
        #T = np.zeros((S, S))
        #for s in range (S):
        #    for s_ in range (S):
        #        for a in range (A):
        #            T[s][s_] += pi[s][a] * p_star[s][a][s_]
        #T = (pi[:, :, None] * p_star).sum(axis=1)
        ##———————————————————————————————————————————————————————————————————————
        ## 本篇论文该任务的 c(s, a) 比较特殊， 我们令  c(s, a) = c(s)
        ## 本来应该是  c^{\pi}_i[s] = \sum_{a\in \mathbb{A}} \pi (a \vert s) c_i (s, a) 
        c_pi = c_eval
        
        ##———————————————————————————————————————————————————————————————————————
        ## 这里为什么不要迭代？？？？？
        #V = c_pi + gamma * T @ V
        ## NOTICE
        
        
        ##———————————————————————————————————————————————————————————————————————
        expected_V = p_star @ V
        Q = c_pi[:, None] + gamma * expected_V
        #Q = np.zeros((S, A))
        #for s in range (S):
        #    for a in range (A):
        #       Q[s][a] += c_pi[s]
        #       for s_ in range (S):
        #            Q[s][a] += gamma * p_star[s][a][s_] * V[s_]
        ##———————————————————————————————————————————————————————————————————————
        
        
        
        #? 修改点1  ---------------
        V = np.sum(pi * Q, axis=1)
        #? -----------------------
        
        
        J = float(rho @ V)
        J -= b
        #J = 0
        #for s in range (S):
        #    j += rho[s] * V[s]
        return J, Q


    def update (self, pi, Q, obj_type):
        alpha = self.alpha
        
        
        
        #? 修改点2------------------------------------------
        logits = -alpha * Q
        if obj_type == 0:
            logits *= -1

        logits = logits - logits.max(axis=1, keepdims=True)
        weights = pi * np.exp(logits)
        #? ------------------------------------------------



        
        pi = weights / weights.sum(axis=1, keepdims=True)
        return pi