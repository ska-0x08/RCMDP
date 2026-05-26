import random
class Env:
    def __init__ (self):
        self.dic = {
            0 : {
                "reward" : 0.001,
                "cost" : 0.2
            },
            1 : {
                "reward" : 0,
                "cost" : 0.035
            },
            2 : {
                "reward" : 0,
                "cost" : 0
            },
            3 : {
                "reward" : 0,
                "cost" : 0.01
            },
            4 : {
                "reward" : 0.1,
                "cost" : 0.08
            },
            5 : {
                "reward" : 1,
                "cost" : 0.9
            }
        }

    def reset(self):
        return 0
    

    def step (self, state, action):
        n = random.randint(1, 10)
        if state == 0 and action == 0:
            if n == 1:
                return state + 1, self.dic[state]["reward"], self.dic[state]["cost"]
            else :
                return state, self.dic[state]["reward"], self.dic[state]["cost"]
            
        if state == 5 and action == 1:
            if n == 1:
                return state - 1, self.dic[state]["reward"], self.dic[state]["cost"]
            else :
                return state, self.dic[state]["reward"], self.dic[state]["cost"]

        if state > 0 and action == 0:
            new_state = min (state + 1, 5)
            if n == 1:
                return new_state, self.dic[state]["reward"], self.dic[state]["cost"]
            elif n <= 7:
                return state, self.dic[state]["reward"], self.dic[state]["cost"]
            else :
                return state - 1, self.dic[state]["reward"], self.dic[state]["cost"]
        if state < 5 and action == 1:
            if n == 1:
                new_state = max(0, state - 1)
                return new_state, self.dic[state]["reward"], self.dic[state]["cost"]
            elif n <= 7:
                return state, self.dic[state]["reward"], self.dic[state]["cost"]
            else :
                return state + 1, self.dic[state]["reward"], self.dic[state]["cost"]