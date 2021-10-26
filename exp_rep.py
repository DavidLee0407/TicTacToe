import collections
from collections import deque
import torch, gym, numpy as np, random

def change_to_tensor_float(np):
    return torch.from_numpy(np).float()
def change_to_tensor_integer(np):
    return torch.from_numpy(np).int()
def list_to_numpy(lst):
    return np.array(lst)

class Experience_Replay:
    def __init__(self):
        self.maxlen = 2048
        self.buffer = collections.deque(maxlen=self.maxlen)
        self.batch_size = 64
        self.min_len = 256

    def put(self, transition):  # transition: [s, a, r, s']
        self.buffer.append(transition)
    
    def sample(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        s_list, a_list, r_list, s_prime_list, done_list = [], [], [], [], []

        for transitions in mini_batch:
            state, action, reward, new_state, done = transitions
            s_list.append(state)
            a_list.append(action)
            r_list.append([reward])
            s_prime_list.append(new_state)
            done_list.append([done])
        
        s_list = torch.stack(s_list)
        a_list = torch.stack(a_list)
        #r_list = torch.stack(r_list)
        s_prime_list = torch.stack(s_prime_list)
        #done_list = torch.stack(done_list)

        return torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
            torch.tensor(r_list), torch.tensor(s_prime_list, dtype=torch.float), \
                torch.tensor(done_list)
