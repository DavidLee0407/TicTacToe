import numpy as np
import random

def check_if_end(state, turn):  # turn: 1 or 2, 5개씩

    count = 0

    for j in range(0, 3):
        for i in range(0, 3):
            if state[i][j] == turn:
                count += 1
        if count==3: return True
        else: count = 0

    for j in range(0, 3):
        for i in range(0, 3):
            if state[j][i] == turn:
                count += 1
        if count==3: return True
        else: count = 0

    if (state[0][0]==turn) and (state[1][1]==turn) and (state[2][2]==turn):
        return True
    elif (state[2][0]==turn) and (state[1][1]==turn) and (state[0][2]==turn):
        return True

class TicTacToeEnv:

    reward = 0

    def init(self):
        self.rows = 3
        self.columns = 3
        self.state = np.zeros((self.rows, self.columns))
        self.turn = 1   #1: O 차례, 2: X 차례
        self.available_spaces = self.rows*self.columns

    def step(self, action):

        done = False
        reward = 0

        # if self.turn==1: self.turn==2
        # elif self.turn==2: self.turn==1

        action_index_1 = action % 3
        action_index_2 = action//3
        if self.state[action_index_2][action_index_1]==0:
            self.state[action_index_2][action_index_1] = 1
        else:
            self.state[action_index_2][action_index_1] = 1
            reward = - 10
            done = True
            return self.state, reward, done

        if (check_if_end(self.state, 1) == True):
            done = True
            reward = 4
            return self.state, reward, done

        if len(self.action_list) > 0:
            random_action = self.action_list[random.randint(0, len(self.action_list)-1)]
            action_index_1 = random_action % 3
            action_index_2 = random_action//3
            self.state[action_index_2][action_index_1] = 2
            self.action_list.remove(random_action)   

            if (check_if_end(self.state, 2) == True):
                done = True
                reward = -2
                return self.state, reward, done

        self.available_spaces -= 1
        if not done:
            if self.available_spaces == 0:
                done = True
                reward = 2
                return self.state, reward, done

        return self.state, reward, done

    # def render(self):
    #     #self.state[self.action[2]-1][self.action[1]-1] = self.turn
    #     #print(self.turn)
    #     pass

    def reset(self):
        self.state = np.zeros((3, 3))
        self.turn = 1   
        self.available_spaces = 3*3
        self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        return self.state