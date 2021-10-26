from TicTacToe_env import TicTacToeEnv
import random, numpy as np, torch
from exp_rep import Experience_Replay
import matplotlib.pyplot as plt


EPISODES = 50_000
EPSILON_DECAY = 0.999
RECORD_EVERY = 100
learning_rate = 1e-4
discount = 0.9
epsilon = 1
scores = []
losses = []

model = torch.nn.Sequential(
    torch.nn.Linear(27, 3**4),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(3**4, 3**5),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(3**5, 3**3),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(3**3, 9))
loss_fn = torch.nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def convert_state(state):
    a = np.zeros([3,3])
    b = np.zeros([3,3])
    c = np.zeros([3,3])

    for i in range(0,3):
        for j in range(0,3):
            if state[i][j]==0:
                a[i][j] = 1
            elif state[i][j]==1:
                b[i][j] = 1
            elif state[i][j]==2:
                c[i][j] = 1
    numpy_ = np.array([b,c,a])

    return torch.from_numpy(numpy_).float()

def train_model():
    epsilon = 1
    env = TicTacToeEnv()
    memory = Experience_Replay()
    run = 1

    for i in range(EPISODES):
        initial_state = env.reset()
        done = False
        score = 0

        while not done:
            initial_state_reshaped = convert_state(initial_state).reshape(1, 27)
            qval = model(initial_state_reshaped)
            if np.random.random() > epsilon:
                qval_ = qval.data.numpy()
                action = np.argmax(qval_)
            else:
                action = random.randint(0,8)

            new_state, reward, done = env.step(action)
            new_state_reshaped = convert_state(new_state).reshape(1, 27)
            done_mask = 1.0 if done else 0.0
            memory.put((initial_state_reshaped, torch.tensor([action]), reward, new_state_reshaped, done_mask))
            initial_state = new_state
            score += reward
            if done: break

            if len(memory.buffer) > memory.min_len:
                s_T, a_T, r_T, s2_T, d_T = memory.sample()
                Q1 = model(s_T).squeeze()
                with torch.no_grad():
                    Q2 = model(s2_T).squeeze()
                    max_Q2 = Q2.max(1)[0].unsqueeze(1)

                Y = (r_T + discount * ((1 - d_T) * max_Q2)).float()
                X = Q1.gather(1, a_T)

                loss = loss_fn(X, Y.detach())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        run += 1
        if run%RECORD_EVERY==0: scores.append(score)

        if epsilon > 0.2:
            epsilon = epsilon*EPSILON_DECAY
        else: epsilon = 0

def test_env():
    env = TicTacToeEnv()
    env.reset()
    initial_state = env.reset()
    done = False
    score = 0
    for i in range(3):
        while not done:
            action = torch.argmax(model(convert_state(initial_state).reshape(1,27)))
            new_state, reward, done = env.step(action)
            print(new_state)
            score += reward
            initial_state = new_state
            
            if done: break
    print(model(convert_state(np.array([[0,0,0], [0,0,0], [0,0,0]])).reshape(1, 27)))
    print(model(convert_state(np.array([[1,2,0], [1,0,2], [0,0,0]])).reshape(1, 27)))
    print(model(convert_state(np.array([[1,2,0], [1,2,0], [0,0,0]])).reshape(1, 27)))

def show_graph():
    plt.figure(figsize=(10,7))
    plt.plot(scores)
    plt.xlabel("Epochs(episodes) per 33",fontsize=22)
    plt.ylabel("Rewards(scores)",fontsize=22)
    plt.show()

    plt.plot(losses)
    plt.xlabel("Steps",fontsize=22)
    plt.ylabel("Losses",fontsize=22)
    plt.show()
    #torch.save(model, 'model_Test.pt')

train_model()
show_graph()
test_env()