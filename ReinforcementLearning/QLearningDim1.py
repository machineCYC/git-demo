import numpy as np
import pandas as pd
import time

for i in range(1, 5):
    if i %2 == 0:
        interaction = "even %s" % i
        print("\r{}".format(interaction), end="")
    else:
        interaction = "odd %s" % i
        print("\r{}".format(interaction), end="")

# set seed
np.random.seed(2)

N_STATES = 6
ACTION = ["left", "right"]
EPSILON = 0.9
ALPHA = 0.1
MAX_EPISODES = 13
GAMMA = 0.9
FREASH_TIME = 0.3

def Build_q_table(n_states, action):
    table = pd.DataFrame(np.zeros([n_states, len(action)]), columns=action)
    return table

def choose_action(state, q_table):
    state_action = q_table.ix[state,]
    if (np.random.uniform(0, 1) > EPSILON) or (state_action.all() == 0):
        action_name = np.random.choice(ACTION)
    else:
        action_name = state_action.argmax()
    return action_name

def get_env_feedback(S_cur, A):
    if A == "right":
        if S_cur == N_STATES - 2:
            S_new = "terminal"
            R = 1
        else:
            S_new = S_cur + 1
            R = 0
    else:
        R = 0
        if S_cur == 0:
            S_new = S_cur
        else:
            S_new = S_cur - 1
    return S_new, R

def update_env(S, episode, step_counter):
    env_list = ["-"] * (N_STATES-1) + ["T"]
    if S == "terminal":
        interaction = "Episode %s: total_steps= %s" % (episode + 1, step_counter)
        print("\r{}".format(interaction))
        time.sleep(1)
    else:
        env_list[S] = "o"
        interaction = "".join(env_list)
        print("\r{}".format(interaction), end = "")
        time.sleep(FREASH_TIME)

def lr():
    q_table = Build_q_table(N_STATES, ACTION)

    for episode in range(MAX_EPISODES):
        step_counter = 0
        S_cur = 0
        is_terminated = False
        update_env(S_cur, episode, step_counter)

        while not is_terminated:
            A = choose_action(S_cur, q_table)
            S_new, R = get_env_feedback(S_cur, A)
            q_predict = q_table.ix[S_cur, A]
            if S_new != "terminal":
                q_target = R + GAMMA*q_table.iloc[S_new,].max()
            else:
                q_target = R
                is_terminated = True

            q_table.ix[S_cur, A] += ALPHA * (q_target - q_predict)
            S_cur = S_new

            step_counter += 1
            update_env(S_cur, episode, step_counter)
    return q_table


if __name__ == "__main__":
    q_table = lr()
    print(q_table)
