#! python3

import numpy as np
import matplotlib.pyplot as plt
import gymnasium

import lake_info



def value_func_to_policy(env, gamma, value_func):
    '''
    Outputs a policy given a value function.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute the policy for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.

    Returns
    -------
    np.ndarray
        An array of integers. Each integer is the optimal action to take in
        that state according to the environment dynamics and the given value
        function.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    # BEGIN STUDENT SOLUTION
    for state in range(env.observation_space.n):
        val = value_func[state]
        best_action = None
        best_action_val = 0
        for action in range(env.action_space.n):
          for (prob, nextstate, reward, _) in env.P[state][action]:
            action_val = prob * (reward + gamma * value_func[nextstate])
            if action_val > best_action_val or best_action == None:
              best_action_val = action_val
              best_action = action
        policy[state] = best_action
    # END STUDENT SOLUTION
    return (policy)



def evaluate_policy_sync(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    i = 0
    delta = tol
    n_states = env.observation_space.n
    while delta >= tol and i < max_iters:
        delta = 0
        old_value_func = value_func.copy()
        for state in range(n_states):
            value = old_value_func[state]
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][policy[state]]:
                new_value += prob * (reward + gamma * (not is_terminal) * old_value_func[nextstate])
            value_func[state] = new_value
            delta = max(delta, abs(value - new_value))
        i += 1
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION
    delta = tol
    n_states = env.observation_space.n
    i = 0
    while delta >= tol and i < max_iters:
        delta = 0
        for state in range(n_states):
            value = value_func[state]
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][policy[state]]:
                new_value += prob * (reward + (not is_terminal) * gamma * value_func[nextstate])
            value_func[state] = new_value
            delta = max(delta, abs(value - new_value))
        i += 1
    # END STUDENT SOLUTION
    return(value_func, i)



def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters=int(1e3), tol=1e-3):
    '''
    Performs policy evaluation.

    Evaluates the value of a policy. Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    value_func: np.ndarray
        The current value function estimate.
    gamma: float
        Discount factor, must be in range [0, 1).
    policy: np.ndarray
        The policy to evaluate, maps states to actions.
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, int)
        The value for the given policy and the number of iterations the value
        function took to converge.
    '''
    # BEGIN STUDENT SOLUTION 
    delta = tol
    n_states = env.observation_space.n
    i = 0
    while delta >= tol and i < max_iters:
        delta = 0
        for state in np.random.permutation(list(range(n_states))):
            value = value_func[state]
            new_value = 0
            for prob, nextstate, reward, is_terminal in env.P[state][policy[state]]:
                new_value += prob * (reward + (not is_terminal) * gamma * value_func[nextstate])
            value_func[state] = new_value
            delta = max(delta, abs(value - new_value))
        i += 1
    # END STUDENT SOLUTION
    return(value_func, i)



def improve_policy(env, gamma, value_func, policy):
    '''
    Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    value_func: np.ndarray
        The current value function estimate.
    policy: np.ndarray
        The policy to improve, maps states to actions.

    Returns
    -------
    (np.ndarray, bool)
        Returns the new policy and whether the policy changed.
    '''
    policy_changed = False
    # BEGIN STUDENT SOLUTION
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    for state in range(n_states):
        old_action = policy[state]

        q_values = []
        for action in range(n_actions):
            q = 0
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                q += prob * (reward + gamma * (not is_terminal) * value_func[nextstate])
            q_values.append(q)

        new_action = np.argmax(q_values)
        policy[state] = new_action

        if new_action != old_action:
            policy_changed = True
    # END STUDENT SOLUTION
    return(policy, policy_changed)



def policy_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    change = True
    while change and pe_steps < max_iters:
        value_func, pe_eval_steps = evaluate_policy_sync(env, value_func, gamma, policy, max_iters)
        policy, change = improve_policy(env, gamma, value_func, policy)
        pi_steps += 1
        pe_steps += pe_eval_steps
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    change = True
    while change and pe_steps < max_iters:
      value_func, pe_eval_steps = evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iters)
      policy, change = improve_policy(env, gamma, value_func, policy)
      pi_steps += 1
      pe_steps += pe_eval_steps
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def policy_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
        Returns optimal policy, value function, number of policy improvement
        iterations, and number of policy evaluation iterations.
    '''
    policy = np.zeros(env.observation_space.n, dtype='int')
    value_func = np.zeros(env.observation_space.n)
    pi_steps, pe_steps = 0, 0
    # BEGIN STUDENT SOLUTION
    change = True
    while change and pe_steps < max_iters:
      value_func, pe_eval_steps = evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iters)
      policy, change = improve_policy(env, gamma, value_func, policy)
      pi_steps += 1
      pe_steps += pe_eval_steps
    # END STUDENT SOLUTION
    return(policy, value_func, pi_steps, pe_steps)



def value_iteration_sync(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    i = 0
    delta = tol
    while i < max_iters:
      i += 1
      delta = 0
      new_value_func = np.zeros(env.observation_space.n)
      for state in range(env.observation_space.n):
        row = state // env.ncol
        col = state - row * env.ncol
        tile_type = env.desc[row, col]
        if tile_type == b'G' or tile_type == b'H':
           continue
        val = value_func[state]
        max_q = -float('inf')
        for action in range(env.action_space.n):
          q = 0
          for (prob, nextstate, reward, is_terminal) in env.P[state][action]:
            q += prob * (reward + gamma * value_func[nextstate])
          max_q = max(max_q, q)
        new_value_func[state] = max_q
        delta = max(delta, abs(val - new_value_func[state]))
      value_func = new_value_func.copy()
      if delta < tol:
        break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_ordered(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    i = 0
    delta = tol
    while i < max_iters:
      i += 1
      delta = 0
      for state in range(env.observation_space.n):
        row = state // env.ncol
        col = state - row * env.ncol
        tile_type = env.desc[row, col]
        if tile_type == b'G' or tile_type == b'H':
           continue

        val = value_func[state]
        max_q = -float('inf')
        for action in range(env.action_space.n):
          q = 0
          for (prob, nextstate, reward, is_terminal) in env.P[state][action]:
            q += prob * (reward + gamma * value_func[nextstate])
          max_q = max(max_q, q)
        value_func[state] = max_q
        delta = max(delta, abs(val - value_func[state]))
      if delta < tol:
        break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_randperm(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    i = 0
    delta = tol
    while i < max_iters:
      i += 1
      delta = 0
      states = np.random.permutation(np.arange(0, env.observation_space.n))
      for state in states:
        row = state // env.ncol
        col = state - row * env.ncol
        tile_type = env.desc[row, col]
        if tile_type == b'G' or tile_type == b'H':
           continue

        val = value_func[state]
        max_q = -float('inf')
        for action in range(env.action_space.n):
          q = 0
          for (prob, nextstate, reward, is_terminal) in env.P[state][action]:
            q += prob * (reward + gamma * value_func[nextstate])
          max_q = max(max_q, q)
        value_func[state] = max_q
        delta = max(delta, abs(val - value_func[state]))
      if delta < tol:
        break
    # END STUDENT SOLUTION
    return(value_func, i)



def value_iteration_async_custom(env, gamma, max_iters=int(1e3), tol=1e-3):
    '''
    Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to compute value iteration for.
    gamma: float
        Discount factor, must be in range [0, 1).
    max_iters: int
        The maximum number of iterations to run before stopping.
    tol: float
        Determines when value function has converged.

    Returns
    -------
    (np.ndarray, iteration)
        Returns the value function, and the number of iterations it took to
        converge.
    '''
    value_func = np.zeros(env.observation_space.n)
    # BEGIN STUDENT SOLUTION
    goal_r = -1
    goal_c = -1
    # We find the goal row and column
    for state in range(env.observation_space.n):
      row = state // env.ncol
      col = state - row * env.ncol
      tile_type = env.desc[row, col]
      if tile_type == b'G':
          goal_r = row
          goal_c = col
          break
    
    def manhattan(x):
      row = x // env.ncol
      col = x - row * env.ncol
      return abs(row - goal_r) + abs(col - goal_c)

    i = 0
    delta = tol
    while i < max_iters:
      i += 1
      delta = 0
      state_manhattans = list(map(manhattan, [x for x in range(env.observation_space.n)]))
      states = np.argsort(state_manhattans)

      for state in states:
        row = state // env.ncol
        col = state - row * env.ncol
        tile_type = env.desc[row, col]
        if tile_type == b'G' or tile_type == b'H':
           continue

        val = value_func[state]
        max_q = -float('inf')
        for action in range(env.action_space.n):
          q = 0
          for (prob, nextstate, reward, is_terminal) in env.P[state][action]:
            q += prob * (reward + gamma * value_func[nextstate])
          max_q = max(max_q, q)
        value_func[state] = max_q
        delta = max(delta, abs(val - value_func[state]))
      if delta < tol:
        break
    # END STUDENT SOLUTION
    return(value_func, i)



# Here we provide some helper functions for your convinience.

def display_policy_letters(env, policy):
    '''
    Displays a policy as an array of letters.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    policy: np.ndarray
        The policy to display, maps states to actions.
    '''
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_info.actions_to_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.unwrapped.nrow, env.unwrapped.ncol)

    for row in range(env.unwrapped.nrow):
        print(''.join(policy_letters[row, :]))



def value_func_heatmap(env, value_func):
    '''
    Visualize a policy as a heatmap.

    Parameters
    ----------
    env: gymnasium.core.Environment
        The environment to display the policy for.
    value_func: np.ndarray
        The current value function estimate.
    '''
    fig, ax = plt.subplots(figsize=(7,6))

    # Reshape value_func to match the environment dimensions
    heatmap_data = np.reshape(value_func, [env.unwrapped.nrow, env.unwrapped.ncol])

    # Create a heatmap using Matplotlib
    cax = ax.matshow(heatmap_data, cmap='GnBu_r')

    # Set ticks and labels
    ax.set_yticks(np.arange(0, env.unwrapped.nrow))
    ax.set_xticks(np.arange(0, env.unwrapped.ncol))
    ax.set_yticklabels(np.arange(1, env.unwrapped.nrow + 1)[::-1])
    ax.set_xticklabels(np.arange(1, env.unwrapped.ncol + 1))

    # Display the colorbar
    cbar = plt.colorbar(cax)

    plt.show()



if __name__ == '__main__':
    np.random.seed(10003)
    maps = lake_info.maps
    gamma = 0.9

    for map_name, mapping in maps.items():
        print(f"Using map {map_name}")
        env = gymnasium.make('FrozenLake-v1', desc=mapping, map_name=map_name, is_slippery=False)
        
        # BEGIN STUDENT SOLUTION


        # print("policy sync")
        # policy, value_func, pi_steps, pe_steps = policy_iteration_sync(env, gamma=gamma)
        # print(f"pi_steps: {pi_steps}, pe_steps: {pe_steps}") 
        # display_policy_letters(env, policy)
        # value_func_heatmap(env, value_func)
        # plt.show()

        # print("value sync")
        # value_func, iters = value_iteration_sync(env, gamma, max_iters=int(1e4), tol=1e-3)
        # print(f"{iters} iterations")
        # value_func_heatmap(env, value_func)
        # plt.show()
        # policy = value_func_to_policy(env, gamma, value_func)
        # display_policy_letters(env, policy)
    
        # print("policy async ordered")
        # policy, value_func, pi_steps, pe_steps = policy_iteration_async_ordered(env, gamma=gamma)
        # print(f"pi_steps: {pi_steps}, pe_steps: {pe_steps}") 
        # display_policy_letters(env, policy)
        # value_func_heatmap(env, value_func)
        # plt.show()

        # print("policy async randperm")
        # pi_steps_list, pe_steps_list = [], []
        # for _ in range(10):
        #     policy, value_func, pi_steps, pe_steps = policy_iteration_async_randperm(env, gamma=gamma)
        #     pi_steps_list.append(pi_steps)
        #     pe_steps_list.append(pe_steps) 
        #     print(f"pi_steps: {pi_steps}, pe_steps: {pe_steps}") 
        # print(f"mean pi_steps: {np.mean(pi_steps)}, mean pe_steps: {np.mean(pe_steps)}") 

        # print("value async ordered")
        # value_func, iters = value_iteration_async_ordered(env, gamma, max_iters=int(1e4), tol=1e-3)
        # print(f"{iters} iterations")
        # value_func_heatmap(env, value_func)
        # plt.show()
        # policy = value_func_to_policy(env, gamma, value_func)
        # display_policy_letters(env, policy)

        # print("value async randperm")
        # iters_list = []
        # for _ in range(10):
        #     value_func, iters = value_iteration_async_randperm(env, gamma, max_iters=int(1e4), tol=1e-3)
        #     iters_list.append(iters)
        #     print(f"{iters} iterations")
        # print(f"{np.mean(iters)} mean iterations")

        print("value async manhattan")
        value_func, iters = value_iteration_async_custom(env, gamma, max_iters=int(1e4), tol=1e-3)
        print(f"{iters} iterations")
        value_func_heatmap(env, value_func)
        plt.show()
        policy = value_func_to_policy(env, gamma, value_func)
        display_policy_letters(env, policy)


        # END STUDENT SOLUTION
