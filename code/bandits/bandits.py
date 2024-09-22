#! python3

import numpy as np
import matplotlib.pyplot as plt



# Use the run exploration algorithm we have provided to get you
# started, this returns the expected rewards after running an exploration 
# algorithm in the K-Armed Bandits problem. We have already specified a number
# of parameters specific to the 10-armed testbed for guidance.



def runExplorationAlgorithm(explorationAlgorithm, param, seed = 1003):
    cumRewards = []
    t = 1000
    k = 10
    iters = 1000
    np.random.seed(seed)
    meanRewards = np.random.normal(1, 1, (iters, k))
    for i in range(iters):    
        n = np.zeros(k)
        currentRewards = explorationAlgorithm(param, t, k, meanRewards[i])
        cumRewards.append(currentRewards)
    expectedRewards = np.mean(cumRewards, axis=0)
    return expectedRewards



def epsilonGreedyExploration(epsilon, steps, k, meanRewards):
    expectedRewards = np.zeros(steps)
    n = np.zeros(k)
    Q = np.zeros(k)
    for t in range(steps):
        max_Q = np.max(Q)
        argmaxs = np.where(Q == max_Q)[0]
        if np.random.rand() < epsilon:
            a = np.random.choice(k)
        else:
            a = np.random.choice(argmaxs)
        reward = np.random.normal(meanRewards[a], 1)
        n[a] += 1
        Q[a] += (reward - Q[a]) / n[a]
        # compute expected reward analytically 
        expectedRewards[t] = (epsilon) * meanRewards.mean() + (1 - epsilon) * np.array([meanRewards[a] for a in argmaxs]).mean()

    return(expectedRewards)



def optimisticInitialization(value, steps, k, meanRewards):
    n = np.zeros(k)
    expectedRewards = np.zeros(steps)
    Q = np.ones(k) * value
    for t in range(steps):
        max_Q = np.max(Q)
        argmaxs = np.where(Q == max_Q)[0]
        a = np.random.choice(argmaxs)
        reward = np.random.normal(meanRewards[a], 1)
        n[a] += 1
        Q[a] += (reward - Q[a]) / n[a]
        expectedRewards[t] = np.array([meanRewards[a] for a in argmaxs]).mean()
    return(expectedRewards)



def ucbExploration(c, steps, k, meanRewards):
    expectedRewards = np.zeros(steps)
    Q = np.zeros(k)
    ts = 0
    n = np.zeros(k)
    for a in range(k):
        reward = np.random.normal(meanRewards[a], 1)
        n[a] += 1
        Q[a] += (reward - Q[a]) / n[a]
        expectedRewards[ts] = reward
        ts += 1
    for t in range(ts, steps):
        ucb_values = Q + c * np.sqrt(np.log(t+1) / n)
        max_ucb = np.max(ucb_values)
        argmaxs = np.where(ucb_values == max_ucb)[0]
        a = np.random.choice(argmaxs)
        reward = np.random.normal(meanRewards[a], 1)
        n[a] += 1
        Q[a] += (reward - Q[a]) / n[a]
        expectedRewards[t] = np.array([meanRewards[a] for a in argmaxs]).mean()
    return(expectedRewards)



def boltzmannExploration(temperature, steps, k, meanRewards):
    expectedRewards = np.zeros(steps)
    Q = np.zeros(k)
    n = np.zeros(k)
    for t in range(steps):
        max_Q = np.max(temperature * Q)
        exp_Q = np.exp(temperature * Q - max_Q)
        probabilities = exp_Q / np.sum(exp_Q)
        a = np.random.choice(k, p=probabilities)
        reward = np.random.normal(meanRewards[a], 1)
        n[a] += 1
        Q[a] += (reward - Q[a]) / n[a]
        expectedRewards[t] = (probabilities * meanRewards).sum()
    return(expectedRewards)



# plot template
def plotAlgorithms(alg_param_list, title='', paramname=''):
    plt.figure()
    algs = [alg for (alg, _) in alg_param_list]
    algs_unique = list(set(algs))
    alg_to_name = {epsilonGreedyExploration : 'Epsilon Greedy',
                   optimisticInitialization : 'Optimistic Initialization',
                   ucbExploration: 'UCB Exploration',
                   boltzmannExploration: 'Boltzmann Exploration'}
    
    alg_to_paramname = {
        epsilonGreedyExploration: r'$\epsilon$',
        optimisticInitialization: 'init val',
        ucbExploration: 'c',
        boltzmannExploration: 'temp'
    }
    alg_to_paramnames = {
        epsilonGreedyExploration: r'$\epsilon$',
        optimisticInitialization: 'Initial Values',
        ucbExploration: 'c',
        boltzmannExploration: 'Temperature'
    }
    
    label_prefixes = [f'{alg_to_name[alg]}, {alg_to_paramname[alg]} = ' for alg in algs]
    param_names = [alg_to_paramname[alg] for alg in algs]
    if len(algs_unique)==1:
        label_prefixes = ['' for _ in algs]
    

    params = [param for (_, param) in alg_param_list]
    meanRewards = []
    for i,(alg, param) in enumerate(alg_param_list):
        expectedRewards = runExplorationAlgorithm(alg, param)
        meanRewards.append(expectedRewards.mean())
        label = f'{label_prefixes[i]}{param}'
        plt.plot(expectedRewards, label=label)
    

    plt.xlabel('Steps')
    plt.ylabel('Expected Reward')
    plt.legend(title=alg_to_paramnames[algs_unique[0]] if len(algs_unique)==1 else '')
    if title:
        plt.title(title)
    plt.savefig(f'{title}.pdf')

    i = np.argmax(meanRewards)
    best_param = params[i]
    return best_param



if __name__ == '__main__':
    np.random.seed(10003)
    # Q1
    epsilons = [0, 0.001, 0.01, 0.1, 1.0]
    alg_param_list = [(epsilonGreedyExploration, epsilon) for epsilon in epsilons]
    best_epsilon = plotAlgorithms(alg_param_list, title='Epsilon Greedy Exploration', paramname='epsilon')
    
    # Q2
    initial_values = [0, 1, 2, 5, 10]
    alg_param_list = [(optimisticInitialization, value) for value in initial_values]
    best_init_value = plotAlgorithms(alg_param_list, title='Optimistic Initialization', paramname='init val')
    
    # Q3
    cs = [0, 1, 2, 5]
    alg_param_list = [(ucbExploration, c) for c in cs]
    best_c = plotAlgorithms(alg_param_list, title='UCB Exploration', paramname='c')
    
    # Q4
    temperatures = [1, 3, 10, 30, 100]
    alg_param_list = [(boltzmannExploration, temp) for temp in temperatures]
    best_temp = plotAlgorithms(alg_param_list, title='Boltzmann Exploration', paramname='temp')
    
    # Q5
    alg_param_list = [
        (epsilonGreedyExploration, best_epsilon),
        (optimisticInitialization, best_init_value),
        (ucbExploration, best_c),
        (boltzmannExploration, best_temp)
    ]
    plotAlgorithms(alg_param_list, title='Best Exploration Strategies')
