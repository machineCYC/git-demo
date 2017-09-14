"""Genetic Algorithm Example1
DNA population is close to the maximum value of the target function by GA
Use visualization to observe this phenomenon
"""

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10
POP_SIZE = 100
CROSS_RATE = 0.8 # mating probability
MUTATION_RATE = 0.03 # mutation probability
N_GENERATIONS = 300
DNA_BOUND = [0, 10]

# generate the initial population DNA
pop = np.random.randint(0, 2, size=(POP_SIZE, DNA_SIZE))

# convert the binary DNA to range(0, 5)
def tanslateDNA2x(pop):
    return pop.dot(2 ** np.arange(0, DNA_SIZE)[::-1])/(2 ** DNA_SIZE - 1)*DNA_BOUND[1]

# target function
def F(x):
    return np.sin(10*x)*x + np.cos(2*x)*x

# calculate the value of fitness
def get_fitness(value):
    return value - np.min(value) + 1e-3

# choose the good DNA from population
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness/fitness.sum())
    return pop[idx]

# generate child from parent
def crossover(parent_F, pop):
    if np.random.rand() < CROSS_RATE:
        idx_M = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
        parent_F[cross_points] = pop[idx_M, cross_points]
    return parent_F

# mutation in the child
def mutate(child):
    for dna_index in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
           if child[dna_index] == 1:
               child[dna_index] = 0
           else:
               child[dna_index] = 1
    return child

# dynamic plot
plt.ion()
x = np.linspace(DNA_BOUND[0], DNA_BOUND[1], 300)
plt.plot(x, F(x))

for generation in range(N_GENERATIONS):
    F_value = F(tanslateDNA2x(pop))
    F_avg = round(F_value.mean(), 3)

    if "plot_points" in globals():
        plot_points.remove()
        plot_text1.remove()
        plot_text2.remove()

    plot_points = plt.scatter(tanslateDNA2x(pop), F_value, c="red")
    plot_text1 = plt.text(0, 15, "generation: {0}".format(generation))
    plot_text2 = plt.text(0, 13, "DNA average value: {0}".format(F_avg))
    plt.pause(0.05)

    fitness = get_fitness(F_value)
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent_F in pop:
        child = crossover(parent_F, pop)
        child = mutate(child)
        parent_F[:] = child

plt.ioff()
plt.show()

