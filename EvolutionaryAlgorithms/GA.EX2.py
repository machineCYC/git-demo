"""Genetic Algorithm Example2
The best DNA in population match the target sentence "CYC is here!" by GA
Print the result to observe this phenomenon
"""

import numpy as np

TARGET_SENTENCE = "CYC is here!"
DNA_SIZE = len(TARGET_SENTENCE)
POP_SIZE = 100
CROSS_RATE = 0.6
MUTATION_RATE = 0.01

# convert string(target sentence) to number
TARGET_ASCII = np.fromstring(TARGET_SENTENCE, dtype=np.uint8)

# generate the initial population DNA
POP = np.random.randint(32, 126, size=(POP_SIZE, DNA_SIZE)).astype(np.int8)

# calculate the value of fitness
def get_fitness(pop, target_ascii):
    count_match = (pop == target_ascii).sum(axis=1)
    return count_match

# choose the good DNA from population
def select(pop, fitness):
    fitness = fitness + 1e-5 # add a small number, avoid all zero fitness
    select_idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True, p=fitness / fitness.sum())
    return POP[select_idx]

# generate child from parent
def crossover(parent_F, pop, cross_rate):
    if np.random.random() < cross_rate:
        idx = np.random.randint(0, POP_SIZE, size=1) # choose parent_M from population
        cross_position = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool) # define the crossover position
        parent_F[cross_position] = pop[idx, cross_position] # the new DNA replaces the old one(parent_F)
    return parent_F

# mutation in the child
def mutate(child, mutation_rate):
    for i in range(DNA_SIZE):
        if np.random.random() < mutation_rate:
            child[i] = np.random.randint(32, 126, size=1)
    return child

# convert the string DNA to number
def translateDNA2num(DNA):
    return DNA.tostring().decode("ascii")

Not_terminal = True
g = 0
while Not_terminal:
    fitness = get_fitness(POP, TARGET_ASCII)
    POP = select(POP, fitness)
    Best_DNA = POP[np.argmax(fitness)]
    Best_sentence = translateDNA2num(Best_DNA)

    if Best_sentence == TARGET_SENTENCE:
        Not_terminal = False

    print("Gen", g, ":", Best_sentence)
    POP_copy = POP.copy()
    for parent_F in POP:
        child = crossover(parent_F, POP_copy, CROSS_RATE)
        child = mutate(child, MUTATION_RATE)
        parent_F[:] = child
    g += 1





