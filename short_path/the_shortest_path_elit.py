from deap import base, tools, creator, algorithms
import random as rnd
import matplotlib.pyplot as plt
import numpy
from graph_show import show_graph
from algelitism import eaSimpleElitism

inf = 100

D = (
    (0, 3, 1, 2, inf, inf),
    (3, 0, 4, inf, inf, inf),
    (1, 4, 0, inf, 7, 5),
    (3, inf, inf, 0, inf, 2),
    (inf, inf, 7, inf, 0, 4),
    (inf, inf, 5, 2, 4, 0)
)

startV = 0

LENGHT_D = len(D)
LENGHT_CROM = len(D[0]) * LENGHT_D

POPULATION_SIZE = 500
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 30
HALL_OF_FAME_SIZE = 1

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
RND_SEED = 42
rnd.seed(RND_SEED)


def fitness(individual):
    s = 0
    for n, path in enumerate(individual):
        path = path[:path.index(n) + 1]

        si = startV
        for j in path:
            s += D[si][j]
            si = j
    return s,


def cx_ordered(ind1, ind2):
    for p1, p2 in zip(ind1, ind2):
        tools.cxOrdered(p1, p2)
    return ind1, ind2


def show(ax):
    ax.clear()
    show_graph(ax, hof.items[0])

    plt.draw()
    plt.gcf().canvas.flush_events()
    plt.show()


def mut_shuffle_indexes(individual, indpb):
    for ind in individual:
        tools.mutShuffleIndexes(ind, indpb)
    return individual,


def main():
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("random_order", rnd.sample, range(LENGHT_D), LENGHT_D)
    toolbox.register("individual_creator", tools.initRepeat, creator.Individual, toolbox.random_order, LENGHT_D)
    toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)
    toolbox.register("evaluate", fitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", cx_ordered)
    toolbox.register("mutate", mut_shuffle_indexes, indpb=1.0 / LENGHT_CROM / 10)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("mean", numpy.mean)

    population = toolbox.population_creator(n=POPULATION_SIZE)

    plt.ion()
    fig, ax = plt.subplots()

    population, logbook = eaSimpleElitism(population, toolbox,
                                          cxpb=P_CROSSOVER / LENGHT_D,
                                          mutpb=P_MUTATION / LENGHT_D,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          callback=(show, (ax,)),
                                          verbose=True)

    max_fitness_value, mean_fitness_value = logbook.select("min", "mean")
    plt.ioff()
    plt.show()
    best = hof.items[0]
    print(f'Best individual: {best}')
    plt.plot(max_fitness_value, color='red')
    plt.plot(mean_fitness_value, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Dependence of maximum and average fitness on generation")
    plt.show()

    fig, ax = plt.subplots()
    show_graph(ax, best)
    plt.title("Best individual")
    plt.show()


if __name__ == '__main__':
    main()
