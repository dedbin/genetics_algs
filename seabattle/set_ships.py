import numpy as np
from deap import tools, base, creator

import numpy
import random as rnd
import matplotlib.pyplot as plt

from short_path.algelitism import eaSimpleElitism
from short_path.graph_show import show_ships

POLE_SIZE = 10
SHIPS = 10
LENGTH_CHROM = 3 * SHIPS

POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

RANDOM_SEED = 42
rnd.seed(RANDOM_SEED)


def random_ship(total):
    ships = []
    for _ in range(total):
        ships.extend([rnd.randint(1, POLE_SIZE), rnd.randint(1, POLE_SIZE), rnd.randint(0, 1)])

    return creator.Individual(ships)


def ships_fitness(individual):
    type_ship = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

    inf = 1000
    P0 = np.zeros((POLE_SIZE, POLE_SIZE))
    P = np.ones((POLE_SIZE + 6, POLE_SIZE + 6)) * inf
    P[1:POLE_SIZE + 1, 1:POLE_SIZE + 1] = P0

    th = 0.2
    h = np.ones((3, 6)) * th
    ship_one = np.ones((1, 4))
    v = np.ones((6, 3)) * th

    for *ship, t in zip(*[iter(individual)] * 3, type_ship):
        if ship[-1] == 0:
            sh = np.copy(h[:, :t + 2])
            sh[1, 1:t + 1] = ship_one[0, :t]
            P[ship[0] - 1:ship[0] + 2, ship[1] - 1:ship[1] + t + 1] += sh
        else:
            sh = np.copy(v[:t + 2, :])
            sh[1:t + 1, 1] = ship_one[0, :t]
            P[ship[0] - 1:ship[0] + t + 1, ship[1] - 1:ship[1] + 2] += sh

    s = np.sum(P[np.bitwise_and(P > 1, P < inf)])
    s += np.sum(P[P > inf + th * 4])

    return s,


def mut_ships(individual, indpb):
    for i in range(len(individual)):
        if rnd.random() < indpb:
            individual[i] = rnd.randint(0, 1) if (i + 1) % 3 == 0 else rnd.randint(1, POLE_SIZE)
    return individual,


def show(ax):
    ax.clear()
    show_ships(ax, hof.items[0], POLE_SIZE)

    plt.draw()
    plt.gcf().canvas.flush_events()


def registration():
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("random_ship", random_ship, SHIPS)
    toolbox.register("population_creator", tools.initRepeat, list, toolbox.random_ship)

    population = toolbox.population_creator(n=POPULATION_SIZE)

    toolbox.register("evaluate", ships_fitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_ships, indpb=1.0 / LENGTH_CHROM)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("mean", numpy.mean)

    return population, toolbox, stats


def show_fitness(logbook):
    max_fitness_value, mean_fitness_value = logbook.select("min", "mean")
    plt.plot(max_fitness_value, color='red')
    plt.plot(mean_fitness_value, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Dependence of maximum and average fitness on generation")
    plt.show()


def main():
    population, toolbox, stats = registration()
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    ax.set_xlim(-2, POLE_SIZE + 3)
    ax.set_ylim(-2, POLE_SIZE + 3)

    population, logbook = eaSimpleElitism(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          callback=(show, (ax,)),
                                          verbose=True)
    plt.ioff()
    plt.show()
    best = hof.items[0]
    print(f'Best individual: {best}')

    show_fitness(logbook)


if __name__ == '__main__':
    main()
