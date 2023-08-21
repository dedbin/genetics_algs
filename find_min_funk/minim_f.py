from setup import *


def random_point(a, b):
    return [rnd.uniform(a, b), rnd.uniform(a, b)]


def registration():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("random_point", random_point, LOW_HIM, UP_HIM)
    toolbox.register("individual_creator", tools.initIterate, creator.Individual, toolbox.random_point)
    toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)

    population = toolbox.population_creator(n=POPULATION_SIZE)

    toolbox.register("evaluate", himmelblau)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW_HIM, up=UP_HIM, eta=ETA)
    toolbox.register("mutate", tools.mutPolynomialBounded,  low=LOW_HIM, up=UP_HIM, eta=ETA, indpb=1.0/LENGTH_CHROM)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("mean", np.mean)

    return population, toolbox, stats


def main():
    population, toolbox, stats = registration()

    population, logbook = eaSimpleElitism(population, toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          callback=(show, (population, ax, xgrid, ygrid, f_himblau)),
                                          verbose=True)

    max_fitness_value, mean_fitness_value = logbook.select("min", "mean")

    best = hof.items[0]
    print(f'Best individual: {best}')

    plt.ioff()
    plt.show()

    plt.plot(max_fitness_value, color='red')
    plt.plot(mean_fitness_value, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Dependence of maximum and average fitness on generation")
    plt.show()


if __name__ == '__main__':
    main()
