import numpy
from deap import base, tools, creator, algorithms
import random as rnd
import matplotlib.pyplot as plt

ONE_MAX_LENGTH = 100  # length of one max

POPULATION_SIZE = 200  # size of population
P_CROSSOVER = 0.9  # probability of crossover
P_MUTATION = 0.1  # probability of mutation
MAX_GENERATIONS = 50  # number of generations

RND_SEED = 42
rnd.seed(RND_SEED)

creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)


# Fitness function for the one max problem
def one_max_fitness(individual):
    return sum(individual),


toolbox = base.Toolbox()
toolbox.register("zero_or_one", rnd.randint, 0, 1)
toolbox.register("individual_creator", tools.initRepeat, creator.Individual, toolbox.zero_or_one, ONE_MAX_LENGTH)
toolbox.register("population_creator", tools.initRepeat, list, toolbox.individual_creator)
toolbox.register("evaluate", one_max_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)


# Main genetic algorithm function
def main():
    # Create the initial population
    population = toolbox.population_creator(n=POPULATION_SIZE)
    gen_count = 0
    fitness_values = list(map(one_max_fitness, population))

    # Assign fitness values to individuals
    for individual, fitness_values in zip(population, fitness_values):
        individual.fitness.values = fitness_values

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("mean", numpy.mean)
    population, logbook = algorithms.eaSimple(population, toolbox,
                                              cxpb=P_CROSSOVER,
                                              mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS,
                                              stats=stats,
                                              verbose=True)

    max_fitness_value, mean_fitness_value = logbook.select("max", "mean")

    # Plot the maximum and average fitness values over generations
    plt.plot(max_fitness_value, color='red')
    plt.plot(mean_fitness_value, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Dependence of maximum and average fitness on generation")
    plt.show()


if __name__ == "__main__":
    main()