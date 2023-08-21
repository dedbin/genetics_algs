import random as rnd
import matplotlib.pyplot as plt

ONE_MAX_LENGTH = 100  # length of one max

POPULATION_SIZE = 200  # size of population
P_CROSSOVER = 0.9  # probability of crossover
P_MUTATION = 0.1  # probability of mutation
MAX_GENERATIONS = 50  # number of generations

RND_SEED = 42
rnd.seed(RND_SEED)


# FitnessMax class to store fitness values
class FitnessMax:
    def __init__(self):
        self.values = [0]


# Individual class derived from list to represent individuals in the population
class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


# Fitness function for the one max problem
def one_max_fitness(individual):
    return sum(individual),


# Creator functions to initialize individuals and population
def individual_creator():
    return Individual([rnd.randint(0, 1) for _ in range(ONE_MAX_LENGTH)])


def population_creator():
    return [individual_creator() for _ in range(POPULATION_SIZE)]


# Clone function to create a deep copy of an individual
def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


# Tournament selection function
def sel_tournament(population, p_length):
    offspring = []
    for _ in range(p_length):
        i1 = i2 = i3 = 0
        # Select three unique individuals randomly
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = rnd.randint(0, p_length - 1), rnd.randint(0, p_length - 1), rnd.randint(0, p_length - 1)

        # Append the individual with the highest fitness among the three
        offspring.append((max([population[i1], population[i2], population[i3]], key=lambda x: x.fitness.values[0])))

    return offspring


# One-point crossover function
def cx_one_point(ind1, ind2):
    s = rnd.randint(2, len(ind1) - 3)
    # Swap the bits after the randomly selected crossover point
    ind1[s:], ind2[s:] = ind2[s:], ind1[s:]
    return ind1, ind2


# Bit flip mutation function
def mut_flip_bit(individual, indpb=0.01):
    for idx in range(len(individual)):
        if rnd.random() < indpb:
            # Flip the bit with probability indpb
            individual[idx] = 0 if individual[idx] == 1 else 1
    return individual


# Main genetic algorithm function
def main():
    # Create the initial population
    population = population_creator()
    gen_count = 0
    fitness_values = list(map(one_max_fitness, population))

    # Assign fitness values to individuals
    for individual, fitness_values in zip(population, fitness_values):
        individual.fitness.values = fitness_values

    max_fitness_value = []
    mean_fitness_value = []

    fitnessValues = [individual.fitness.values[0] for individual in population]

    # Iterate until the maximum fitness value reaches the target or the maximum number of generations is reached
    while max(fitnessValues) < ONE_MAX_LENGTH and gen_count < MAX_GENERATIONS:
        gen_count += 1
        # Select parents using tournament selection
        offspring = sel_tournament(population, len(population))
        # Create a deep copy of offspring
        offspring = list(map(clone, offspring))

        # Apply crossover with a certain probability
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if rnd.random() < P_CROSSOVER:
                cx_one_point(ind1, ind2)

        # Apply mutation with a certain probability
        for idx in offspring:
            if rnd.random() < P_MUTATION:
                idx = mut_flip_bit(idx, indpb=1.0 / ONE_MAX_LENGTH)

        # Calculate fitness values for the offspring
        fresh_fitness_values = list(map(one_max_fitness, offspring))
        for individual, fitness_values in zip(offspring, fresh_fitness_values):
            individual.fitness.values = fitness_values

        # Replace the population with the offspring
        population[:] = offspring

        # Update fitness values of the population
        fitnessValues = [individual.fitness.values[0] for individual in population]

        # Calculate and store the maximum and average fitness values for the generation
        max_fitness = max(fitnessValues)
        mean_fitness = sum(fitnessValues) / len(fitnessValues)
        max_fitness_value.append(max_fitness)
        mean_fitness_value.append(mean_fitness)
        print(f"Generation: {gen_count}, Max Fitness: {max_fitness}, Mean Fitness: {mean_fitness}")

        # Find the index of the best individual in the population
        best_idx = fitnessValues.index(max(fitnessValues))
        print(f"Best individual: {population[best_idx]} \n")

    # Plot the maximum and average fitness values over generations
    plt.plot(max_fitness_value, color='red')
    plt.plot(mean_fitness_value, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Dependence of maximum and average fitness on generation")
    plt.show()


if __name__ == "__main__":
    main()
