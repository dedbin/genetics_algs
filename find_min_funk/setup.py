import math
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
from numba import njit
from short_path.algelitism import eaSimpleElitism
import time

LOW_HIM, UP_HIM = -5, 5
ETA = 20
LENGTH_CHROM = 2

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.2
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 5

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
RND_SEED = 42
rnd.seed(RND_SEED)

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xgrid, ygrid = np.meshgrid(x, y)

f_himblau = (xgrid**2 + ygrid - 11)**2 + (xgrid + ygrid**2 - 7)**2

plt.ion()
fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

ax.set_xlim(LOW_HIM-3, UP_HIM+3)
ax.set_ylim(LOW_HIM-3, UP_HIM+3)

@njit
def eggholder(individual):
    x, y = individual
    f = -(y + 47) * math.sin(math.sqrt(abs((x / 2) + (y + 47)))) - x * math.sin(math.sqrt(abs(x - (y + 47))))
    return f,

@njit
def himmelblau(individual):
    x, y = individual
    f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return f,

def show(population, ax, xgrid, ygrid, f):
    ptMins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584458, -1.848126]]

    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(*zip(*ptMins), marker='X', color='red', zorder=1)
    ax.scatter(*zip(*population), color='green', s=2, zorder=0)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.2)