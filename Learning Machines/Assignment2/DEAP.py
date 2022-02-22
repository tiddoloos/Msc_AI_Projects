import numpy as np
from random import *
from deap import base, creator, tools, algorithms
# from commands import *
from commands_with_green_box import *
import time
import numpy as np
from numpy import inf
from controller import *
import robobo
import signal

MU, LAMBDA = 25, 50
CXPB, MUTPB, NGEN = 0.2, 0.4, 12
STEPS = 300
input = 4
output = 3
n_vars = ((input+1) * output)
dom_u = 1
dom_l = -1

signal.signal(signal.SIGINT, terminate_program)
rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
controller = Player_controller(output)

def fitness_f(ind):
    return [simulation(rob, ind, controller, STEPS)]

def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = randint(1, size)
    cxpoint2 = randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2

def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Initialize DEAP
toolbox=base.Toolbox()
IND_SIZE = n_vars
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_f)
toolbox.register("generate", tools.initRepeat, list, toolbox.individual)
toolbox.decorate("mate", checkBounds(dom_l, dom_u))
toolbox.decorate("mutate", checkBounds(dom_l, dom_u))
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# logbook = tools.Logbook()
# logbook.header = "gen", "evals", "std", "min", "avg", "max"
# pop = toolbox.population(n=MU)
# hof = tools.HallOfFame(1, similar = lambda x,y: np.all(x==y))
# pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)

def main_mu_plus_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN):
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(3, similar = lambda x, y: np.all(x==y))
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
            cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
        return pop, logbook, hof

def experiment():
    pop, logbook, hof = main_mu_plus_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN)

    
    record = stats.compile(pop)
    print(record)
    hof1 = list(hof[0])
    hof2 = list(hof[1])
    hof3 = list(hof[2])
    f = open('Logbook/classification_fitness=' + str(record['max']) + '_NGEN=' + str(NGEN) + '_steps=' + str(STEPS) + '_MU='+str(MU)+'_LA=' +str(LAMBDA) + '.txt', 'w+')
    f.write(str(logbook))
    f.close
    f = open('HOF/classification1_fitness=' + str(record['max']) + '_NGEN=' + str(NGEN) + '_steps=' + str(STEPS) + '_MU='+str(MU)+'_LA=' +str(LAMBDA) + '.txt', 'w+')
    for i in hof1:
        f.write(str(i) +'\n')
    f.close

    f = open('HOF/classification2_fitness=' + str(record['max']) + '_NGEN=' + str(NGEN) + '_steps=' + str(STEPS) + '_MU='+str(MU)+'_LA=' +str(LAMBDA) + '.txt', 'w+')
    for i in hof2:
        f.write(str(i) +'\n')
    f.close

    f = open('HOF/classification3_fitness=' + str(record['max']) + '_NGEN=' + str(NGEN) + '_steps=' + str(STEPS) + '_MU='+str(MU)+'_LA=' +str(LAMBDA) + '.txt', 'w+')
    for i in hof3:
        f.write(str(i) +'\n')
    f.close

experiment()