################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com
################################
"""authors: Tom de Leeuw,
Tiddo Loos, Hidde van Oijen,
Quinten van der kaaij"""

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# import modules
import time
import numpy as np
import os
import random
import pandas as pd
# import DEAP
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'dummy_demo_Tom'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
n_hidden_neurons = 10
env = Environment(experiment_name=experiment_name,
                  enemies=[2,5],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini="yes",
                  multiplemode = "yes")

env.state_to_log() # checks environment state

ini = time.time()  # sets time marker

run_mode = 'train' # train or test

n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 10
gens = 2
mutation = 0.2
last_best = 0


def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f


def fitness_f(x):
    x = np.array(x)
    return [simulation(env, x)]


def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


# Tool decorator
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


#EA's
def main_mu_plus_lambda(MU, CXPB, MUTPB, NGEN):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar = lambda x,y: np.all(x==y))
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=3*MU, 
        cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
    return pop, logbook, hof


def main_mu_comma_lambda(MU, CXPB, MUTPB, NGEN):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1, similar = lambda x, y: np.all(x==y))
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=3*MU, 
        cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
    return pop, logbook, hof    


def param_experiment(function, MU, TOUR_SIZE, NGEN, CXPB, MUTPB):
    results = []
    for i in range(20,MU, 20):
        for j in range(10,i,20):
            print('MU',i)
            print('TOUR_SIZE', j)
            toolbox.register("select", tools.selTournament, tournsize=j)
            print('MU: ', i, 'TOURSIZE: ', j)
            pop, logbook, hof = function(i, CXPB, MUTPB, NGEN)
            results.append([logbook, hof, i, j, NGEN])
            save_results_df(results)
            plot_save_results(results)
    return results


def save_results_df(results):
    TOURSIZE_values = []
    MU_values = []
    avg_results = []
    for i in range(len(results)):
        TOURSIZE_values.append(results[i][3])
        MU_values.append(results[i][2])
        avg_results.append(results[i][0][NGEN - 1]['avg'])
    data = {'MU': MU_values, 'TOURSIZE': TOURSIZE_values, 'avg_fitness': avg_results}
    df = pd.DataFrame(data=data)
    df.to_csv('param_results/param_plus_toursize_20.csv')


def plot_save_results(results):
    TOURSIZE_values = []
    MU_values = []
    avg_results = []
    for i in range(len(results)):
        TOURSIZE_values.append(results[i][3])
        MU_values.append(results[i][2])
        avg_results.append(results[i][0][NGEN-1]['avg'])

    max_value = max(avg_results)
    max_index = avg_results.index(max_value)
    max_values = ['MU=', MU_values[max_index], 'TOUR_SIZE=', TOURSIZE_values[max_index], 'Max Fitness=', max(avg_results)]
    ax = plt.axes(projection='3d')
    zdata = avg_results
    xdata = TOURSIZE_values
    ydata = MU_values
    ax.set_xlabel('TOURSIZE')
    ax.set_ylabel('MU')
    ax.set_zlabel('Avg Fitness')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    if os.path.exists("param_results/ass2_param_tuning_gridplus_TOUR_SIZE_20.png"):
        print('removed')
        os.remove("param_results/ass2_param_tuning_gridplus_TOUR_SIZE_20.png")
    # os.remove('')
    plt.savefig('param_results/ass2_param_tuning_gridplus_TOUR_SIZE_20')
    # plt.show()

    print("...Saving result data to txt file...")
    #change name of algorithm that you use to save the file
    # f = open("param_results/params_McommaL_step=0.1.txt", "x")
    f = open("param_results/params_MplusL_Toursize_20.txt","w+")
    f.write('TOURSIZE=' + str(TOURSIZE_values) + '\n')
    f.write('MU='+ str(MU_values) + '\n')
    f.write('Best Values=' + str(max_values) + '\n')
    f.write('avg_results =' + str(avg_results))
    f.close()

# set up DEAP framework
creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
# Initialize DEAP
toolbox = base.Toolbox()
IND_SIZE = n_vars
toolbox = base.Toolbox()
toolbox.register("attribute", np.random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("evaluate", fitness_f)
toolbox.register("generate", tools.initRepeat, list, toolbox.individual)
toolbox.decorate("mate", checkBounds(dom_l, dom_u))
toolbox.decorate("mutate", checkBounds(dom_l, dom_u))
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# run experiment
NGEN = 10
CXPB = 0.3
MUTB = 0.1
MU = 161
TOUR_SIZE = 150
print('MU',MU)
print('TOUR_SIZE', TOUR_SIZE)

results = param_experiment(main_mu_plus_lambda, MU, TOUR_SIZE, NGEN, CXPB, MUTB)

