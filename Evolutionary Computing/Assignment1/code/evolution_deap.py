################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
"""authors: Tom de Leeuw,
Tiddo Loos, Hidde van Oijen,
Quinten van der kaaij"""

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# import modules
import time
import numpy as np
import pandas as pd
import random

# import DEAP
from deap import base, creator, tools, algorithms

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_names = ['dummy_demo_mu_plus_lambda_enemy=4','dummy_demo_mu_plus_lambda_enemy=5']
different_enemies = [[4],[5]]
for experiment_name, enemy in zip(experiment_names, different_enemies):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    n_hidden_neurons = 10

    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                    enemies=enemy,
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    randomini='yes')

    # default environment fitness is assumed for experiment

    env.state_to_log() # checks environment state
    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

    ini = time.time()  # sets time marker
    # genetic algorithm params

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
    # set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    # Initialize DEAP
    toolbox=base.Toolbox()
    IND_SIZE = n_vars
    toolbox = base.Toolbox()
    toolbox.register("attribute", np.random.uniform, dom_l, dom_u)
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

    def main_mu_plus_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN):
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1, similar = lambda x,y: np.all(x==y))
        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
            cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
        return pop, logbook, hof
        
    def main_mu_comma_lambda(MU, LAMBDA, CXPB, MUTPB, NGEN):
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(1, similar = lambda x, y: np.all(x==y))
        pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
            cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)
        return pop, logbook, hof    


    def experiment(function, runs, MU, LAMBDA, CXPB, MUTPB, NGEN, enemies):
        env.update_parameter('enemies',enemies)
        results = []
        best = []
        means = []
        maxima = []
        minima = []
        stds = []
        for i in range(runs):
            pop, logbook, hof = function(MU, LAMBDA, CXPB, MUTPB, NGEN)
            results.append([pop, logbook, hof])
            best.append(results[i][2][0].fitness.values[0])
            minimum = []
            maximum = []
            mean = []
            std = []
            for g in range(NGEN):
                maximum.append(results[i][1][g]['max'])
                minimum.append(results[i][1][g]['min'])
                mean.append(results[i][1][g]['avg'])
                std.append(results[i][1][g]['std'])
                
            maxima.append(maximum)
            minima.append(minimum)
            means.append(mean)
            stds.append(std)
            
            # saves simulation state
            solutions = [pop, evaluate(pop)]
            env.update_solutions(solutions)
            env.save_state()
            
            np.savetxt(experiment_name+'/best_{run}'.format(run=i+1)+'.txt',hof[0])
            np.savetxt('solutions/best_{run}'.format(run=i+1)+'.txt', hof[0])
            
        df = pd.DataFrame(data = {'means_avg':np.average(means, axis=0), 
                            'means_std':np.std(means, axis=0), 
                            'max_avg':np.average(maxima, axis=0), 
                            'max_std':np.std(maxima, axis=0)},
                        index = range(1, NGEN+1))
        df.to_csv('{exp}/results_{exp}.csv'.format(exp=experiment_name))
        return results, df

    # PARAMTERS
    MU, LAMBDA = 100, 300
    CXPB, MUTPB, NGEN = 0.4, 0.2, 20
    enemies = enemy

    # indicate algorithm
    algorithm = main_mu_plus_lambda
    results, df = experiment(algorithm, 10, MU, LAMBDA, CXPB, MUTPB, NGEN, enemies)
    # print total time
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
