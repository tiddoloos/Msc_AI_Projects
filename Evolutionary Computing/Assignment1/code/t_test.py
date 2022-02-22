from scipy import stats
import pandas as pd
import numpy as np


def t_test(enemies):
    mean_one = []
    std_one = []
    mean_two = []
    std_two = []
    p_values = []
    t_values = []
    for enemy in enemies:
        data_alg_one = pd.read_csv('boxplot_csv/testrun_test_run_comma_enemy{en}.csv'.format(en=enemy), index_col=0)
        data_alg_two = pd.read_csv('boxplot_csv/testrun_test_run_plus_enemy{en}.csv'.format(en=enemy), index_col=0)
        mean_one.append(round(np.average(data_alg_one['gain']), 2))
        mean_two.append(round(np.average(data_alg_two['gain']), 2))
        std_one.append(round(np.std(data_alg_one['gain']), 2))
        std_two.append(round(np.std(data_alg_two['gain']), 2))
        test = stats.wilcoxon(data_alg_one['gain'], data_alg_two['gain'])
        p_values.append(round(test[1], 2))
        t_values.append(round(test[0], 2))
    results = pd.DataFrame(data={'mean(+)': mean_two, 'mean(,)': mean_one, 'std(+)': std_two, 'std(,)': std_one, 't-stat': t_values, 'p-value': p_values}, index=enemies)
    print(results.to_latex())

enemies=[2,5,8]
t_test(enemies)
