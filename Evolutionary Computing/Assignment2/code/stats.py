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
        data_alg_one = pd.read_csv('boxplot_csv/train={en}_enemies=all.csv'.format(en=enemy), index_col=0)
        data_alg_two = pd.read_csv('boxplot_csv/train=cma{en}_enemies=all.csv'.format(en=enemy), index_col=0)
        mean_one.append(round(np.mean(data_alg_one['gain']), 2))
        mean_two.append(round(np.mean(data_alg_two['gain']), 2))
        std_one.append(round(np.std(data_alg_one['gain']), 2))
        std_two.append(round(np.std(data_alg_two['gain']), 2))
        print(stats.shapiro(data_alg_one['gain']))
        print(stats.shapiro(data_alg_two['gain']))
        # test = stats.wilcoxon(data_alg_one['gain'], data_alg_two['gain'])
        test = stats.ttest_ind(data_alg_one['gain'], data_alg_two['gain'])
        p_values.append(round(test[1], 2))
        t_values.append(round(test[0], 2))
    results = pd.DataFrame(data={'mean(+)': mean_one, 'mean(CMA-ES)': mean_two, 'std(+)': std_two, 'std(CMA-ES)': std_one, 't-stat': t_values, 'p-value': p_values}, index=enemies)
    print(results.to_latex())

enemies=['2_5','6_8']
t_test(enemies)
