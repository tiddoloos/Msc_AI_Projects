#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:52:00 2021
@author: Tom
"""

import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

def linechart(enemy):
    df1 = pd.read_csv('dummy_demo_mu_plus_lambda_enemy={en}/results_dummy_demo_mu_plus_lambda_enemy={en}.csv'.format(en=enemy), index_col=0)
    df2 = pd.read_csv('cma_{en}/results_cma_es_{en}.csv'.format(en=enemy), index_col=0)
    data = df1.join(df2, lsuffix=' plus', rsuffix=' comma')
    sns.set(style='whitegrid', rc = {'figure.figsize':(6, 7)})
    dash_styles = ["",
                   (4, 1.5),
                   (1, 1),
                   (3, 1, 1.5, 1),
                   (5, 1, 1, 1),
                   (5, 1, 2, 1, 2, 1),
                   (2, 2, 3, 1.5),
                   (1, 2.5, 3, 1.2)]
    ax1 = sns.lineplot(data=data,
                     estimator='mean',
                     ci='sd',
                     color='gray',
                     markers=False,
                     dashes=dash_styles
                     )
    ax1.set(xlabel='Generations', ylabel='Fitness')
    ax1.set_xticks(range(0,21,2))
    ax1.set_yticks(range(-100, 101, 20))
    plt.title('Enemy {en}'.format(en=enemy), fontsize=16)
    plt.legend(loc='lower right', labels =['mean avg ( $\mu + \lambda$)', 'mean std ( $\mu + \lambda$)',
                                           'max avg ( $\mu + \lambda$)', 'max std ( $\mu + \lambda$)',
                                         'mean avg CMA-ES', 'mean std CMA-ES',
                                           'max avg CMA-ES', 'max std CMA-ES'], fontsize=11)
    plt.savefig("figures/Line Charts/linechart_enemy{en}.png".format(en=enemy), dpi=300)
    plt.show()


def boxplot(enemy):
    path = 'boxplot_csv/'
    dfb1 = pd.read_csv(path + 'train={en}_enemies=all.csv'.format(en=enemy), index_col=0)
    dfb2 = pd.read_csv(path + 'train=cma{en}_enemies=all.csv'.format(en=enemy), index_col=0)
    ax1 = sns.boxplot(data=[dfb1['gain'], dfb2['gain']], orient='v',)
    ax1.set_yticks(range(-400, -49, 20))
    ax1.set(ylabel='Gain')
    ax1.set_xticklabels(['( $\mu + \lambda$)','CMA-ES'])
    plt.title('Enemy {en}'.format(en=enemy), fontsize=16)
    #plt.legend(labels=['$\mu + \lambda$)', '( $\mu , \lambda$)'])
    plt.savefig("figures/Boxplots/boxplot_enemy{en}.png".format(en=enemy), dpi=300)
    plt.show()
    plt.close()

sys.path.insert(0, 'clean_evoman')

#
enemies = ['2_5', '6_8']
for enemy in enemies:
    linechart(enemy)
    boxplot(enemy)