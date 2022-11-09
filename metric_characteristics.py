import numpy as np
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd


def metric_monotonicity(ells, distances):
	return scipy.stats.spearmanr(ells, distances).correlation


def metric_separability(ells, distances):
	df = pd.DataFrame(data=list(zip(ells, distances)), columns=['ell', 'distance'])
	model = ols('distance ~ C(ell)', data=df).fit()
	aov_table = sm.stats.anova_lm(model, typ=2)
	return anova_table(aov_table)['omega_sq'][0]


def anova_table(aov):
	aov['mean_sq'] = aov[:]['sum_sq'] / aov[:]['df']
	aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
	aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / (
				sum(aov['sum_sq']) + aov['mean_sq'][-1])
	cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
	aov = aov[cols]
	return aov


def metric_linearity(ells, distances):
	return scipy.stats.linregress(ells, y=distances).rvalue


def metric_size_robustness(sizes, distances, true_distance):
	return 1 - np.nansum(np.abs((distances - true_distance))) / (true_distance * 10 * len(np.unique(sizes)))


def metric_imbalance_robustness(sizes, comp_sizes, distances, true_distance):
	return 1 - sum(np.abs((distances - true_distance))) / (10 * len(np.unique(sizes)))

