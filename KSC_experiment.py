import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from KSC import KSC
from metric_characteristics import *
from experiment_config import get_metric_dependant_data, ksc_measures, metrics_names, metrics
from compcor import Corpus
import os.path

import numpy as np
import random
import torch
import utils


SMALL_SIZE = 15
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

sns.set_theme(style="whitegrid")
sns.set(font_scale=2)


def runKSC(metrics, metric_names, corpus1:Corpus, corpus2:Corpus, n=30, k=7, repetitions=5, output="ksc"):
	ksc_results = []
	distance_results = []

	for metric_idx, metric in enumerate(metrics):
		c1 = get_metric_dependant_data(metric, corpus1)
		c2 = get_metric_dependant_data(metric, corpus2)

		for rep in range(repetitions):

			distances_metric = []

			ksc = KSC._known_similarity_corpora(c1, c2, n=n, k=k, unique_samples_corpora=True)
			start = time.time()
			accuracy, weighted_accuracy, distance_stats = KSC.test_ksc(ksc, dist=metric)
			ksc_time = (time.time() - start) / len(distance_stats)

			distances_metric.append(
				np.vstack([[metric_names[metric_idx], rep, a, b, b - a, y] for (a, b, y) in distance_stats]))

			distances_metric = np.vstack(distances_metric)

			# normalize the score for a specific metric.
			distances_metric = np.append(distances_metric, sklearn.preprocessing.StandardScaler().fit_transform(
				distances_metric[:, 5].reshape(-1, 1)), axis=1)
			distance_results.extend(distances_metric)

			ells = distances_metric[:, 4].astype('float')
			ds_normalized = distances_metric[:, 6].astype('float')

			monotonicity = metric_monotonicity(ells, ds_normalized)
			separability = metric_separability(ells, ds_normalized)
			linearity = metric_linearity(ells, ds_normalized)
			ksc_results.append(
				[metric_names[metric_idx], accuracy, weighted_accuracy, ksc_time, monotonicity, separability, linearity])

	metrics_measures_df = pd.DataFrame(data=ksc_results, columns=['metric'] + ksc_measures)
	metrics_measures_df['Time'] = (1 / metrics_measures_df['Time'])/100

	all_distance_samples_df = pd.DataFrame(data=distance_results,
									columns=['metric', 'repetition', 'i', 'j', 'l', 'distance', 'distance_score'])
	all_distance_samples_df["l"] = pd.to_numeric(all_distance_samples_df["l"])
	all_distance_samples_df["distance"] = pd.to_numeric(all_distance_samples_df["distance"])
	all_distance_samples_df["distance_score"] = pd.to_numeric(all_distance_samples_df["distance_score"])
	metrics_measures_df.to_csv(path_or_buf=output+'metrics_measures_df.csv', index=False, float_format='%.3f')
	all_distance_samples_df.to_csv(path_or_buf=output+'all_distance_samples_df.csv', index=False, float_format='%.3f')

	return metrics_measures_df, all_distance_samples_df


def plotKSC(all_distance_samples_df):

	sns.set_theme(style="whitegrid")
	sns.set(font_scale=2)

	metrics_names = np.unique(all_distance_samples_df['metric'])
	fig, axlist = plt.subplots(1, len(metrics_names), figsize=(35, 5))
	for i, metric in enumerate(metrics_names):
		metric_df = all_distance_samples_df[all_distance_samples_df['metric'] == metric]
		sns.scatterplot(x='l', y='distance', data=metric_df, ax=axlist[i], color='orange')
		sns.regplot(x='l', y='distance', data=metric_df, ax=axlist[i],
					scatter=False, truncate=False)
		axlist[i].set_title('{}'.format(metric))
	[axi.set(xlabel=None) for axi in axlist]
	[axi.set(ylabel=None) for axi in axlist]

	plt.subplots_adjust(left=0.05,
						bottom=0.1,
						right=0.99,
						top=0.9,
						wspace=0.3,
						hspace=0.4)

	plt.show()


def plot_measures_results(metrics_measures_df):
	fig, ax = plt.subplots(1, 6, figsize=(35, 5))
	[sns.boxplot(ax=ax[i], x='metric', y=measure, data=metrics_measures_df) for i, measure in enumerate(ksc_measures)]

	plt.subplots_adjust(left=0.1,
						bottom=0.1,
						right=0.99,
						top=0.9,
						wspace=0.3,
						hspace=0.4)

	[axi.set(xlabel=None) for axi in ax]
	[axi.set_xticklabels(ax[0].get_xticklabels(), fontsize=12) for axi in ax]
	plt.savefig('ksc_metrics.png')
	plt.show()


if __name__ == '__main__':

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	L = [100, 7]
	H = [100, 12]
	rep = 5

	max_samples = H[0] * H[1] * rep
	output_folder = 'output/ksc_measures'

	clinc, banking, atis, yahoo = utils.load_data(max_samples)

	for R in [L,H]:
		for i, pair in enumerate([(clinc, banking), (atis, yahoo)]):
			results_file_name = os.path.join(output_folder, "pair{}_{}_{}_".format(i, R[0], R[1]))
			metrics_measures_df, all_distance_samples_df = \
				runKSC(metrics, metrics_names, pair[0], pair[1], n=R[0], k=R[1], repetitions=rep, output=results_file_name)
			plotKSC(all_distance_samples_df)
			plot_measures_results(metrics_measures_df)
			mu12, std12 = utils.summarize_results(metrics_measures_df)

			# all_distance_samples_df = pd.read_csv(results_file_name)
			# metrics_measures_df = pd.read_csv(
			# 	'src/experiments/meme/output/ksc_measures/clinic150_banking77_100_12_metrics_measures_df.csv')

