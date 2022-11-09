
import os.path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import sklearn
import torch

import utils
from metric_characteristics import metric_size_robustness, metric_imbalance_robustness
from experiment_config import get_metric_dependant_data, metrics, metrics_names

SMALL_SIZE = 15
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

sns.set_theme(style="whitegrid")
sns.set(font_scale=2)


def size_imbalance_sensitivity_experiment(metrics, metrics_names, corpus1, corpus2, sizes, repetitions, output_folder):
	distance_results = []
	source_corpora_distance = []
	for metric_idx, metric in enumerate(metrics):
		metric_distances = []
		for rep in range(repetitions):
			c1 = get_metric_dependant_data(metric, corpus1)
			c2 = get_metric_dependant_data(metric, corpus2)
			c1 = np.array(c1)
			c2 = np.array(c2)

			for (s, sc) in zip(sizes, reversed(sizes)):
				indices = random.sample(range(len(c1)), s)
				set1 = list(c1[indices])

				indices = random.sample(range(len(c2)), sc)
				set2 = list(c2[indices])

				indices = random.sample(range(len(c2)), s)
				set2_same_size = list(c2[indices])

				dist_complemeting = metric(set1, set2)
				dist_same_size = metric(set1, set2_same_size)

				metric_distances += [[metrics_names[metric_idx], rep, s, sc, dist_same_size, dist_complemeting]]

		metric_distances = np.stack(metric_distances)

		normalizer = sklearn.preprocessing.StandardScaler().fit(metric_distances[:, 4].reshape(-1, 1))

		metric_distances = np.append(metric_distances, normalizer.transform(metric_distances[:, 4].reshape(-1, 1)),
									 axis=1)
		metric_distances = np.append(metric_distances, sklearn.preprocessing.StandardScaler().fit_transform(
			metric_distances[:, 5].reshape(-1, 1)), axis=1)

		metric_distances[:, range(1, 7)] = metric_distances[:, range(1, 7)].astype(float)
		distance_results += metric_distances.tolist()
		sources_distance = metric(c1, c2)
		source_corpora_distance += [[metrics_names[metric_idx], sources_distance,
									 normalizer.transform((sources_distance).reshape(-1, 1))[0][0]]]

	size_imbalance_df = pd.DataFrame(data=distance_results,
									 columns=['metric', 'repetition', 'size', 'size_complementing', 'distance(same)',
											  'distance(comp)', 'distance(same)_norm', 'distance(comp)_norm'])
	source_corpora_distance_df = pd.DataFrame(data=source_corpora_distance,
											  columns=['metric', 'distance', 'distance_norm'])

	size_imbalance_df.to_csv(
		path_or_buf=os.path.join(output_folder, corpus1.name + corpus2.name + 'size_imbalance_df.csv'))
	source_corpora_distance_df.to_csv(
		path_or_buf=os.path.join(output_folder, corpus1.name + corpus2.name + 'source_corpora_distance_df.csv'))

	return size_imbalance_df, source_corpora_distance_df


def size_imbalance_robustness_measure(size_imbalance_df, source_corpora_distance_df):
	source_corpora_distance_df['size_robustness'] = np.empty(len(source_corpora_distance_df))
	source_corpora_distance_df['imbalance_robustness'] = np.empty(len(source_corpora_distance_df))
	for metric_name in np.unique(size_imbalance_df['metric']):
		metric_sizes_distance_samples = size_imbalance_df[size_imbalance_df['metric'] == metric_name]
		metric_true_sources_distance = \
		source_corpora_distance_df[source_corpora_distance_df['metric'] == metric_name]['distance'].iloc[0]

		metric_size_sens = metric_size_robustness(list(metric_sizes_distance_samples['size']),
												  list(metric_sizes_distance_samples['distance(same)']),
												  metric_true_sources_distance)

		metric_imbalance_sens = metric_imbalance_robustness(list(metric_sizes_distance_samples['size']),
															list(metric_sizes_distance_samples['size_complementing']),
															metric_sizes_distance_samples['distance(comp)'],
															metric_true_sources_distance)

		source_corpora_distance_df.loc[
			source_corpora_distance_df['metric'] == metric_name, 'size_robustness'] = metric_size_sens
		source_corpora_distance_df.loc[
			source_corpora_distance_df['metric'] == metric_name, 'imbalance_robustness'] = metric_imbalance_sens

	return size_imbalance_df, source_corpora_distance_df


###########################################
###########################################

# columns = 'distance(comp)_norm' or 'distance(same)_norm'
def plot_size_imbalance_scatter(df_distances, df_distance_corpora_all, column='distance(same)_norm'):
	# Save a palette to a variable:
	palette = sns.color_palette("Paired")
	metrics_names = np.unique(df_distances['metric'])
	x_min_max = [np.min(df_distances['size']), np.max(df_distances['size'])]
	fig, ax = plt.subplots(1, len(metrics_names), figsize=(len(metrics_names) * 5, 5))
	for i, metric in enumerate(metrics_names):
		sns.scatterplot(x='size', y=column, data=df_distances[df_distances['metric'] == metric],
						ax=ax[i], color=palette[1], s=50)
		df_metric = df_distance_corpora_all[df_distance_corpora_all['metric'] == metric]
		sns.lineplot(x_min_max, [np.mean(df_metric['distance']), np.mean(df_metric['distance'])], ax=ax[i],
					 linewidth=3, color=palette[2])
		ax[i].set_title('{}'.format(metric))

	[axi.set(xlabel=None) for axi in ax]
	[axi.set(ylabel=None) for axi in ax]

	plt.subplots_adjust(left=0.05,
						bottom=0.1,
						right=0.99,
						top=0.9,
						wspace=0.3,
						hspace=0.4)

	plt.savefig('size_sens.png')
	plt.show()


if __name__ == '__main__':

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	N = 2900
	repetitions = 10
	start = 50
	step = 200

	output_folder = 'output/size_robustness'

	clinc, banking, atis, yahoo = utils.load_data()
	for i, pair in enumerate([(clinc, banking), (atis, yahoo)]):

		size_imbalance_df, source_corpora_distance_df = size_imbalance_sensitivity_experiment(metrics, metrics_names, pair[0], pair[1],
																							  list(range(start, N+ 1, step)), repetitions, output_folder)

		# size_imbalance_df = pd.read_csv(filepath_or_buffer='src/experiments/meme/output/size_robustness/atis_yahoo_size_imbalance_df.csv')
		# source_corpora_distance_df = pd.read_csv(
		# 	filepath_or_buffer='src/experiments/meme/output/size_robustness/atis_yahoo_source_corpora_distance_df.csv')

		source_corpora_distance_df = source_corpora_distance_df.sort_values(by="metric", ascending=1)

		size_imbalance_df, source_corpora_distance_df = size_imbalance_robustness_measure(size_imbalance_df, source_corpora_distance_df)
		plot_size_imbalance_scatter(size_imbalance_df, source_corpora_distance_df, column='distance(same)')
		plot_size_imbalance_scatter(size_imbalance_df, source_corpora_distance_df,column='distance(comp)')
