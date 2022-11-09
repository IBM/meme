import random
import matplotlib
import numpy as np
import torch

import experiment_config as config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

import utils
from experiment_config import get_metric_dependant_data

SMALL_SIZE = 25
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)


def compute_IFC(metrics, metrics_names, reference_corpus, generated_corpora_path, rang,
                output_path, num_samples, repetitions=10):

    original, _ = utils.load_labeled_corpus(reference_corpus, max_samples=3000)

    distance_trend = []
    self_distance = []
    for i in rang:
        #sentences = utils.load_corpus(os.path.join(generated_corpora_path,'iteration_{}.txt'.format(i)), max_samples=3000)
        generated, _ = utils.load_labeled_corpus(os.path.join(generated_corpora_path ,'{}.tsv').format(i), max_samples=3000)
        for m,metric in enumerate(metrics):
            print(metric)
            generated_c = np.array(get_metric_dependant_data(metric, generated))
            original_c = np.array(get_metric_dependant_data(metric, original))
            for rep in range(repetitions):
                indices = random.sample(range(len(original_c)), num_samples)
                indices_generated = random.sample(range(len(generated_c)), num_samples)
                original_subset = original_c[indices].tolist()
                generated_subset = generated_c[indices_generated].tolist()
                distance_trend.append([i, metrics_names[m],rep, metric(original_subset,generated_subset)])

            #Estimate reference-reference distance
            indices = random.sample(range(len(original_c)), num_samples)
            original_subset1 = original_c[indices].tolist()
            indices = random.sample(range(len(original_c)), num_samples)
            original_subset2 = original_c[indices].tolist()

            self_distance.append([i, metrics_names[m], metric(original_subset1, original_subset2)])

    df_IFC_distance = pd.DataFrame(data=distance_trend, columns=['iteration', 'metric', 'rep', 'distance'])
    df_reference_self_distance = pd.DataFrame(data=self_distance, columns=['iteration', 'metric', 'distance'])

    df_IFC_distance.to_csv(path_or_buf=os.path.join(output_path, 'ifc_distance.csv'), index=False)
    df_reference_self_distance.to_csv(path_or_buf=os.path.join(output_path, 'df_reference_self_distance.csv'), index=False)

    return df_IFC_distance, df_reference_self_distance


def plot_IFC(df_distance, df_self_distance):
    plt.style.use('seaborn-whitegrid')
    metrics_names = np.unique(df_distance['metric'])

    fig, ax = plt.subplots(1, len(metrics_names), figsize=(35, 5))
    for i, metric_name in enumerate(metrics_names):
        metric_df = df_distance[df_distance['metric'] == metric_name]

        sns.scatterplot(x='iteration', y='distance', data=metric_df, ax=ax[i], color='orange',  s=40)
        sns.regplot(x='iteration', y='distance', data=metric_df, ax=ax[i],
                    scatter=False, truncate=False, color='blue')
        self_metric_df = df_self_distance[df_self_distance['metric'] == metric_name]

        x_min_max = [np.min(self_metric_df['iteration']), np.max(self_metric_df['iteration'])]
        mean_self_distance = np.mean(self_metric_df['distance'])
        sns.lineplot(x=x_min_max, y=mean_self_distance, ax=ax[i], color='green', linewidth=2)

    [axi.set_title(metrics_names[i]) for i, axi in enumerate(ax)]
    [axi.set(xlabel=None) for axi in ax]
    [axi.set(ylabel=None) for axi in ax]

    plt.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.99,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.4)

    plt.show()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(42)
    torch.manual_seed(0)

    # reference_path = './data/news_data/news_data.tsv'
    # generated_path = './data/news_data/generated_news_data/'
    # sampled_samples = 100
    # rang = range(1, 75, 2)

    reference_path = 'data/banking77_data.tsv'
    generated_path = 'data/banking77_generated'
    sampled_samples = 700
    rang = range(1, 35, 1)


    df_distance, df_self_distance = compute_IFC(config.metrics,config.metrics_names,reference_path, generated_path,
                                                rang=rang,
                                                output_path=os.path.join('/output/IFC','banking_77_new'),
                                                num_samples=sampled_samples)

    plot_IFC(df_distance, df_self_distance)
    # df_distance = pd.read_csv(filepath_or_buffer='src/experiments/meme/output/IFC/banking_77_new/ifc_distance.csv')
    # df_self_distance = pd.read_csv(filepath_or_buffer='src/experiments/meme/output/IFC/banking_77_new/df_reference_self_distance.csv')


