import numpy as np
import pandas as pd


def load_corpus(corpus_path, max_samples=np.inf):
	file = open(corpus_path, 'r', encoding='utf-8')
	sentences = [sentence for sentence in [line.strip() for line in file.readlines()] if sentence]
	return sentences


def load_labeled_corpus(filename, sep='\t', max_samples=np.inf):
	data = pd.read_csv(filename, sep=sep, names=["label", "sample"], skipinitialspace=True, index_col=False)
	data.drop(np.where(pd.isnull(data))[0], axis=0, inplace=True)
	data = data.apply(lambda x: x.str.strip())
	data = data.sample(frac=1, random_state=1).reset_index()
	if not np.isinf(max_samples):
		data = data.head(max_samples)
	data.sort_values("label", ascending=True, inplace=True)
	sentences = data['sample'].to_numpy()
	labels = data['label'].to_numpy()
	return sentences, labels

def load_data(max_samples=np.inf):
	clinc, _ = load_labeled_corpus('data/clinc150_uci_data.tsv', max_samples=max_samples)
	banking, _ = load_labeled_corpus('data/banking77_data.tsv', max_samples=max_samples)
	atis, _ = load_labeled_corpus('data/atis_intents.csv', sep=',',max_samples=max_samples)
	yahoo, _ = load_labeled_corpus('data/yahoo_data.tsv', max_samples=max_samples)

	return clinc, banking, atis, yahoo


def summarize_results(metrics_measures_df):
	mu = metrics_measures_df.groupby(['metric']).mean()
	mu = mu.round(decimals=3)
	std = metrics_measures_df.groupby(['metric']).std()
	return mu, std