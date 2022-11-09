import compcor as compcor
from compcor import Corpus
from compcor import TextTokenizerEmbedder
import pandas as pd
import numpy as np

metrics = [
	compcor.chi_square_distance,
	compcor.zipf_distance,
	compcor.classifier_distance,
	compcor.IRPR_distance,
	compcor.fid_distance,
	compcor.pr_distance,
	compcor.dc_distance,
	compcor.mauve_distance,
]

metrics_names = [((str(dist).split()[1]).split('_')[0]).upper() for dist in metrics]

ksc_measures = ['Accuracy', 'Weighted Accuracy', 'Time', 'Monotonicity', 'Separability', 'Linearity']


def get_metric_dependant_data(metric, corpus: Corpus):
	if metric in (compcor.zipf_distance, compcor.chi_square_distance):
		c = TextTokenizerEmbedder().tokenize_sentences(corpus)
	else:
		c = TextTokenizerEmbedder().embed_sentences(corpus)
	return c
