__author__ = 'gkour'

import random
import numpy as np
import compcor.corpus_metrics as corpus_metrics


class KSC:

    @staticmethod
    def _known_similarity_corpora(sentences_set1, sentences_set2, n=50, k=5, unique_samples_corpora=True):
        """
        given 2 sets of sentences, creates k Known Similarity corpora of size n.
        :param sentences_set1: sentences of domain 1.
        :param sentences_set2: sentences of domain 1.
        :param n: length of the created corpora.
        :param k: the number of created corpora.
        :param unique_samples_corpora: maintain that each sample is used in a single corpus in the KSC
        :return: a set of sets containing the k created corpora.
        """
        if unique_samples_corpora and (len(sentences_set1)<k*n or len(sentences_set2)<k*n):
            raise Exception("To build KSC with n={} and k={} there should be at least {} items in the initial corpora which currenly contains: {},{}."
                            "(to ensure all items in all combination corpora should are different".format(n, k, k * n, len(sentences_set1),  len(sentences_set2)))

        sentences_set1_unused = np.array(sentences_set1)
        sentences_set2_unused = np.array(sentences_set2)

        ksc = []
        for i in np.linspace(0, 1, k, endpoint=True):
            p = int(i * n)
            current_set1_indx = random.sample(range(len(sentences_set1_unused)), p)
            current_set1 = sentences_set1_unused[current_set1_indx]
            if unique_samples_corpora:
                sentences_set1_unused = np.delete(sentences_set1_unused, current_set1_indx, axis=0)
            current_set2_indx = random.sample(range(len(sentences_set2_unused)), n-p)
            current_set2 = sentences_set2_unused[current_set2_indx]
            if unique_samples_corpora:
                sentences_set2_unused = np.delete(sentences_set2_unused, current_set2_indx, axis=0)
            ksc += [list(current_set1) + list(current_set2)]

        return ksc


    @staticmethod
    def _KSC_comparisons(k):
        s, _ = KSC._KSC_comparisons_rec(0, k-1)
        return s

    @staticmethod
    def _KSC_comparisons_rec(start, end):
        s = []
        if end - start == 1:
            return [], []
        son1 = (start, end - 1)
        son2 = (start + 1, end)
        s.append([son1, (start, end)])
        s.append([son2, (start, end)])
        right_judg, right_sons = KSC._KSC_comparisons_rec(son1[0], son1[1])
        left_judg, left_sons = KSC._KSC_comparisons_rec(son2[0], son2[1])
        s.extend(right_judg)
        s.extend(left_judg)
        all_sons = set(right_sons + left_sons)
        desc_judge = [[son, (start, end)] for son in all_sons]
        s.extend(desc_judge)
        return s, list(all_sons) + [son1] + [son2]

    @staticmethod
    def score (sent_set1, sent_set2, dist=corpus_metrics.fid_distance, n=100, k=10):
        """
        :param sent_set1: contain the sentences of the first corpus, or vectors if t-test/classifier/ is used.
        :param sent_set2: contain the sentences of the second corpus, or vectors if t-test/classifier is used.
        :param dist: the dissimilarity metric
        :param n: number of sentence to use from each corpus
        :param k: number of splits
        :return:
        """
        ksc = KSC._known_similarity_corpora(sent_set1, sent_set2, n, k)
        return KSC.test_ksc(ksc, dist=dist)

    @staticmethod
    def test_ksc(ksc, dist=corpus_metrics.fid_distance):
        ksc_doc = []
        for corpus in ksc:
            if dist == corpus_metrics.chi_square_distance:
                ksc_doc.append([x for xs in corpus for x in xs])
            else:
                ksc_doc.append(corpus)

        distance_stats = []
        d_ksc = np.ones([len(ksc), len(ksc)]) * np.nan
        for i in range(len(ksc)):
            for j in range(i + 1, len(ksc)):
                d_ksc[i, j] = dist(ksc_doc[i], ksc_doc[j])
                distance_stats.append((i, j, d_ksc[i, j]))

        all_comparisons = KSC._KSC_comparisons(len(ksc))
        comparisons_results = []
        print("Num judgments: {}".format(len(all_comparisons)))
        for comparison in all_comparisons:
            i = comparison[0][0]
            j = comparison[0][1]
            m = comparison[1][0]
            n = comparison[1][1]
            d_ij = d_ksc[i, j]
            d_mn = d_ksc[m, n]
            comparisons_results.append([(i, j), (m, n), 1 if d_ij < d_mn else 0])

        comparison_accuracy = np.vstack(comparisons_results)[:, 2]
        d0 = [j - i for (i, j) in np.vstack(comparisons_results)[:, 0]]
        d1 = [j - i for (i, j) in np.vstack(comparisons_results)[:, 1]]
        weights = 1 / (np.array(d1) - np.array(d0))

        weighted_comparison_accuracy = np.sum(comparison_accuracy * (weights)) / np.sum(weights)
        print(
            "{}: KSC_Score: {},Weighted KSC:{}".format(str(dist).split()[1].split('_')[0], np.mean(comparison_accuracy),
                                                       weighted_comparison_accuracy))

        return np.mean(comparison_accuracy), weighted_comparison_accuracy, distance_stats

