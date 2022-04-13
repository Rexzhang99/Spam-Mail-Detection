# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math
from collections import Counter

import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir, testdir, stemming, lowercase,
                                                                            silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    print(len(X), 'X')

    def flatten(t):
        return [item for sublist in t for item in sublist]

    pos_x = [i for (i, v) in zip(X, y == 1) if v]
    neg_x = [i for (i, v) in zip(X, y == 0) if v]

    pos_vocab = Counter(flatten(pos_x))
    neg_vocab = Counter(flatten(neg_x))

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    # print(len(X),'X')

    X_bi = [[' '.join(item) for item in zip(email, email[1:])] for email in X]
    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(X, y, max_size=None)
    pos_vocab_bi, neg_vocab_bi = create_word_maps_uni(X_bi, y, max_size=None)

    return {**pos_vocab_uni, **pos_vocab_bi}, {**neg_vocab_uni, **neg_vocab_bi}


# Keep this in the provided template
def print_paramter_vals(laplace, pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""


def get_posterior(dev_set, laplace, pos_prior, pos_vocab, neg_vocab, uni=True):
    len_pos_total_vocab = sum(dict(pos_vocab).values())
    len_neg_total_vocab = sum(dict(neg_vocab).values())
    len_pos_distinct_vocab = len(pos_vocab)
    len_neg_distinct_vocab = len(neg_vocab)

    # Without laplace smoothing
    # p_pos_vocab={k: v /len_pos_vocab  for k, v in pos_vocab.items()}
    # p_neg_vocab = {k: v / len_neg_vocab for k, v in neg_vocab.items()}

    def laplace_smoothing(word_counts, pos, k):
        if pos:
            return (word_counts + k) / (len_pos_total_vocab + k * (1 + len_pos_distinct_vocab))
        else:
            return (word_counts + k) / (len_neg_total_vocab + k * (1 + len_neg_distinct_vocab))

    def loglikelihood(item, pos, vocabulary):
        if item in vocabulary:
            return math.log(laplace_smoothing(vocabulary[item], pos, laplace))
        else:
            return math.log(laplace_smoothing(0, pos, laplace))

    # if uni:
    #     p_pos_likelih = [[loglikelihood(item, 1, pos_vocab) for item in email] for email in dev_set]
    #     p_neg_likelih = [[loglikelihood(item, 0, neg_vocab) for item in email] for email in dev_set]
    # else:
    #     bi_pos_vocab = {k: v for k, v in pos_vocab.items() if " " in k}
    #     bi_neg_vocab = {k: v for k, v in neg_vocab.items() if " " in k}
    #     p_pos_likelih = [[loglikelihood(item, 1, bi_pos_vocab) for item in email] for email in dev_set]
    #     p_neg_likelih = [[loglikelihood(item, 0, bi_neg_vocab) for item in email] for email in dev_set]

    if uni:
        p_pos_loglikelih = [[loglikelihood(item, 1, pos_vocab) for item in email] for email in dev_set]
        p_neg_loglikelih = [[loglikelihood(item, 0, neg_vocab) for item in email] for email in dev_set]
    else:
        p_pos_loglikelih = [[loglikelihood(' '.join(item), 1, pos_vocab) for item in zip(email, email[1:])] for email in dev_set]
        p_neg_loglikelih = [[loglikelihood(' '.join(item), 0, neg_vocab) for item in zip(email, email[1:])] for email in dev_set]


    p_pos_posterior = [math.log(pos_prior) + sum(item) for item in p_pos_loglikelih]
    p_neg_posterior = [math.log(1 - pos_prior) + sum(item) for item in p_neg_loglikelih]

    return p_pos_posterior, p_neg_posterior


def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.9, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace, pos_prior)

    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)

    p_pos_posterior, p_neg_posterior = get_posterior(dev_set, laplace, pos_prior, pos_vocab, neg_vocab)

    res = [1 if pos >= neg else 0 for pos, neg in zip(p_pos_posterior, p_neg_posterior)]
    return res


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,
                pos_prior=0.8, silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    max_vocab_size = None

    pos_vocab_uni, neg_vocab_uni = create_word_maps_uni(train_set, train_labels)
    pos_vocab_bi, neg_vocab_bi = create_word_maps_bi(train_set, train_labels)

    p_pos_posterior_uni, p_neg_posterior_uni = get_posterior(dev_set, unigram_laplace, pos_prior, pos_vocab_uni,
                                                             neg_vocab_uni)
    p_pos_posterior_bi, p_neg_posterior_bi = get_posterior(dev_set, bigram_laplace, pos_prior, pos_vocab_bi,
                                                           neg_vocab_bi, uni=False)

    def combine_uni_bi(log_posterior_uni, log_posterior_bi, bigram_lambda):
        return [(1 - bigram_lambda) * loguni + bigram_lambda * logbi for loguni, logbi in
                zip(log_posterior_uni, log_posterior_bi)]

    p_pos_posterior = combine_uni_bi(p_pos_posterior_uni, p_pos_posterior_bi, bigram_lambda)
    p_neg_posterior = combine_uni_bi(p_neg_posterior_uni, p_neg_posterior_bi, bigram_lambda)

    res = [1 if pos >= neg else 0 for pos, neg in zip(p_pos_posterior, p_neg_posterior)]
    return res
