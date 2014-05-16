#! /usr/bin/env python2.7
#coding=utf-8

"""
Use positive and negative review set as corpus to train a sentiment classifier.
This module use labeled positive and negative reviews as training set, then use nltk scikit-learn api to do classification task.
Aim to train a classifier automatically identifiy review's positive or negative sentiment, and use the probability as review helpfulness feature.

"""

import textprocessing as tp
import pickle
import itertools
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score


# 1. Load positive and negative review data
pos_review = tp.seg_fil_senti_excel("D:/code/sentiment_test/pos_review.xlsx", 1, 1)
neg_review = tp.seg_fil_senti_excel("D:/code/sentiment_test/neg_review.xlsx", 1, 1)

pos = pos_review
neg = neg_review


"""
# Cut positive review to make it the same number of nagtive review (optional)

shuffle(pos_review)
size = int(len(pos_review)/2 - 18)

pos = pos_review[:size]
neg = neg_review

"""


# 2. Feature extraction function
# 2.1 Use all words as features
def bag_of_words(words):
    return dict([(word, True) for word in words])


# 2.2 Use bigrams as features (use chi square chose top 200 bigrams)
def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams)


# 2.3 Use words and bigrams as features (use chi square chose top 200 bigrams)
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


# 2.4 Use chi_sq to find most informative features of the review
# 2.4.1 First we should compute words or bigrams information score
def create_word_scores():
    posdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/pos_review.xlsx", 1, 1)
    negdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/neg_review.xlsx", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in negWords:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def create_bigram_scores():
    posdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/pos_review.xlsx", 1, 1)
    negdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/neg_review.xlsx", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)

    pos = posBigrams
    neg = negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# Combine words and bigrams and compute words and bigrams information scores
def create_word_bigram_scores():
    posdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/pos_review.xlsx", 1, 1)
    negdata = tp.seg_fil_senti_excel("D:/code/sentiment_test/neg_review.xlsx", 1, 1)
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd.inc(word)
        cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd.inc(word)
        cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# Choose word_scores extaction methods
# word_scores = create_word_scores()
# word_scores = create_bigram_scores()
# word_scores = create_word_bigram_scores()


# 2.4.2 Second we should extact the most informative words or bigrams based on the information score
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# 2.4.3 Third we could use the most informative words and bigrams as machine learning features
# Use chi_sq to find most informative words of the review
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

# Use chi_sq to find most informative bigrams of the review
def best_word_features_bi(words):
    return dict([(word, True) for word in nltk.bigrams(words) if word in best_words])

# Use chi_sq to find most informative words and bigrams of the review
def best_word_features_com(words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3



# 3. Transform review to features by setting labels to words in review
def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg']
        negFeatures.append(negWords)
    return negFeatures


best_words = find_best_words(word_scores, 1500) # Set dimension and initiallize most informative words

# posFeatures = pos_features(bigrams)
# negFeatures = neg_features(bigrams)

# posFeatures = pos_features(bigram_words)
# negFeatures = neg_features(bigram_words)

posFeatures = pos_features(best_word_features)
negFeatures = neg_features(best_word_features)

# posFeatures = pos_features(best_word_features_com)
# negFeatures = neg_features(best_word_features_com)



# 4. Train classifier and examing classify accuracy
# Make the feature set ramdon
shuffle(posFeatures)
shuffle(negFeatures)

# 75% of features used as training set (in fact, it have a better way by using cross validation function)
size_pos = int(len(pos_review) * 0.75)
size_neg = int(len(neg_review) * 0.75)

train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
test_set = posFeatures[size_pos:] + negFeatures[size_neg:]

test, tag_test = zip(*test_set)

def clf_score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)

    predict = classifier.batch_classify(test)
    return accuracy_score(tag_test, predict)

print 'BernoulliNB`s accuracy is %f' %clf_score(BernoulliNB())
print 'GaussianNB`s accuracy is %f' %clf_score(GaussianNB())
print 'MultinomiaNB`s accuracy is %f' %clf_score(MultinomialNB())
print 'LogisticRegression`s accuracy is %f' %clf_score(LogisticRegression())
print 'SVC`s accuracy is %f' %clf_score(SVC(gamma=0.001, C=100., kernel='linear'))
print 'LinearSVC`s accuracy is %f' %clf_score(LinearSVC())
print 'NuSVC`s accuracy is %f' %clf_score(NuSVC())



# 5. After finding the best classifier, then check different dimension classification accuracy
def score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(trainset)

    pred = classifier.batch_classify(test)
    return accuracy_score(tag_test, pred)

dimention = ['500','1000','1500','2000','2500','3000']

for d in dimention:
    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, int(d))

    posFeatures = pos_features(best_word_features_com)
    negFeatures = neg_features(best_word_features_com)

    # Make the feature set ramdon
    shuffle(posFeatures)
    shuffle(negFeatures)

    # 75% of features used as training set (in fact, it have a better way by using cross validation function)
    size_pos = int(len(pos_review) * 0.75)
    size_neg = int(len(neg_review) * 0.75)

    trainset = posFeatures[:size_pos] + negFeatures[:size_neg]
    testset = posFeatures[size_pos:] + negFeatures[size_neg:]

    test, tag_test = zip(*testset)

    print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB())
    print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB())
    print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression())
    print 'SVC`s accuracy is %f' %score(SVC())
    print 'LinearSVC`s accuracy is %f' %score(LinearSVC())
    print 'NuSVC`s accuracy is %f' %score(NuSVC())
    print 



# 6. Store the best classifier under best dimension
def store_classifier(clf, trainset, filepath):
    classifier = SklearnClassifier(clf)
    classifier.train(trainset)
    # use pickle to store classifier
    pickle.dump(classifier, open(filepath,'w'))
