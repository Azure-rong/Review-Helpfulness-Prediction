#! /usr/bin/env python2.7
#coding=utf-8

"""
Compute review's entropy and perplexity.
This module aim to bulid review ngram language model then compute review entropy and perplexity as features

"""

import textprocessing as tp
import itertools
import nltk
from nltk.model.ngram import NgramModel


# Function of reading excel file contain review data and filtering stopwords, then return a list of review words
def word_by_word_review(filepath, sheetnum, colnum):
    # Read product review data from excel file and segment every review
    review_data = []
    for cell in tp.get_excel_data(filepath, sheetnum, colnum, 'data')[0:get_excel_data(filepath, sheetnum, colnum, 'rownum')]:
        review_data.append(tp.segmentation(cell, 'list')) # Seg every reivew
    
    # Read txt file contain stopwords
    stopwords = tp.get_txt_data('D:/code/stopword.txt', 'lines')
 
    # Filter stopwords from reviews
    seg_fil_result = []
    for review in review_data:
        fil = [word for word in review if word not in stopwords and word != ' ']
        seg_fil_result.append(fil)
        fil = []

    # Return review set as onedimentional list
    review = list(itertools.chain(*seg_fil_result))
    return review


# Function of computing review entropy and perplexity of a review ngram language model
def entropy_perplexity(model, dataset):
    ent_per = []
    for r in dataset:
        ent = model.entropy(r)
        per = model.perplexity(r)
        ent_per.append((ent, per))
    return ent_per


# Store features
def store_name_brand_attribute_features(review_data, filepath, sheetnum, colnum, storepath):
	# Building an ngram language model of a certain product category review
	lm = NgramModel(1, review_data, estimator=None) # Need initiallized

	# Read full review dataset
	data =  tp.seg_fil_excel(filepath, sheetnum, colnum)

	ep = entropy_perplexity(lm, data)

	p = open(storepath,'w')
	for j in ep:
	    p.write(str(j[0]) + '\t' + str(j[1]) + '\n')
	p.close()