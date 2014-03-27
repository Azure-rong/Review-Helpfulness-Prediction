#! /usr/bin/env python2.7
#coding=utf-8

"""
Compute review centroid score by combinating every word's tfidf score.
This module use filtered review data in a txt file and gensim tf-idf model to extract this review feature.

"""

import textprocessing as tp
import logging
from gensim import corpora, models, similarities

"""
1. Create a txt file with seg and filtered reviews
input: An excel file with product reviews
    手机很好，很喜欢。
    三防出色，操作系统垃圾！
    Defy用过3年感受。。。
    刚买很兴奋。当时还流行，机还很贵
output: A txt file store filtered reviews, every line is a review
    手机 很 好 很 喜欢 
    三防 出色 操作系统 垃圾 
    Defy 用过 3 年 感受 
    刚买 很 兴奋 当时 还 流行 机 还 很 贵
"""
 
def store_seg_fil_result(filepath, sheetnum, colnum, storepath):
    # Read excel file of review and segmention and filter stopwords
    seg_fil_result = tp.seg_fil_excel(filepath, sheetnum, colnum)
 
    # Store filtered reviews
    fil_file = open(storepath, 'w')
    for sent in seg_fil_result:
        for word in sent:
            fil_file.write(word.encode('utf8')+' ')
        fil_file.write('\n')
    fil_file.close()

"""
input: A txt file store filtered reviews as corpus
        手机 很 好 很 喜欢 
        三防 出色 操作系统 垃圾 
        Defy 用过 3 年 感受 
        刚买 很 兴奋 当时 还 流行 机 还 很 贵
output: A list of tfidf total score of every review (store in a txt file)
"""
def centroid(datapath, storepath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Read review data from txt file
    class MyCorpus(object):
        def __iter__(self):
            for line in open(datapath):
                yield line.split()

    # Change review data to gensim corpus format
    Corp = MyCorpus()
    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]

    # Make the corpus become a tf-idf model
    tfidf = models.TfidfModel(corpus)

    # Compute every word's tf-idf score
    corpus_tfidf = tfidf[corpus]

    # Compute review centroid score by combinating every word's tf-idf score
    centroid = 0
    review_centroid = []
    for doc in corpus_tfidf:
        for token in doc:
            centroid += token[1]
        review_centroid.append(centroid)
        centroid = 0

    # Store review centroid score into a txt file
    centroid_file = open(storepath, 'w')
    for i in review_centroid:
        centroid_file.write(str(i)+'\n')
    centroid_file.close()