#! /usr/bin/env python2.7
#coding=utf-8

"""  
Compute editorial review and product review similarity feature.

This module use gensim to build review tf-idf model and compute similarity of every review and a given txt.
So this module need a excel file contain all reviews and a txt file contain editorial review as input data.

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
2. Create a txt file with filtered editorial review
input: A query txt file of only one paragraph
    一款畅销的产品是如何创造出来的？一种是迎合需求，解决消费者现实的需求而受到
    欢迎，现在绝大部分的手机属于这一种。而另外一种则是创造需求，让用户有了新的
    需求，这类产品总是可遇而不可求。…………
output: A txt file with filtered document
     一款 畅销 产品 创造 出来 一种 迎合 需求 解决 消费者 现实 需求 受到 欢迎
     现在 绝大部分 手机 属于 一种 一种 创造 需求 用户 新 需求 类产品 总是 可遇
     不可 求 ………
"""

def seg_filter_txt(filepath, storepath):
    txtfile = open(filepath, 'r')
    txtdata = txtfile.readlines()
    txtfile.close()

    review_data = tp.segmentation(txtdata[0], 'list')

    stopfile = open('D:/code/seg_fil_test/stopword.txt', 'r')
    stopdata1 = stopfile.readlines()
    stopdata2 = ''.join(stopdata1)
    stopwords = stopdata2.decode('utf8').split('\n')
    stopfile.close()

    seg_fil_result = []
    for review in review_data:
        fil = [word for word in review if word not in stopwords and word != ' ']
        seg_fil_result.append(fil)
        fil = []

    fil_file = open(storepath, 'w')
    for word in seg_fil_result:
        fil_file.write(word.encode('utf8')+' ')
    fil_file.close()


"""
3. Compute similarity score of editorial review and every review, and store the result into a txt file
input: A txt file store filtered reviews as corpus
        手机 很 好 很 喜欢 
        三防 出色 操作系统 垃圾 
        Defy 用过 3 年 感受 
        刚买 很 兴奋 当时 还 流行 机 还 很 贵
       A txt file with filtered document as query document
        一款 畅销 产品 创造 出来 一种 迎合 需求 解决 消费者 现实 需求 受到 欢迎
         现在 绝大部分 手机 属于 一种 一种 创造 需求 用户 新 需求 类产品 总是 可遇
         不可 求 ………
output: A list of tfidf similarity score of every review (store in a txt file)
"""

def gensim(datapath, querypath, storepath):
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

    # Read filtered editorial review from txt file
    q_file = open(querypath, 'r')
    query = q_file.readline()
    q_file.close()

    # Based on the review tf-idf model, compute its tf-idf score
    vec_bow = dictionary.doc2bow(query.split())
    vec_tfidf = tfidf[vec_bow]

    # Compute similarity
    index = similarities.MatrixSimilarity(corpus_tfidf)
    sims = index[vec_tfidf]

    similarity = list(sims)

    # Store similarity score into a txt file
    sim_file = open(storepath, 'w')
    for i in similarity:
        sim_file.write(str(i)+'\n')
    sim_file.close()