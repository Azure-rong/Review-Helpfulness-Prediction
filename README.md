Review Helpfulness Prediction
==============================================

# Intro: Project of automatically finding helpful reviews.
Using text mining technique. Include natural language processing, sentiment analysis and machine learning technique. <br />
Language: Python. (standard library: xlrd, jieba, gensim, nltk, scikit-learn)<br />

This project include four main parts: <br />
1. Review data set<br />
2. Text preprocessing module. <br />
3. Review helpfulness feature extraction module.<br />
4. Review helpfulness prediction module.<br />

# 1. Review data set
This dataset include review data of 7 cellphone brands grabbed from (http://mobile.zol.com.cn).<br />

# 2. Text preprocessing module
Functions include: Read txt file and excel file, doing Chinese word segmentation and postag.<br />

Using Python xlrd library to read excel data.<br />
Using Python jieba library to do Chinese word segmentation and postag<br />

*Files include:* <br />
User dictionary to increase segmentation accuracy.<br />
Stopwords and sentiment stopwords to filter irrelevant words.<br />

# 3. Feature extraction module
There is four categories of review helpfulness feature.<br />
* Linguistic features<br />
  6 features including review words and sentences number, review average length, review adjectives, adverbs and verbs number.<br />
* Informative features<br />
  4 features including product name, brand and attributes' appearing times in a review. And review centroid score.<br />
* Difference features<br />
  2 features including review entropy and perplexity socre.<br />
* Sentiment feautres<br />
  8 features including review positive/negative score, average score and standard deviation score. And review positive/negative probability score.<br />

Using Python nltk library to compute review entropy/perplexity and doing sentiment analysis.<br />
Using Python gensim library to calculate review words tf-idf weight.<br />

*Files include:* <br />
Labeled positive and negative review corpus.<br />
Trained sentiment classifier, aim to automatically classify review positive and negative.<br />
Sentiment dictionary including positive and negative words and adverbs of degree.<br />
Raw sentiment dictionary including Hownet and NTUSD.<br />

# 4. Review helpfulness prediction module
This module use features calculated from above module as training set. Using machine learning method to train review helpfulness classifier. We test five popular machine learning algorithm and use cross validation method to evaluate helpfulness prediction accuracy.<br />

Using Python scikit-learn library to train classifier and evaluated classifier performance.<br />

*Files include:*<br />
Feature vector matrix with different threshold.<br />
