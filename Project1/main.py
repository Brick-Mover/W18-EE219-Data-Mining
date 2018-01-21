from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
import numpy as np


categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
               'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey' ]

class Project1(object):

    def __init__(self, minDf):
        self.eightTrainingData = None
        self.minDf = minDf
        self.XTrainCounts = None

    """
    (a) Plot a histogram of the number of training documents per class to check if they are evenly distributed.
    """
    def problemA(self):
        self.plot_size()

    def plot_size(self):
        for category in categories:
            trainingData = fetch_20newsgroups(subset='train', categories=[category])
            print(category, len(trainingData.filenames))


    """
    (b) Modeling Text Data: tokenize each document into words. Then, excluding the stop words,
    punctuations, and using stemmed version of words, create a TFxIDF vector representations.
    min_df = 2 or 5
    """
    def problemB(self):
        self.model_text_data()

    def model_text_data(self):
        # load data
        if not self.eightTrainingData:
            self.eightTrainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

        # tokenization
        countVec = CountVectorizer(min_df=self.minDf, stop_words=text.ENGLISH_STOP_WORDS)
        self.XTrainCounts = countVec.fit_transform(self.eightTrainingData.data)
        print('Size of feature vectors when minDf is %s: %s' % (self.minDf, self.XTrainCounts.shape))

        # compute tf-idf
        tfidfTransformer = TfidfTransformer()
        XTrainTfidf = tfidfTransformer.fit_transform(self.XTrainCounts)
        print('Size of tf-idf when minDf is %s: %s' % (self.minDf, XTrainTfidf.shape))


    """
    (c) Report 10 most significant terms base on tf-icf
    """
    def problemC(self):
        classes = [ 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                    'misc.forsale', 'soc.religion.christian' ]
        tfIcf = self.calc_tf_icf(classes)




    def calc_tf_icf(self, classes):
        CTrainingData = fetch_20newsgroups(subset='train', categories=classes)

        if not self.XTrainCounts or not self.countVec:
            self.countVec = CountVectorizer(min_df=self.minDf, stop_words=text.ENGLISH_STOP_WORDS)
            self.XTrainCounts = self.countVec.fit_transform(CTrainingData.data)

        idx2Word = self.countVec.get_feature_names()
        word2Idx = {}
        for idx, word in enumerate(idx2Word):
            word2Idx[word] = idx

        print(len(self.countVec.get_feature_names()))

        # tf_icf = Matrix(#class, #words)
        tf = np.zeros(shape=(len(classes), self.XTrainCounts.shape[1]))
        cf = np.zeros(shape=(1, self.XTrainCounts.shape[1]))

        # iterate through the four classes to get term frequency
        for cIdx, c in enumerate(classes):
            CData = fetch_20newsgroups(subset='train', categories=[c])
            CcountVec = CountVectorizer(min_df=self.minDf, stop_words=text.ENGLISH_STOP_WORDS)
            CwordCountSum = CcountVec.fit_transform(CData.data).sum(axis=0)
            Cidx2Word = CcountVec.get_feature_names()
            print(len(Cidx2Word))
            for idx, word in enumerate(Cidx2Word):
                tf[cIdx, word2Idx[word]] += CwordCountSum[0, idx]   # first get cf value

        for c in list(fetch_20newsgroups(subset='train').target_names):
            CData = fetch_20newsgroups(subset='train', categories=[c])
            CcountVec = CountVectorizer(min_df=self.minDf, stop_words=text.ENGLISH_STOP_WORDS)
            CwordCountSum = CcountVec.fit_transform(CData.data).sum(axis=0)
            Cidx2Word = CcountVec.get_feature_names()
            for idx, word in enumerate(Cidx2Word):
                if word in word2Idx and CwordCountSum[0, idx] > 0:
                    cf[0, word2Idx[word]] += 1

        


def main():
    p = Project1(minDf=2)
    p.problemC()

if __name__ == "__main__":
    main()