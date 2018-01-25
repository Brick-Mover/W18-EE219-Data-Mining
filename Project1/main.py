from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import svm
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']


class mytokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __call__(self, text):
        tokens = re.sub(r'[^A-Za-z]', " ", text)
        tokens = re.sub("[,.-:/()?{}*$#&]"," ",tokens)
        tokens =[word for tk in nltk.sent_tokenize(tokens) for word in nltk.word_tokenize(tk)]
        new_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]{2,}', token):
                new_tokens.append(token)     
        stems = [self.stemmer.stem(t) for t in new_tokens]
        return stems


class Project1(object):

    def __init__(self, minDf):
        self.eightTrainingData = None
        self.eightTestingData = None
        self.minDf = minDf
        self.XTrainingCount = None
        self.XTrainTfidf = None
        self.countVec = None
        self.XLSITraining = None
        self.yLSITraining = None
        self.XNMFTraining = None
        self.yNMFTraining = None
        self.XLSITesting = None
        self.yLSITesting = None
        self.XNMFTesting = None
        self.yNMFTesting = None

    def load8TestingData(self):
        if not self.eightTestingData:
            self.eightTestingData = fetch_20newsgroups(subset='test', categories=categories,
                                                    remove=('headers','footers','quotes'))

    def load8TrainingData(self):
        if not self.eightTrainingData:
            self.eightTrainingData = fetch_20newsgroups(subset='train', categories=categories,
                                                    remove=('headers','footers','quotes'))
    def createXTrainingCounts(self):
        if not self.countVec:
            self.countVec = CountVectorizer(min_df=self.minDf, analyzer='word',
                                            stop_words=text.ENGLISH_STOP_WORDS, tokenizer=mytokenizer())
        self.load8TrainingData()
        if self.XTrainingCount is None:
            self.XTrainingCount = self.countVec.fit_transform(self.eightTrainingData.data)

    """
    (a) Plot a histogram of the number of training documents per class to check if they are evenly distributed.
    """
    def problemA(self):
        self.plot_size()

    def plot_size(self):
        for category in categories:
            trainingData = fetch_20newsgroups(subset='train', categories=[category], remove=('headers','footers','quotes'))
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
        self.load8TrainingData()
        # tokenization
        self.createXTrainingCounts()

        print('Size of feature vectors when minDf is %s: %s' % (self.minDf, self.XTrainingCount.shape))

        # compute tf-idf
        tfidfTransformer = TfidfTransformer()

        if not self.XTrainTfidf:
            self.XTrainTfidf = tfidfTransformer.fit_transform(self.XTrainingCount)
        print('Size of tf-idf when minDf is %s: %s' % (self.minDf, self.XTrainTfidf.shape))


    """
    (c) Report 10 most significant terms base on tf-icf
    """
    def problemC(self):
        classes = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                   'misc.forsale', 'soc.religion.christian']
        tfIcf, features = self.calc_tf_icf(classes)
        for i in range(0, 4):
            result = tfIcf[i, :]
            idx = np.argpartition(result, -10)[-10:]
            idx = idx[np.argsort(result[idx])]
            print("top 10 feature of %s:" % (classes[i])),
            print([features[i] for i in idx])


    def calc_tf_icf(self, classes):
        if not self.countVec:
            self.countVec = CountVectorizer(min_df=self.minDf, analyzer='word',
                                            stop_words=text.ENGLISH_STOP_WORDS, tokenizer=mytokenizer())

        features = set()
        for c in classes:
            CTrainingData = fetch_20newsgroups(subset='train', categories=[c])
            self.countVec.fit_transform(CTrainingData.data)
            Cword = set(self.countVec.vocabulary_.keys())
            features |= Cword

        features = list(features)
        word2Idx = {}
        for idx, word in enumerate(features):
            word2Idx[word] = idx

        print("number of words we care: ", len(features))

        # tf_icf = Matrix(#class, #words)
        tf = np.zeros(shape=(len(classes), len(features)))
        cf = np.zeros(shape=(1, len(features)))

        # iterate through the four classes to get term frequency
        for cIdx, c in enumerate(classes):
            CData = fetch_20newsgroups(subset='train', categories=[c])
            CwordCountSum = self.countVec.fit_transform(CData.data).sum(axis=0)
            Cword = self.countVec.get_feature_names()
            for idx, word in enumerate(Cword):
                tf[cIdx, word2Idx[word]] += CwordCountSum[0, idx]

        for c in list(fetch_20newsgroups(subset='train').target_names):
            CData = fetch_20newsgroups(subset='train', categories=[c])
            CwordCountSum = self.countVec.fit_transform(CData.data).sum(axis=0)
            Cword = self.countVec.get_feature_names()
            for idx, word in enumerate(Cword):
                if word in word2Idx and CwordCountSum[0, idx] > 0:
                    cf[0, word2Idx[word]] += 1

        cf[cf == 0] = 1
        icf = np.log2(20 / cf) + 1

        return tf * icf, features


    """
    (d) Apply LSI to TF*IDF
    """
    def problemD(self):
        # load data
        self.load8TrainingData()
        # tokenization
        self.createXTrainingCounts()

        print('Size of feature vectors when minDf is %s: %s' % (self.minDf, self.XTrainingCount.shape))

        # compute tf-idf
        tfidfTransformer = TfidfTransformer()

        if not self.XTrainTfidf:
            self.XTrainTfidf = tfidfTransformer.fit_transform(self.XTrainingCount)
        print('Size of tf-idf when minDf is %s: %s' % (self.minDf, self.XTrainTfidf.shape))

        svd = TruncatedSVD(n_components=50)
        self.XLSITraining = svd.fit_transform(self.XTrainTfidf)
        self.yLSITraining = [ x / 4 for x in self.eightTrainingData.target ]

        nmf = NMF(n_components=50)
        self.XNMFTraining = nmf.fit_transform(self.XTrainTfidf)
        self.yNMFTraining = self.yLSITraining

    """
    (e) Use hard margin classifier to separate the
    documents into ‘Computer Technology’ vs ‘Recreational Activity’ groups
    """
    def problemE(self):
        if not self.XLSITraining or not self.yLSITraining:
            self.problemD()     # will have to use everything in part D

        self.load8TestingData()

        lSVC = svm.LinearSVC(C=1000)
        lSVC.fit(self.XLSITraining, self.yLSITraining)

        yScore = lSVC.decision_function(self.XLSITesting)
        self.plot_ROC(yScore)


    def plot_ROC(self, yScore):
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, thresholds = roc_curve(self.yLSITesting, yScore)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

def main():
    # debug mytokenizer (spam)
    # corpus = ['stem stems stemming, go stemmed']
    # # a = Project1(minDf=2)
    # # countVec = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, tokenizer=a.mytokenizer)
    # countVec = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, tokenizer=mytokenizer())
    # out = countVec.fit_transform(corpus)
    # name = countVec.get_feature_names()
    # print(name)

    p = Project1(minDf=2)
    # p.problemA()
    # p.problemB()
    # p.problemC()
    p.problemD()


if __name__ == "__main__":
    main()