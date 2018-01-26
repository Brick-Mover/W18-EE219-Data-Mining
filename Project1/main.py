from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import svm
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import KFold
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import nltk
import itertools
import re





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
        self.XTestingCount = None
        self.XTestTfidf = None
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
        if self.eightTestingData is None:
            self.eightTestingData = fetch_20newsgroups(subset='test', categories=categories,
                                                    remove=('headers','footers','quotes'))

    def load8TrainingData(self):
        if self.eightTrainingData is None:
            self.eightTrainingData = fetch_20newsgroups(subset='train', categories=categories,
                                                    remove=('headers','footers','quotes'))
    def createXTrainingCounts(self):
        if self.countVec is None:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            self.countVec = CountVectorizer(min_df=self.minDf, analyzer='word',
                                            stop_words=text.ENGLISH_STOP_WORDS, tokenizer=mytokenizer())
        self.load8TrainingData()
        if self.XTrainingCount is None:
            self.XTrainingCount = self.countVec.fit_transform(self.eightTrainingData.data)

    def createXTestingCounts(self):
        if self.XTrainingCount is None:
            self.createXTrainingCounts()

        print('XTrainingCount size is %s' % (self.XTrainingCount.shape, ))

        self.XTestingCount = self.countVec.transform(self.eightTestingData.data)
        print('XTestingCount size is %s' % (self.XTestingCount.shape, ))


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

        if self.XTrainTfidf is None:
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
        if self.countVec is None:
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
    (d) Apply LSI and NMF to TF*IDF
    """
    def problemD(self):
        # load data
        self.load8TrainingData()
        # tokenization
        self.createXTrainingCounts()

        print('Size of feature vectors when minDf is %s: %s' % (self.minDf, self.XTrainingCount.shape))

        # compute tf-idf
        tfidfTransformer = TfidfTransformer()

        if self.XTrainTfidf is None:
            self.XTrainTfidf = tfidfTransformer.fit_transform(self.XTrainingCount)
        print('Size of train tf-idf when minDf is %s: %s' % (self.minDf, self.XTrainTfidf.shape))

        svd = TruncatedSVD(n_components=50)
        self.XLSITraining = svd.fit_transform(self.XTrainTfidf)
        self.yLSITraining = [ int(x / 4) for x in self.eightTrainingData.target ]

        nmf = NMF(n_components=50)
        self.XNMFTraining = nmf.fit_transform(self.XTrainTfidf)
        self.yNMFTraining = self.yLSITraining

        #
        # apply LSI and NMF to testing data
        #

        # load test data
        self.load8TestingData()
        # tokenization
        self.createXTestingCounts()

        if self.XTestTfidf is None:
            self.XTestTfidf = tfidfTransformer.transform(self.XTestingCount)
        print('Size of test tf-idf when minDf is %s: %s' % (self.minDf, self.XTestTfidf.shape))

        self.XLSITesting = svd.transform(self.XTestTfidf)
        self.yLSITesting = [ int(x / 4) for x in self.eightTestingData.target ]

        self.XNMFTesting = nmf.transform(self.XTestTfidf)
        self.yNMFTesting = self.yLSITesting


    """
    (e) Use hard margin classifier to separate the
    documents into ‘Computer Technology’ vs ‘Recreational Activity’ groups
    """
    def problemE(self, method, penalty):
        if self.XLSITraining is None or self.yLSITraining is None or \
            self.XNMFTraining is None or self.yNMFTraining is None:
            self.problemD()     # will have to use everything in part D

        self.load8TestingData()

        assert penalty == "hard" or penalty == "soft"
        if penalty == "hard":
            lSVC = svm.LinearSVC(C=1000)
        elif penalty == 'soft':
            lSVC = svm.LinearSVC(C=0.001)
        else:
            lSVC = svm.LinearSVC(C=penalty)

        assert method == "LSI" or method == "NMF"
        if method == "LSI":
            XTrain, yTrain, XTest, yTest = \
                self.XLSITraining, self.yLSITraining, self.XLSITesting, self.yLSITesting
        else:
            XTrain, yTrain, XTest, yTest = \
                self.XNMFTraining, self.yNMFTraining, self.XNMFTesting, self.yNMFTesting

        lSVC.fit(XTrain, yTrain)
        yScore = lSVC.decision_function(XTest)

        #plot ROC
        self.plot_ROC(yScore, 'LSI')
        plt.savefig('fig/roc_%s_%s_df%d.png' % (method, penalty, self.minDf), bbox_inches='tight')
        plt.show()

        #plot confusion matrix
        class_names = ['Com Tech', 'Recreation']
        svm_pred = lSVC.predict(XTest)
        conf_mat = confusion_matrix(yTest, svm_pred)

        self.plot_confusion_matrix(conf_mat, classname=class_names, title='Confusion matrix')
        plt.savefig('fig/conf_mat_%s_%s_df%d.png' %
                    (method, penalty, self.minDf), bbox_inches='tight')

        self.plot_confusion_matrix(conf_mat, classname=class_names, normalize=True, 
                                        title='Normalized confusion matrix')
        plt.savefig('fig/conf_mat_norm_%s_%s_df%d.png' %
                    (method, penalty, self.minDf), bbox_inches='tight')
        plt.show()

        # accuracy
        svm_accuracy = accuracy_score(yTest, svm_pred)
        print('SVM accuracy for %s and %s Margin with df %d is: %s' %
              (method, penalty, self.minDf, str(svm_accuracy)))

        # recall
        svm_recall = recall_score(yTest, svm_pred)
        print('SVM recall for %s and %s Margin with df %d is: %s' %
              (method, penalty, self.minDf, str(svm_recall)))

        # precision
        svm_precision = precision_score(yTest, svm_pred)
        print('SVM precision for %s and %s Margin with df %s is: %s' %
              (method, penalty, str(self.minDf), str(svm_precision)))


    def plot_ROC(self, yScore, method):
        assert method == "LSI" or method == "NMF"
        if method == "LSI":
            fpr, tpr, thresholds = roc_curve(self.yLSITesting, yScore)
        elif method == "NMF":
            fpr, tpr, thresholds = roc_curve(self.yNMFTesting, yScore)

        roc_auc = auc(fpr, tpr)

        print(len(fpr))
        print(len(tpr))
        print(len(thresholds))

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

    # make confusion matrix plot
    def plot_confusion_matrix(self, cmat, classname, normalize=False, title='Confusion matrix'):
        plt.figure()
        cmap = plt.cm.Blues
        plt.imshow(cmat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classname))
        plt.xticks(tick_marks, classname, rotation=45)
        plt.yticks(tick_marks, classname)

        if normalize:
            cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]

        print(cmat)

        thresh = cmat.max() / 2.
        for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
            plt.text(j, i, cmat[i, j], horizontalalignment="center", color="white" if cmat[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    """
    (f) Use a 5-fold cross-validation to find the best value of the parameter
    """
    def problemF(self, method):

        if self.XLSITraining is None or self.yLSITraining is None or self.XNMFTraining is None or self.yNMFTraining is None:
            self.problemD()     # will have to use everything in part D

        if method == "LSI":
            XTrain, yTrain, XTest, yTest = \
                self.XLSITraining, self.yLSITraining, self.XLSITesting, self.yLSITesting
        else:
            XTrain, yTrain, XTest, yTest = \
                self.XNMFTraining, self.yNMFTraining, self.XNMFTesting, self.yNMFTesting

        kf = KFold(n_splits=5, shuffle=True)
        matrix = [[0]*7 for i in range(5)]
        
        #
        # LSI
        #
        i = 0
        for train_index, test_index in kf.split(XTrain):
            X_train, X_test = XTrain[train_index], XTrain[test_index]
            j = 0
            for k in [-3, -2, -1, 0, 1, 2, 3]:
                y_train = [ int(x / 4) for x in self.eightTrainingData.target[train_index]]    
                y_test = [ int(x / 4) for x in self.eightTrainingData.target[test_index]]    
                SVC = svm.LinearSVC(C=10**k)
                SVC.fit(X_train, y_train)
                score = SVC.score(X_test, y_test)
                matrix[i][j]=score
                j = j + 1
            i = i + 1

        avg_value = np.array(matrix)
        print (avg_value.shape)

        max = 0
        max_index = 0
        for i in range (7):
            mean = np.mean(avg_value[:,i:i+1])
            if max < mean:
                max = mean
                max_index = i
        print (max, max_index)
        penalty = [-3, -2, -1, 0, 1, 2, 3]
        print ('The best penalty value for '+str(method)+' method is',10**penalty[max_index]) 

        self.problemE(method, 10**penalty[max_index])


    def problemGH(self, classifier, method, penalty="l2", reg=1):
        if self.XLSITraining is None or self.yLSITraining is None or \
            self.XNMFTraining is None or self.yNMFTraining is None:
            self.problemD()     # will have to use everything in part D

        assert classifier == "MultiNB" or classifier == "Logi"
        if classifier == "MultiNB":
            clf = GaussianNB()
        else:
            clf = LogisticRegression(penalty=penalty, C=reg)

        assert method == "LSI" or method == "NMF"
        if method == "LSI":
            XTrain, yTrain, XTest, yTest = \
                self.XLSITraining, self.yLSITraining, self.XLSITesting, self.yLSITesting
        else:
            XTrain, yTrain, XTest, yTest = \
                self.XNMFTraining, self.yNMFTraining, self.XNMFTesting, self.yNMFTesting

        clf.fit(XTrain, yTrain)
        if classifier == "MultiNB":
            yScore = clf.predict(XTest)
        else:
            yScore = clf.decision_function(XTest)
        self.plot_ROC(yScore, method)
        plt.savefig('fig/roc_%s_%s_penalty_%s_reg_%s_df%d.png' %
                    (classifier, method, penalty, str(reg), self.minDf), bbox_inches='tight')
        plt.show()


    def problemI(self, method):
        if self.XLSITraining is None or self.yLSITraining is None or \
                self.XNMFTraining is None or self.yNMFTraining is None:
            self.problemD()     # will have to use everything in part D

        assert method == "LSI" or method == "NMF"
        if method == "LSI":
            XTrain, yTrain, XTest, yTest = \
                self.XLSITraining, self.yLSITraining, self.XLSITesting, self.yLSITesting
        else:
            XTrain, yTrain, XTest, yTest = \
                self.XNMFTraining, self.yNMFTraining, self.XNMFTesting, self.yNMFTesting

        for penalty in ["l1", "l2"]:
            for reg in [0.01, 0.1, 1, 10, 100, 1000]:
                self.problemGH("Logi", "LSI", penalty, reg)


    def fetch_data(self, subset, cate):
        data = fetch_20newsgroups(subset=subset, categories=cate, shuffle=True,
                                    random_state=42, remove=('headers','footers','quotes'))
        return data

    def dim_red(self, method):
        pass

    def problemJ(self):
        categories_j = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
                        'misc.forsale','soc.religion.christian']



def main():
    # debug mytokenizer (spam)
    # corpus = ['stem stems stemming, go stemmed']
    # # a = Project1(minDf=2)
    # # countVec = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, tokenizer=a.mytokenizer)
    # countVec = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, tokenizer=mytokenizer())
    # out = countVec.fit_transform(corpus)
    # name = countVec.get_feature_names()
    # print(name)

    #
    p = Project1(minDf=2)
    # p.problemA()
    # p.problemB()
    # p.problemC()
    # p.problemD()
    # p.problemE("LSI", "hard")
    # p.problemE("LSI", "soft")
    # p.problemE("NMF", "hard")
    # p.problemE("NMF", "soft")
    # p.problemF()
    # p.problemGH("MultiNB", "LSI")
    p.problemI("LSI")

    # p.problemF('LSI')
    # p.problemF('NMF')
    # p.problemGH()
    # p.problemJ()
    # p.problemGH("MultiNB", "LSI")

if __name__ == "__main__":
    main()