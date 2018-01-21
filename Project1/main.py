from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text


categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
               'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey' ]

class Project1(object):

    def __init__(self, minDf):
        self.eightTrainingData = None
        self.minDf = minDf

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
        XTrainCounts = countVec.fit_transform(self.eightTrainingData.data)
        print('Size of feature vectors when minDf is %s: %s' % (self.minDf, XTrainCounts.shape))

        # compute tf-idf
        tfidfTransformer = TfidfTransformer()
        XTrainTfidf = tfidfTransformer.fit_transform(XTrainCounts)
        print('Size of tf-idf when minDf is %s: %s' % (self.minDf, XTrainTfidf.shape))


    """
    (c) Report 10 most significant terms base on tf-icf
    """
    def problemC(self):
        self.calc_tf_icf()

    def calc_tf_icf(self):
        pass



def main():
    p = Project1(minDf=2)
    p.problemB()

if __name__ == "__main__":
    main()