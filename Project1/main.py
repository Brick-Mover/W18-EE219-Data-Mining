from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text


categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
               'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey' ]


"""
Q1: plot a histogram of the number of training documents per class to check if they are evenly distributed.
"""
def plot_size():
    for category in categories:
        trainingData = fetch_20newsgroups(subset='train', categories=[category])
        print(category, len(trainingData.filenames))


"""
Modeling Text Data: tokenize each document into words. Then, excluding the stop words,
punctuations, and using stemmed version of words, create a TFxIDF vector representations.
min_df = 2 or 5
"""


def model_text_data(minDf):
    # load data
    eightTrainData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

    # tokenization
    countVec = CountVectorizer(min_df=minDf, stop_words=text.ENGLISH_STOP_WORDS)
    XTrainCounts = countVec.fit_transform(eightTrainData.data)
    print('Size of feature vectors when minDf is %d: %d' % (minDf, XTrainCounts.shape))

    # compute tf-idf
    tfidfTransformer = TfidfTransformer()
    XTrainTfidf= tfidfTransformer.fit_transform(XTrainCounts)
    print('Size of tf-idf when minDf is %d: %d' % (minDf, XTrainTfidf.shape))

def main():
    #plot_size()
    model_text_data()

if __name__ == "__main__":
    main()