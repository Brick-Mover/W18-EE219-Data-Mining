import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA


min_df = 3
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

eightTrainingData = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers','footers','quotes'))

eightLabels = eightTrainingData.target

def main():
    print("=" * 40)
    """
    Problem 1: Building the TF-IDF matrix.
    """
    print('-' * 40)
    pipe = Pipeline( [
        ('vect', CountVectorizer(min_df=3, analyzer='word', stop_words='english')),
        ('tfidf', TfidfTransformer())
    ] )
    TFIDF = pipe.fit_transform(eightTrainingData.data)
    print("p1: dimension of the TF-IDF matrix is: ", TFIDF.shape)

    """
    Problem 2: Apply K-means clustering with k = 2 using the TF-IDF data.
    """
    print('-' * 40)
    km = KMeans(n_clusters=2)
    km.fit(TFIDF)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(eightLabels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(eightLabels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(eightLabels, km.labels_))
    print("Adjusted rand score: %.3f" % metrics.adjusted_rand_score(eightLabels, km.labels_))
    print("Adjusted mutual info score: %.3f" % metrics.adjusted_mutual_info_score(eightLabels, km.labels_))

    """
    Problem 3: Apply K-means clustering with k = 2 using the TF-IDF data.
    """
    pca = PCA(n_components=1000)
    pca.fit(TFIDF)
    ratios = pca.explained_variance_ratio_
    np.sort(ratios)

    print("=" * 40)

if __name__ == "__main__":
    main()