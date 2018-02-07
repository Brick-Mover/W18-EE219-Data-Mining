import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, homogeneity_score, completeness_score, 
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score)
from sklearn.decomposition import TruncatedSVD, NMF
import matplotlib.pyplot as plt
import itertools


min_df = 3
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

# fetch all the data
eightTrainingData = fetch_20newsgroups(subset='all', categories=categories,
                                remove=('headers','footers','quotes'))

eightLabels = [ int(x / 4) for x in eightTrainingData.target ]

# plot contingency matrix plot
def plot_contingency_matrix(label_true, label_pred, classname, normalize=False, title='Contingency Matrix'):
    plt.figure()
    cmat = confusion_matrix(label_true, label_pred)
    cmap = plt.cm.Blues
    plt.imshow(cmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classname))
    plt.xticks(tick_marks, classname, rotation=45)
    plt.yticks(tick_marks, classname)

    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]

    # print(cmat)

    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        if normalize == False:
            plt.text(j, i, cmat[i, j], horizontalalignment="center", color="white" if cmat[i, j] > thresh else "black")
        else:
            plt.text(j, i, "%.2f"%cmat[i, j], horizontalalignment="center", color="white" if cmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# ys has the format [[y1,y1_label],[y2, y2_label]]
def make_plot(x, ys, xlabel, ylabel, xticks=None, grid=False):
    for y, label in ys:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(x)
    plt.legend()
    if grid == True:
        plt.grid()
    plt.show()

def main():
    print("=" * 60)
    """
    Problem 1: Building the TF-IDF matrix.
    """
    print('-' * 60)
    pipe = Pipeline( [
        ('vect', CountVectorizer(min_df=3, analyzer='word', stop_words='english')),
        ('tfidf', TfidfTransformer())
    ] )
    TFIDF = pipe.fit_transform(eightTrainingData.data)
    print("p1: dimension of the TF-IDF matrix is: ", TFIDF.shape)

    """
    Problem 2: Apply K-means clustering with k = 2 using the TF-IDF data.
    """
    print('-' * 60)
    km = KMeans(n_clusters=2)
    km.fit(TFIDF)

    """
    Problem 2a
    """

    class_names = ['Recreation', 'Com Tech']
    title = 'TFIDF_k=2'
    plot_contingency_matrix(eightLabels, km.labels_, class_names, normalize=False, title=title)
    plt.show()

    """
    Problem 2b
    """

    print("Homogeneity: %0.3f" % homogeneity_score(eightLabels, km.labels_))
    print("Completeness: %0.3f" % completeness_score(eightLabels, km.labels_))
    print("V-measure: %0.3f" % v_measure_score(eightLabels, km.labels_))
    print("Adjusted rand score: %.3f" % adjusted_rand_score(eightLabels, km.labels_))
    print("Adjusted mutual info score: %.3f" % adjusted_mutual_info_score(eightLabels, km.labels_))

    """
    Problem 3: Apply SVD, NMF to TFIDF and plot measure scores vs n_components.
    """
    print('-' * 60)

    # SVD
    # SVD
    rank = 1000
    svd = TruncatedSVD(n_components=rank)
    svd.fit_transform(TFIDF)
    ratios = svd.explained_variance_ratio_
    # print(ratios)  
    np.sort(ratios)
    print(ratios.shape)
    print(ratios[0:10])

    # make the plot
    x = np.array(range(1,rank+1))
    y = [[ratios,'Percent of Retained Variance']]
    xlabel = 'Rank r'
    ylabel = 'Percent of Retained Variance'
    make_plot(x, y, xlabel, ylabel)

    # NMF
    # nmf = NMF(n_components=1000)
    # nmf.fit_transform(TFIDF)
    # svd.fit_transform(TFIDF)
    # ratios = svd.explained_variance_ratio_
    # np.sort(ratios)

    print("=" * 60)

if __name__ == "__main__":
    main()