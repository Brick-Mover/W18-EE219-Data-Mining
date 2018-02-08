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

print("fetch data")
eightTrainingData = fetch_20newsgroups(subset='all', categories=categories)
eightLabels = [int(x / 4) for x in eightTrainingData.target]

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
    plt.show()

# ys has the format [[y1,y1_label],[y2, y2_label]]
def make_plot(x, ys, xlabel, ylabel, xticks=None, grid=False, title=None):
    for y, label in ys:
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(x)
    plt.legend()
    if grid == True:
        plt.grid()
    if title is not None:
        plt.title(title)
    plt.show()

def prob_3a_ii(method, XData=None, tfidf=None):
    ranks = np.array([1,2,3,5,10,20,50,100,300])
    np_hg = np.array([])
    np_cp = np.array([])
    np_vm = np.array([])
    np_ari = np.array([])
    np_ami = np.array([])
    class_names = ['Com Tech', 'Recreation']

    for r in ranks:
        if method == 'NMF':
            nmf = NMF(n_components=r, max_iter=(50 if r==300 else 200))
            data = nmf.fit_transform(tfidf)
        else:
            data = XData[:,:r]
        km = KMeans(n_clusters=2, max_iter=100, n_init=20)
        km.fit(data)

        title = str(method)+' Rank '+str(r)
        plot_contingency_matrix(eightLabels, km.labels_, class_names, 
                               normalize=False, title=title)

        np_hg = np.append(np_hg, homogeneity_score(eightLabels, km.labels_))
        np_cp = np.append(np_cp, completeness_score(eightLabels, km.labels_))
        np_vm = np.append(np_vm, v_measure_score(eightLabels, km.labels_))
        np_ari = np.append(np_ari, adjusted_rand_score(eightLabels, km.labels_))
        np_ami = np.append(np_ami, adjusted_mutual_info_score(eightLabels, km.labels_))

    x = ranks
    ys = [[np_hg, 'Homogeneity'],[np_cp, 'Completeness'], [np_vm, 'V_measure']]
    xlabel = 'Rank r'
    ylabel = 'Score'
    title = str(method)+' Score'
    make_plot(x, ys, xlabel, ylabel, title=title)
    ys = [[np_ari, 'Adjusted Rand Index'], [np_ami, 'Adjusted Mutual Info']]
    make_plot(x, ys, xlabel, ylabel, title=title)
    return np_hg, np_cp, np_vm, np_ari, np_ami

def visualize_in_2D(method, data, bestR):
    reduced_data = data[:,:bestR]
    km = KMeans(n_clusters=2, max_iter=100, n_init=20)
    km.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = km.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset ('+str(method)+'-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
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
    Problem 2b: Apply K-means clustering with k = 2 using the TF-IDF data.
    """
    print('-' * 60)
    km = KMeans(n_clusters=2, max_iter=100, n_init=20)
    km.fit(TFIDF)

    class_names = ['Com Tech', 'Recreation']
    title = 'TFIDF_k=2'
    plot_contingency_matrix(eightLabels, km.labels_, class_names, normalize=False, title=title)

    """
    Problem 2b: Apply K-means clustering with k = 2 using the TF-IDF data.
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
    rank = 1000
    svd = TruncatedSVD(n_components=rank)
    svd_X = svd.fit_transform(TFIDF)

    ratios = np.array([])
    sum = 0
    for ratio in svd.explained_variance_ratio_:
        sum = sum + ratio
        ratios = np.append(ratios, sum)
    print(ratios[0:10])
    x = np.array(range(1,rank+1))
    y = [[ratios,'Percent of Retained Variance']]
    xlabel = 'Rank r'
    ylabel = 'Percent of Retained Variance'
    make_plot(x, y, xlabel, ylabel)
    print(type(svd_X))
    print(svd_X[:,:2].shape)

    # problem 3 (a) ii
    # SVD
    hg, cp, vm, ari, ami = prob_3a_ii('SVD', svd_X)

    hg, cp, vm, ari, ami = prob_3a_ii('NMF', tfidf=TFIDF)

    # 4(a)
    best_SVD = 2
    best_NMF = 2

    # visualize SVD
    visualize_in_2D('SVD', svd_X, best_SVD)

    # visualize NMF
    nmf = NMF(n_components=best_NMF)
    nmf_X = nmf.fit_transform(TFIDF)
    visualize_in_2D('NMF', nmf_X, best_NMF)


    print("=" * 60)


if __name__ == "__main__":
    main()