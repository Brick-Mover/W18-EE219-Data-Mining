from sklearn.datasets import fetch_20newsgroups

categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
               'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey' ]

#trainingData = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)


def plot_size():
    for category in categories:
        trainingData = fetch_20newsgroups(subset='train', categories=[category])
        print(category, trainingData.filenames.shape[0])

def main():
    #plot_size()
    pass

if __name__ == "__main__":
    main()