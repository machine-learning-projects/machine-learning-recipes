# Writing Our First Classifier - Machine Learning Recipes #5 - https://youtu.be/AoeEHqVSNOw

from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    """
    Barebones KNN
    """

    def fit(self, X_train, y_train):
        """
        Takes features and labels for training set as input
        :param X_train:
        :param y_train:
        :return:
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Receives features for testing data
        Output predictions for labels
        :param X_test:
        :return:
        """
        predictions = []
        for row in X_test:
            # label = random.choice(self.y_train)  # Random decision
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        """
        Find the closest training point
        :param row:
        :return:
        """
        # Distance from test point to first training point
        best_dist = euc(row, self.X_train[0])  # Shortest distance found so far
        best_index = 0  # index of closest training point
        for i in xrange(1, len(self.X_train)):  # Iterate over all other training points
            dist = euc(row, self.X_train[i])
            if dist < best_dist:  # Found closer, update
                best_dist = dist
                best_index = i
        return self.y_train[best_index]  # closest example


from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test, predictions)
