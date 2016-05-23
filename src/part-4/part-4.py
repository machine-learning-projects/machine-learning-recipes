# Let's Write a Pipeline - Machine Learning Recipes #4 - https://youtu.be/84gqSbLcBFE

# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

# Can think of classifier as a function f(x) = y
X = iris.data  # features
y = iris.target  # labels

# partition into training and testing sets
from sklearn.cross_validation import train_test_split

# test_size=0.5 -> split in half
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Classifier
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print predictions

# test
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)

#Repeat using KNN
# Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print predictions

# test
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
