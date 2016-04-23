# Part 2 - Visualizing a Decision Tree

# Use Iris flower data set: https://en.wikipedia.org/wiki/Iris_flower_data_set
# Identify type of flower based on measurements
# Dataset includes 3 species of Iris flowers: setosa, versicolor, virginica
# 4 features used to describe: length and width of sepal and petal
# 50 examples of each type for 150 total examples

# Goals
# 1-Import dataset
# 2-Train a classifier
# 3-Predict label for new flower
# 4-Visualize the tree

# scikit-learn datasets: http://scikit-learn.org/stable/datasets/
# already includes Iris dataset: load_iris

from sklearn.datasets import load_iris

iris = load_iris()

print iris.feature_names  # metadata: names of the features
print iris.target_names  # metadata: names of the different types of flowers
# print iris.data  # features and examples themselves
print iris.data[0]  # first flower
print iris.target[0]  # contains the labels

# print entire dataset
# for i in xrange(len(iris.target)):
#     print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

# Testing Data
# Examples used to test the classifier's accuracy
# Not part of the training data

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# here, we remove the first example of each flower
# found at indices: 0, 50, 100
test_idx = [0, 50, 100]

# create 2 new sets of variables, for training and testing
# training data
# remove the entires from the data and target variables
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# create new classifier
clf = tree.DecisionTreeClassifier()
# train on training data
clf.fit(train_data, train_target)

# what we expect
print test_target
# what tree predicts
print clf.predict(test_data)

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn.externals.six import StringIO
import pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
