import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

# Import the dataset
mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# limit size of datasets for a faster experiment
max_examples = 10000
data = data[:max_examples]
labels = labels[:max_examples]


def display(i):
    """
    Display example digits
    :param i: example number (not the label)
    """
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)


display(0)  # display example 0, label 7
display(1)  # display example 1, label 2
display(8)  # display example 8, label 5

# fit a linear classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)

# evaluate linear classifier accuracy
classifier.evaluate(test_data, test_labels)
print(classifier.evaluate(test_data, test_labels)["accuracy"])

# classify some examples
# this will be classified correctly:
# print("Predicted %d, Label: %d" % (classifier.predict(test_data[0]), test_labels[0]))
display(0)
#
# # this will be classified incorrectly:
# print("Predicted %d, Label: %d" % (classifier.predict(test_data[8]), test_labels[8]))
display(8)

# visualize learned weights
weights = classifier.get_variable_value("linear//weight/d/linear//weight/part_0/Ftrl_1")
f, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(())  # ticks be gone
    a.set_yticks(())
plt.show()
