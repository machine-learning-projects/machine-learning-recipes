def CheckClassifier( classifier , x_test , y_test ):
	total = len(y_test)
	correct = 0
	for i in range(total):
		prediction = classifier.predict( x_test[i] )
		if prediction == y_test[i]:
			correct += 1

	return (correct/total)*100