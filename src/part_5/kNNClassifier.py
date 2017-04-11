__author__ = 'Nilay Shrivastava,nilayshrivastava1729@gmail.com'

import numpy

class kNearestNeighbours:

	def __init__( self,k = 3 ):
		self.k = k

	def fit( self, x_train, y_train ):
		self.x = x_train
		self.y = y_train

	def predict( self, testFeatures ):
		
		def EuclidDist( testPoint , checkPoint ):
			distance = numpy.linalg.norm( testPoint - checkPoint )
			return distance

		def closest( testPoint ):
			distArray = numpy.array( [ ( EuclidDist( testPoint , self.x[i] ), self.y[i] ) for i in range(len(self.x)) ] , dtype = [('dist', float),('lab', int)] )
			distArray.sort(order = 'dist')
			majority  = {}
			for j in range(self.k):
				if majority.get( distArray[j][1] ) == None:
					majority[ distArray[j][1] ] = 0
				else:
					majority[ distArray[j][1] ] += 1

			return max( majority , key = majority.get )

		if testFeatures.ndim == 1:    #only one point to predict
			return closest( testFeatures )
		else:
			# prediction = []
			# for testPoint in testFeatures:
			# 	prediction.append( closest( testPoint ) )
			# return numpy.array(prediction)			
			prediction = numpy.array( [closest(point) for point in testFeatures] )
			return prediction 


if __name__ == '__main__':

	from check import CheckClassifier
	#IRISDATASET


	#IRIS DATASET
	feature_set = numpy.loadtxt('Fisher.csv',delimiter = ',',skiprows = 1,usecols = (1,2,3,4))
	label_set   = numpy.loadtxt('Fisher.csv',delimiter = ',',skiprows = 1, usecols = (0,))


	#TITANIC DATASET
	#raw_feature_set = numpy.loadtxt('titanic.txt',delimiter = ',',skiprows = 1,usecols = ())

	x_train , x_test = numpy.vsplit(feature_set,2)
	y_train , y_test = numpy.hsplit(label_set,2)

	kNN = kNearestNeighbours(5)
	kNN.fit(x_train,y_train)


	#print('Test features: ', x_test[0])
	#print('Test Label: ',y_test[0])
	#print('Prediction: ',kNN.predict( numpy.array( [24,51,28,58] ) ) )
	print(CheckClassifier( kNN , x_test , y_test ))