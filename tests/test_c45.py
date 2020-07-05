#!/usr/bin/env python
import unittest
from c45.c45 import C45

class testC45Methods(unittest.TestCase):
	def testIris(self):
		c1 = C45("./data/iris/iris.data", "./data/iris/iris.names")
		c1.fetchData()
		c1.preprocessData()
		c1.generateTree()

		self.__testTree(c1)
	
	def testLenses(self):
		c1 = C45("./data/lenses/lenses.data", "./data/lenses/lenses.names")
		c1.fetchData()
		c1.preprocessData()
		c1.generateTree()

		self.__testTree(c1)

	def testDivorce(self):
		c1 = C45("./data/divorce/divorce.data", "./data/divorce/divorce.names")
		c1.fetchData()
		c1.preprocessData()
		c1.generateTree()

		self.__testTree(c1)

	def __testTree(self, tree):
		success_predict = 0
		for data_row in tree.data:
			prediction = tree.predict(data_row)
			# print(prediction)
			# print(data_row[-1])
			if prediction == data_row[-1]:
				success_predict += 1

		if success_predict == len(tree.data):
			self.assertEqual(True, True, msg ="Accuracy: " + str(success_predict) + "/" + str(len(tree.data)))
		else:
			self.assertEqual(False, True, msg="Accuracy: " + str(success_predict) + "/" + str(len(tree.data)))


def main():
	unittest.main()

if __name__ == '__main__':
	main()

