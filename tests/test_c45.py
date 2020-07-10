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
		accuracy = tree.score(tree.dataTrain)
		precision = tree.score(tree.dataTest)

		if accuracy == 1.0 and precision == 1.0:
			self.assertEqual(True, True, msg ="Accuracy: " + str(accuracy) + ", precision: " + str(precision))
		else:
			self.assertEqual(False, True, msg="Accuracy: " + str(accuracy) + ", precision: " + str(precision))


def main():
	unittest.main()

if __name__ == '__main__':
	main()

