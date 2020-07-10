import math
from sklearn import model_selection, metrics
import numpy as np

class C45:

    """Creates a decision tree with C4.5 algorithm"""

    def __init__(self, pathToData, pathToNames):
        self.filePathToData = pathToData
        self.filePathToNames = pathToNames
        self.data = []
        self.dataTrain = []
        self.dataTest = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.tree = None

    def fetchData(self):
        with open(self.filePathToNames, "r") as file:
            line = ""
            # Omitir comentarios
            for line in file:
                if line[0] != "|":
                    break
            classes = line
            self.classes = [x.strip() for x in classes.split(",")]
            # add attributes
            for line in file:
                [attribute, values] = [x.strip() for x in line.split(":")]
                values = [x.strip() for x in values.split(",")]
                self.attrValues[attribute] = values
        self.numAttributes = len(self.attrValues.keys())
        self.attributes = list(self.attrValues.keys())
        with open(self.filePathToData, "r") as file:
            for line in file:
                row = [x.strip() for x in line.split(",")]
                if row != [] or row != [""]:
                    self.data.append(row)

    def preprocessData(self):
        for index, row in enumerate(self.data):
            for attr_index in range(self.numAttributes):
                if(not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.data[index][attr_index] = float(
                        self.data[index][attr_index])

        self.dataTrain, self.dataTest = model_selection.train_test_split(
            self.data, test_size=0.3, random_state=3)

    def printTree(self):
        self.printNode(self.tree)

    def printNode(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    valuesForAttribute = self.attrValues[node.label]
                    if child.isLeaf:
                        if child.label != 'Fail':
                            print(indent + node.label + " = " +
                                valuesForAttribute[index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " +
                              valuesForAttribute[index] + " : ")
                        self.printNode(child, indent + "	")
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " +
                          str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " +
                          str(node.threshold)+" : ")
                    self.printNode(leftChild, indent + "	")

                if rightChild.isLeaf:
                    print(indent + node.label + " > " +
                          str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " +
                          str(node.threshold) + " : ")
                    self.printNode(rightChild, indent + "	")

    def generateTree(self):
        self.tree = self.recursiveGenerateTree(self.dataTrain, self.attributes)

    def recursiveGenerateTree(self, curData, curAttributes):
        allSame = self.allSameClass(curData)

        if len(curData) == 0:
            # Fail
            return Node(True, "Fail", None)
        elif allSame is not False:
            # return a node with that class
            return Node(True, allSame, None)
        elif len(curAttributes) == 0:
            # return a node with the majority class
            majClass = self.getMajClass(curData)
            return Node(True, majClass, None)
        else:
            (best, best_threshold, splitted) = self.splitAttribute(
                curData, curAttributes)
            
            if best == -1:
                # return a node with the majority class
                majClass = self.getMajClass(curData)
                return Node(True, majClass, None)
            remainingAttributes = curAttributes[:]
            remainingAttributes.remove(best)
            node = Node(False, best, best_threshold)
            node.children = [self.recursiveGenerateTree(
                subset, remainingAttributes) for subset in splitted]
            return node

    def getMajClass(self, curData):
        freq = [0]*len(self.classes)
        for row in curData:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]

    def allSameClass(self, data):
        if len(data) == 0:
            return False

        for row in data:
            if row[-1] != data[0][-1]:
                return False
        return data[0][-1]

    def isAttrDiscrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True

    def splitAttribute(self, curData, curAttributes):
        # print('\ncurData:', curData)
        # print('curAttributes:', curAttributes)

        splitted = []
        maxEnt = -1*float("inf")
        best_attribute = -1
        # None for discrete attributes, threshold value for continuous attributes
        best_threshold = None

        for attribute in curAttributes:
            # print('attribute:', attribute)
            # print('self.attributes:', self.attributes)
            indexOfAttribute = self.attributes.index(attribute)

            if self.isAttrDiscrete(attribute):
                # split curData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                valuesForAttribute = self.attrValues[attribute]
                # print('valuesForAttribute', valuesForAttribute)
                subsets = [[] for a in valuesForAttribute]

                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break

                e = self.gain(curData, subsets)
                if e > maxEnt:
                    maxEnt = e
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
            else:
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                curData.sort(key=lambda x: x[indexOfAttribute])
                for j in range(0, len(curData) - 1):
                    if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
                        threshold = (
                            curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
                        less = []
                        greater = []
                        for row in curData:
                            if(row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])
                        if e >= maxEnt:
                            splitted = [less, greater]
                            maxEnt = e
                            best_attribute = attribute
                            best_threshold = threshold
        return (best_attribute, best_threshold, splitted)

    def gain(self, unionSet, subsets):
        # input : data and disjoint subsets of it
        # output : information gain
        S = len(unionSet)
        # calculate impurity before split
        impurityBeforeSplit = self.entropy(unionSet)
        # calculate impurity after split
        weights = [len(subset)/S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i]*self.entropy(subsets[i])
        # calculate total gain
        totalGain = impurityBeforeSplit - impurityAfterSplit
        return totalGain

    def entropy(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in dataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        num_classes = [x/S for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num*self.log(num)
        return ent*-1

    def log(self, x):
        if x == 0:
            return 0
        else:
            return math.log(x, 2)

    def predictNode(self, node, data_row):
        if node.isLeaf:
            return node.label
        else:
            indexOfAttribute = self.attributes.index(node.label)

            if node.threshold is None:
                # discrete
                for index, child in enumerate(node.children):
                    valuesForAttribute = self.attrValues[node.label]
                    if data_row[indexOfAttribute] == valuesForAttribute[index]:
                        return self.predictNode(child, data_row)
            else:
                # numerical
                leftChild = node.children[0]
                rightChild = node.children[1]

                if data_row[indexOfAttribute] <= node.threshold:
                    return self.predictNode(leftChild, data_row)
                else:
                    return self.predictNode(rightChild, data_row)

    def predict(self, data):
        predicted = []
        for data_row in data:
            predicted.append(self.predictNode(self.tree, data_row))
        return predicted

    def score(self, data):
        predicted = self.predict(data)
        return metrics.accuracy_score(np.array(data)[:, -1], predicted)


class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []
