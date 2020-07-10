#!/usr/bin/env python
import pdb
from c45.c45 import C45

model = C45("./data/salary/adult.data", "./data/salary/adult.names")
model.fetchData()
model.preprocessData()
model.generateTree()

print("Arbol de desicion:")
model.printTree()

print("\nDatos totales:", len(model.data))

print("Datos de entrenamiento:", len(model.dataTrain))
score = model.score(model.dataTrain)
print( "Exactitud:", score )

print("Datos de prueba:", len(model.dataTest))
score = model.score(model.dataTest)
print( "Precision:", score )