from sklearn import tree
import pandas as pd
#Se crea la instancia del árbol de decisión.
clf = tree.DecisionTreeClassifier()

#instanciamos calf como dataframe, con el propósito de poder acceder a sus datos por medio de una matriz
calf = pd.DataFrame()

#nuestro data set
calf['tiempo'] = [10,20,10,30,10,20,40,30,10,30,20,40,10,30,10,20]
calf['materia'] = ['matematica', 'historia', 'ciencia', 'historia', 
'matematica', 'matematica', 'matematica', 'ciencia',
'historia', 'ciencia', 'ciencia', 'historia',
'ciencia', 'matematica', 'matematica', 'ciencia']
calf['nota'] = [11,16,11,18,11,15,20,17,10,17,15,20,11,18,11,15]

#convertimos las variables a dummies para poder accder a sus posiciones con mayor facilidad
one_hot_data = pd.get_dummies(calf[ ['tiempo', 'materia'] ])

print(one_hot_data)

# instanciamos el arbol de desiciones y lo asignamos a la variable clf
clf = tree.DecisionTreeClassifier()

# entrenamos nuestro arbol con los data ses creados
clf_train = clf.fit(one_hot_data, calf['nota'])

# realizamos una predicción en funcion de los siguientes datos
prediction = clf_train.predict([[20,0,1,0]])

print(prediction)

#realizamos una simple condicional para poder mostrar un mensaje en la pantalla
if prediction >= 13:
    print("Estas aprobado")
else:
    print("Estas reprobado")
