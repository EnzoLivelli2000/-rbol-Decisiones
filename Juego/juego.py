from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz

decisionTree = tree.DecisionTreeClassifier()
juegoTennis = pd.DataFrame()

juegoTennis['cielo'] = ['lluvioso', 'lluvioso', 'nublado', 'soleado', 'soleado', 'soleado', 'nublado', 'lluvioso', 'lluvioso', 'nublado', 'soleado']
juegoTennis['ambiente'] = ['calido', 'calido', 'calido', 'temp media', 'frio', 'frio', 'frio', 'temp media', 'frio', 'temp media', 'temp media']
juegoTennis['humedo'] = ['alto', 'alto', 'alto', 'alto', 'normal', 'normal', 'normal', 'alto', 'normal', 'alto', 'alto']
juegoTennis['velocidadViento'] = ['bajo', 'alto', 'bajo', 'bajo', 'bajo', 'alto', 'alto', 'bajo', 'bajo', 'alto', 'alto']
juegoTennis['juego'] = ['no','no','si','si','si','no','si','no','si','si','no'] 

inputs = juegoTennis.drop('juego', axis='columns')
target = juegoTennis['juego']

le_cielo = LabelEncoder()
le_ambiente = LabelEncoder()
le_humedo = LabelEncoder()
le_velocidadViento = LabelEncoder()

inputs['cielo_n'] = le_cielo.fit_transform(inputs['cielo'])
inputs['ambiente_n'] = le_cielo.fit_transform(inputs['ambiente'])
inputs['humedo_n'] = le_cielo.fit_transform(inputs['humedo'])
inputs['velocidadViento_n'] = le_cielo.fit_transform(inputs['velocidadViento'])

print(inputs)

inputs_n = inputs.drop(['cielo', 'ambiente', 'humedo', 'velocidadViento'], axis= 'columns')

print(inputs_n)

decisionTree = decisionTree.fit(inputs_n,juegoTennis['juego'])
print(decisionTree.score(inputs_n,juegoTennis['juego']))


decision = decisionTree.predict([[1,0,1,1]])

if decision == 'si':
    print(' hoy SI se juega')
else:
    print(' hoy NO se juega')

'''
inputs['le_cielo'] = le_cielo.fit_transform(inputs['company'])

print(juegoTennis)
one_hot_data = pd.get_dummies(juegoTennis[['cielo', 'ambiente', 'humedo', 'velocidadViento']])
print(one_hot_data)

decisionTree = decisionTree.fit(one_hot_data,juegoTennis['juego'])

print(decisionTree.score(one_hot_data,juegoTennis['juego']))

dot_data = tree.export_graphviz(decisionTree, out_file='juego.dot', feature_names=list(one_hot_data.columns.values),
rounded=True, filled=True)
print(dot_data)

prediction = decisionTree.predict([[0,0,1,0,1,0,0,1,1,0]])
print(prediction) '''

''' with open('juego.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph).view() '''
