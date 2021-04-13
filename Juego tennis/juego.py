from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz

#Se crea la instancia del árbol de decisión.
decisionTree = tree.DecisionTreeClassifier()
juegoTennis = pd.DataFrame()

#Creamos los datos, según lo pedido en la rúbrica
juegoTennis['cielo'] = ['lluvioso', 'lluvioso', 'nublado', 'soleado', 'soleado', 'soleado', 'nublado', 'lluvioso', 'lluvioso', 'nublado', 'soleado']
juegoTennis['ambiente'] = ['calido', 'calido', 'calido', 'temp media', 'frio', 'frio', 'frio', 'temp media', 'frio', 'temp media', 'temp media']
juegoTennis['humedo'] = ['alto', 'alto', 'alto', 'alto', 'normal', 'normal', 'normal', 'alto', 'normal', 'alto', 'alto']
juegoTennis['velocidadViento'] = ['bajo', 'alto', 'bajo', 'bajo', 'bajo', 'alto', 'alto', 'bajo', 'bajo', 'alto', 'alto']
juegoTennis['juego'] = ['no','no','si','si','si','no','si','no','si','si','no'] 

#Eliminamos la columna juego, ya que no la necesitaremos por ahora
inputs = juegoTennis.drop('juego', axis='columns')
#Guardando la columna juego en la variable target
target = juegoTennis['juego']

#Codificamos las variables
le_cielo = LabelEncoder()
le_ambiente = LabelEncoder()
le_humedo = LabelEncoder()
le_velocidadViento = LabelEncoder()

#Asignamos las columnas antiguas a nuesvas variables
inputs['cielo_n'] = le_cielo.fit_transform(inputs['cielo'])
inputs['ambiente_n'] = le_cielo.fit_transform(inputs['ambiente'])
inputs['humedo_n'] = le_cielo.fit_transform(inputs['humedo'])
inputs['velocidadViento_n'] = le_cielo.fit_transform(inputs['velocidadViento'])

print(inputs)

inputs_n = inputs.drop(['cielo', 'ambiente', 'humedo', 'velocidadViento'], axis= 'columns')
#conseguimos las direcciones de cada elemento en las columnas
one_hot_data = pd.get_dummies(inputs[['cielo_n', 'ambiente_n', 'humedo_n', 'velocidadViento_n']])

print(one_hot_data)

#entrenamos el arbol con la funcion fit, le pasamos los datos a entrenar y las output's
decisionTree = decisionTree.fit(inputs_n,juegoTennis['juego'])
print(decisionTree.score(inputs_n,juegoTennis['juego']))

#utilzamos la función predict para solicitarle al arbol ya entrenado una posible solución a nuestro problema
decision = decisionTree.predict([[1,0,1,1]])

#realizamos una pequena validacion para poder mostra un mensaje en la pantalla
if decision == 'si':
    print(' hoy SI se juega')
else:
    print(' hoy NO se juega')

#esta parte comentada aún está en desarrollo
""" dot_data = tree.export_graphviz(decisionTree, out_file='juego.dot',feature_names=list(inputs_n), class_names=['Not_Play', 'Play'], rounded=True, filled=True)

with open('juego.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph).view()
 """
