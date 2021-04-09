from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
import graphviz

decisionTree = tree.DecisionTreeClassifier()
juegoTennis = pd.DataFrame()

juegoTennis['cielo'] = ['lluvioso', 'lluvioso', 'nublado', 'soleado', 'soleado', 'soleado', 'nublado', 'lluvioso', 'lluvioso', 'nublado', 'soleado']
juegoTennis['ambiente'] = ['calido', 'calido', 'calido', 'temp media', 'frio', 'frio', 'frio', 'temp media', 'frio', 'temp media', 'temp media']
juegoTennis['humedo'] = ['alto', 'alto', 'alto', 'alto', 'normal', 'normal', 'normal', 'alto', 'normal', 'alto', 'alto']
juegoTennis['velocidadViento'] = ['bajo', 'alto', 'bajo', 'bajo', 'bajo', 'alto', 'alto', 'bajo', 'bajo', 'alto', 'alto']
juegoTennis['juego'] = ['no','no','si','si','si','no','si','no','si','si','no'] 

print(juegoTennis)
one_hot_data = pd.get_dummies(juegoTennis[['cielo', 'ambiente', 'humedo', 'velocidadViento']])
print(one_hot_data)

decisionTree = decisionTree.fit(one_hot_data,juegoTennis['juego'])

dot_data = tree.export_graphviz(decisionTree, out_file='juego.dot', feature_names=list(one_hot_data.columns.values),
rounded=True, filled=True)
print(dot_data)


with open('juego.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph).view()


""" graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png()) """

""" X = [
    [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42],
    [181, 85, 43]
]

Y = [
    'hombre','hombre','mujer','mujer',
    'hombre','hombre','mujer','mujer',
    'mujer','hombre','hombre'
] """