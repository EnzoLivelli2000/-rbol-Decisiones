from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz
#ESTE CÓDIGO FUNCIONA IGUAL QUE JUEGO.PY, LA DIFERENCIA ES QUE ESTA IMPLEMEMTACIÓN NOS PROPORCIONA INCERTIDUMBRE
juegoTennis = pd.DataFrame()

juegoTennis['cielo'] = ['lluvioso', 'lluvioso', 'nublado', 'soleado', 'soleado', 'soleado', 'nublado', 'lluvioso', 'lluvioso', 'nublado', 'soleado']
juegoTennis['ambiente'] = ['calido', 'calido', 'calido', 'temp media', 'frio', 'frio', 'frio', 'temp media', 'frio', 'temp media', 'temp media']
juegoTennis['humedo'] = ['alto', 'alto', 'alto', 'alto', 'normal', 'normal', 'normal', 'alto', 'normal', 'alto', 'alto']
juegoTennis['velocidadViento'] = ['bajo', 'alto', 'bajo', 'bajo', 'bajo', 'alto', 'alto', 'bajo', 'bajo', 'alto', 'alto']
juegoTennis['juego'] = ['no','no','si','si','si','no','si','no','si','si','no'] 

one_hot_data = pd.get_dummies(juegoTennis[ ['cielo', 'ambiente', 'humedo', 'velocidadViento'] ])

print(one_hot_data)

clf = tree.DecisionTreeClassifier()

clf_train = clf.fit(one_hot_data, juegoTennis['juego'])

print(tree.export_graphviz(clf_train, None))

prediction = clf_train.predict([[1,0,0,
1,0,0,
1,0,
0,1]])

print(prediction)

""" dot_data = tree.export_graphviz(clf_train, out_file='juego2.dot', feature_names=list(one_hot_data.columns.values), 
class_names=['No_Play', 'Play'], rounded=True, filled=True) 

with open('juego2.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph).view() """

""" prediction = clf_train.predict([[0,0,1,
1,
0,0,0,0,1,
1]]) """
