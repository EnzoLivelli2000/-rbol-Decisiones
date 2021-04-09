from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

iris = load_iris()
AEntrena,APruebas,BEntrena,BPruebas=train_test_split(iris['data'],iris['target'])


clasficadorArbol=DecisionTreeClassifier()
clasficadorArbol.fit(AEntrena,BEntrena)

DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
 random_state=None, max_leaf_nodes=None, 
 min_impurity_split=None, class_weight=None, ccp_alpha=0.0)

print("Aprendió en un: ")
print(clasficadorArbol.score(APruebas,BPruebas))
print(clasficadorArbol.score(AEntrena,BEntrena))

export_graphviz(clasficadorArbol,out_file='arbol.dot', class_names=iris['target_names'],feature_names=iris['feature_names'],impurity=False,filled=True)

with open('arbol.dot') as f:
    dot_graph=f.read()
graphviz.Source(dot_graph).view()