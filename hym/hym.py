from sklearn import tree
#Se crea la instancia del árbol de decisión.

clf = tree.DecisionTreeClassifier()

#[altura, peso, talla de zapato]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
#La salida donde se dice si es hombre o mujer
Y = ['hombre', 'hombre', 'mujer', 'mujer', 'hombre', 'hombre', 'mujer', 'mujer',
     'mujer', 'hombre', 'hombre']
#Se le pasa los datos  X y Y

clf = clf.fit(X, Y)
print(clf)

#Se definen los datos 1 y 2, estos datos corresponde a un hombre
dato1 = [170, 59, 42]

prediction = clf.predict([dato1])


print(prediction)
if prediction == 'hombre':
    print("Estas caraterísticas corresponden a un hombre")
else:
    print("Estas caraterísticas corresponden a un mujer")
