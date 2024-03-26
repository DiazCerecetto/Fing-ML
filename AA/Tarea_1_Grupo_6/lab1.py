import matplotlib.pyplot as plt
import csv
import pandas as pd
from collections import Counter
import numpy as np
import math
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


#########################################################
########## FUNCIONES DE PREPROCESAMIENTO ################
#########################################################

# Abrimos el archivo CSV y modificamos los valores de "Target para que sean numéricos como se indicó"
# A su vez, reemplazamos el archivo proporcionado ya que este era interpretado como
# un solo elemento de tipo string por el comando pd.read_csv() al tener sus valores.
# separados por ;.
# Imprimimos a su vez el conjunto D para corroborar

# detectar_columnas_no_discretas(data): dado el nombre del archivo csv
# "data" pasado por parámetro, se detectan las columnas discretas
# este procedimiento modificará el conjunto de entrenamiento 
# codificando las etiquetas
def detectar_columnas_no_discretas(data):
  with open(data, newline='', encoding="utf-8") as archivo_csv:
      lector_csv = csv.reader(archivo_csv, delimiter=';')
      filas_modificadas = []
      #Codificacion de etiquetas:
      for fila in lector_csv:
          fila_modificada = fila
          if fila[-1] == 'Dropout':
            fila_modificada[-1] = 0
          elif fila[-1] != 'Target':
            fila_modificada[-1] = 1
          filas_modificadas.append(fila_modificada)
  #Guardamos los cambios en el archivo
  with open(data, 'w', newline='', encoding="utf-8") as archivo_csv:
      escritor_csv = csv.writer(archivo_csv)
      escritor_csv.writerows(filas_modificadas)
  #Cargamos el archivo en pandas para obtener las columnas no discretas
  D = pd.read_csv(data)
  # Número de columnas de D, corroboramos que cambiar ';' por ',' no nos genere nuevos
  # datos que puedan modificar la estructura del conjunto de entrenamiento. Este número
  # debe ser 37. Los 36 atributos y los valores de la función objetivo.
  print("\nNumero de columnas en D: " + str(D.shape[1]))
  # Número de columnas de D con atributos contínuos
  columnas_continuas = D.select_dtypes(include=['float64']).columns
  print("\n" + str(columnas_continuas))
  return



# ganancia_threshold(data, col, target, threshold):
# dado un conjunto de valores para un atributo, separarlo en función de un
# umbral pasado por parámetro. Retornamos el valor de la ganancia correspondiente
def ganancia_threshold(data, col, target, threshold):

    # Dividimos el conjunto basandonos en el umbral pasado por parámetro
    izq = data[data[col] <= threshold]
    der = data[data[col] > threshold]

    # Calculamos la proporcion de muestras
    p_left = len(izq) / len(data)
    p_right = len(der) / len(data)

    ganancia = entropia_tupla(data[target]) - (p_left * entropia_tupla(izq[target]) + p_right * entropia_tupla(der[target]))
    return ganancia

# entropia_preprocesamiento(data): calcula la entropía del conjunto data
# pasado por parámetro, data debe ser una lista de numeros, un vector.
def entropia_tupla(data):
    total = len(data)
    counts = data.value_counts()
    probs = counts / total
    return -sum(p * math.log2(p) for p in probs)

# discretizar(data, col, target): discretizamos
# nuestro conjunto de datos basandonos en la ganancia correspondiente
def discretizar(data, col, target):

    # Obtenemnos los valores únicos y los ordenamos
    valores_unicos = sorted(data[col].unique())
    ganancia_maxima = -float('inf')
    mejor_threshold = None

    # Buscamos el mejor punto de corte
    for i in range(len(valores_unicos) - 1):
        threshold = (float(valores_unicos[i]) + float(valores_unicos[i + 1])) / 2
        ganancia = ganancia_threshold(data, col, target,threshold)
        if ganancia > ganancia_maxima:
            ganancia_maxima = ganancia
            mejor_threshold = threshold

    # Discretizamos la columna correspondiente y retornamos
    if mejor_threshold is not None:
        return pd.cut(data[col], bins=[-float('inf'), mejor_threshold, float('inf')], labels=['0', '1'])

# preprocesar_columna(numero_columna,D): realiza el preprocesamiento de la columna
# numero_columna de D. Discretizando y separando por valores
def preprocesar_columna(numero_columna,D):
  feature_column = [float(row[numero_columna]) for row in D]
  target_column = [float(row[-1]) for row in D]
  data2 = pd.DataFrame({
      'feature':feature_column,
      'target':target_column
  })
  data2 = discretizar(data2, 'feature', 'target')
  for i in range (len(D)):
    D[i][numero_columna] = str(data2[i])
  return D

def preprocesar_corpus(columnas,X_prec):
    for elemento in columnas:
        X_train = preprocesar_columna(elemento,X_prec)
    return X_prec


######################################################
########## IMPLEMENTACION DEL ALGORITMO ##############
######################################################

# Clase Nodo: Implementa la estructura del árbol clasificador
class Nodo:
    def __init__(self, atributo=None, ramas=None, resultado=None):
        self.atributo = atributo
        self.ramas = ramas or {}
        self.resultado = resultado

# Cálculo de la entropía del conjunto "data" pasado por parámetro
def entropia(data):
    data = list(data)
    total_elementos = len(data)
    counter_data = Counter(data)
    return (
       -sum((count / total_elementos) * math.log2(count / total_elementos) 
            for count in counter_data.values())
    )

# Cálculo de la ganancia correspondiente 
# Cálculo de la ganancia
def ganancia(data, attribute_col, numero_columna):
    entropia_total = entropia(fila[numero_columna] for fila in data)
    valores_atributos = set(fila[attribute_col] for fila in data)
    suma = 0.0

    for valor in valores_atributos:
        subconjunto_valor = [fila for fila in data if fila[attribute_col] == valor]
        suma += (len(subconjunto_valor) / len(data)) * entropia([fila[numero_columna] for fila in subconjunto_valor])

    return entropia_total - suma

# Implementación del algoritmo ID3
def id3(X_train,y_train,total_data, attributes, target_col, min_samples_split=2, min_split_gain=0.0):

    data = armar_data(X_train,y_train)

    # Si todos los ejemplos tienen la misma clase, devuelve ese valor de clase
    etiquetas_unicas = set(y_train)
    if len(etiquetas_unicas) == 1:
        return Nodo(resultado=etiquetas_unicas.pop())

    # Si la lista de atributos está vacía, devuelve el valor de clase más común
    if not attributes:
        common_target = Counter(y_train).most_common(1)[0][0]
        return Nodo(resultado=common_target)


    # Verifica los hiperparámetros de división mínima
    if len(X_train) < min_samples_split:
        common_target = Counter(y_train).most_common(1)[0][0]
        return Nodo(resultado=common_target)

    
    # Selecciona el atributo que tiene la mayor ganancia de información
    maximo = 0
    mejor_atributo = 1
    for elemento in attributes:
        ganancia_actual = ganancia(data, elemento, target_col)
        if ganancia_actual > maximo:
            maximo = ganancia_actual
            mejor_atributo = elemento

    #mejor_atributo = max(attributes, key=lambda attr: ganancia(data, attr, target_col))
    attributes = [attr for attr in attributes if attr != mejor_atributo]

    if ganancia(data, mejor_atributo, target_col) < min_split_gain:
        common_target = Counter(y_train).most_common(1)[0][0]
        return Nodo(resultado=common_target)

    

    # Crea un nodo de decisión basado en el mejor atributo
    tree = Nodo(atributo=mejor_atributo)
    for value in set(fila[mejor_atributo] for fila in total_data):
        subset = [fila for fila in data if fila[mejor_atributo] == value]
        if (subset == []):
            # Cuando no tenemos ejemplos para un cierto valor del atributo considerado, etiquetamos con el valor más probable
            # En este caso se cumple que "subset == []".
            tree.ramas[value] = Nodo(resultado=Counter(row[target_col] for row in data).most_common(1)[0][0])
        else:

            # Transformamos el conjunto previamente a las llamadas recursivas
            y_train = []
            X_train = []
            for i in range(len(subset)):
                y_train.append(subset[i][target_col])
                X_train.append(subset[i][:-1])

            # Verificamos la división mínima en el subconjunto, en ese caso es una hoja
            # sino, llamamos recursivamente.
            if len(subset) < min_samples_split:
                
                tree.ramas[value] = Nodo(resultado=Counter(row[target_col] for row in data).most_common(1)[0][0])
            else:
                tree.ramas[value] = id3(X_train,y_train,total_data, attributes, target_col, min_samples_split, min_split_gain)
    return tree



def armar_data(X_train,y_train):
    data_ret = []
    for i in range(len(X_train)):
        combinados = X_train[i] + [y_train[i]]
        data_ret.append(combinados)
    return data_ret





def classify(tree, instance):
    # Si estamos en un nodo hoja, devolvemos el resultado
    if tree.resultado is not None:
        return int(tree.resultado)

    # Si no es un nodo hoja, buscamos el valor del atributo en la instancia
    valor_atributo = instance[tree.atributo]

    # Si el valor del atributo no está en el árbol (es decir, no vimos este valor durante el entrenamiento),
    # esto es una limitación del algoritmo ID3 original y puede tratarse de diversas formas.
    # Por simplicidad, aquí simplemente devolveremos 1.
    if valor_atributo not in tree.ramas:
        return 1

    # De lo contrario, seguimos la rama adecuada y continuamos con la clasificación
    return classify(tree.ramas[valor_atributo], instance)

##########################################################################
######################### FUNCIONES DE EVALUACION ########################
##########################################################################

# Retorna la medida f1 del resultado de evaluar el conjunto de testeo pasado
# como parametro mediante la recorrida del árbol "tree" 
def obtener_f1(tree,X_test,y_test):
  resultados = []
  for i in range(0,len(X_test)):
    aux = X_test[i].copy()
    resultados.append(classify(tree,aux))
  f1 = f1_score(y_test, resultados, average="macro")
  return f1

# imprimir_resultados(tree,X_test,y_test):
# evalúa el arbol en el conjunto test e imprime los resultados en medidas
# accuracy, recall, f1 y precision, tanto por clases como en promedio
def imprimir_resultados(tree,X_test,y_test):
  resultados = []
  for i in range(0,len(X_test)):
    aux = X_test[i].copy()
    resultados.append(int(classify(tree,aux)))

  # Usando precisión (accuracy)
  accuracy = accuracy_score(y_test, resultados)
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Macro-F1: "+ str(obtener_f1(tree, X_test, y_test)))
  # Reporte de clasificación
  print(classification_report(y_test, resultados))

def evaluar_id3_intercambiando_parametros(X_train, y_train,X_test,y_test, attribute_indices, min_split_gain_range, min_samples_split_range):
    for min_samples_split in min_samples_split_range:
        for min_split_gain in min_split_gain_range:
          total_data = armar_data(X_train,y_train)
          tree = id3(X_train, y_train,total_data,range(36), 36, min_samples_split, min_split_gain)
          f1 = obtener_f1(tree, X_test, y_test)
          print("Macro-F1  para min_samples_split=" + str(min_samples_split) + " y min_split_gain="+ str(min_split_gain) + " :" + str(f1))
    return


################################################
############## PROGRAMA PRINCIPAL ##############
################################################



############ CARGA DE DATOS #############


# Abrimos el archivo CSV y modificamos los valores de "Target para que sean numéricos como se indicó"
# A su vez, reemplazamos el archivo proporcionado ya que este era interpretado como
# un solo elemento de tipo string por el comando pd.read_csv() al tener sus valores.
# separados por ;.
# Imprimimos a su vez el conjunto D para corroborar
def cargar_archivo(nombre_archivo):
    #Se carga el corpus de entrenamiento desde el archivo csv proporcionado,
    #habiendo modificado los elementos como en la parte anterior
    D = []
    with open(nombre_archivo,'r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)
        for fila in lector_csv:
            D.append([float(x) for x in fila])
    return D

############ EVALUACION #############

def crear_conjuntos_evaluacion(D):
   
    # Dividir D en características (X) y etiquetas (y)
    X = [row[:-1] for row in D]  # Todo excepto la última columna
    y = [row[-1] for row in D]   # Solo la última columna

    # Dividir en entrenamiento y prueba
    test_size = 0.2
    data_size = len(D)
    test_size = int(data_size * test_size)

    # Usar random para mezclar y dividir
    indices = list(range(data_size))
    random.seed(42)  # para reproducibilidad
    random.shuffle(indices)

    #Cargamos los datasets divididos
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    #Pasamos de string a números los tipos de los datos para convertir a np.array en el algoritmo
    for i in range(len(X_train)):
        instancia = X_train[i]
        for j in range(len(instancia)):
            instancia[j] = int(instancia[j])

    for i in range(len(X_test)):
        instancia = X_test[i]
        for j in range(len(instancia)):
            instancia[j] = int(instancia[j])
        
    return X_train,y_train,X_test,y_test


# Función para predecir un conjunto de muestras
def predict_Set(tree, X):
    return [classify(tree,sample) for sample in X]

# Función de validación cruzada, para el algoritmo ID3. La que propone scikit-learn la utilizamos
# para los algoritmos DecisionTree y RandomForest. Por cuestiones de tiempo implementamos esta adaptacion
# para no rever la implementacion de ID3.
def cross_validation(X, y, num_folds, min_samples_split, min_split_gain):
    kf = KFold(n_splits=num_folds, shuffle=True)
    results = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = [X[i] for i in train_idx], [X[i] for i in test_idx]
        y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

        attributes = list(range(len(X_train[0]))) # Asumiendo que X_train es una lista de listas
        total_data = armar_data(X_train,y_train)
        tree = id3(X_train, y_train,total_data, attributes,36, min_samples_split, min_split_gain)
        predictions = predict_Set(tree, X_test)

        f1 = f1_score(y_test, predictions, average="macro")
        results.append(f1)

    return results



#######################################
############## PARTE ID3 ##############
#######################################

def ID3_CrossValidation(X_train, y_train, X_test, y_test, folds, min_samples_split, min_split_gain):
  X = X_train + X_test #En 5-Cross-Validation usamos todos los datos para entrenar
  y = y_train + y_test
  results = cross_validation(X, y, folds, min_samples_split, min_split_gain)  
  print("Macro-F1 para cada fold: ",results )
  print("Media de la Macro-F1:", sum(results) / len(results))
  return



#################################################
############## PARTE Decision Tree ##############
#################################################

def sk_DecisionTreeClassifier(X_train, y_train, X_test, y_test, min_samples, min_impurity):
  # Entrenamos el modelo
  clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_samples, min_impurity_decrease=min_impurity)
  clf.fit(X_train, y_train)
  # Realizamos la prediccion
  y_pred = clf.predict(X_test)
  return y_pred



def evaluar_DecisionTree(X_train, y_train, X_test, y_test, rangoSamples, rangoSplit):
  print("Evaluación para Decision Tree:")
  for sample in rangoSamples:
    for splitGain in rangoSplit:
      y_pred1 = sk_DecisionTreeClassifier(X_train, y_train, X_test, y_test, sample, splitGain)
      f1 = f1_score(y_test, y_pred1, average="macro")
      print("Macro-F1  para min_samples_split=" + str(sample) + " y min_impurity_decrease=" + str(splitGain) + " :     ",f1)
  return

def reporte_DecisionTree(X_train, y_train, X_test, y_test, min_samples_splitAux, min_impurity_decreaseAux):
  y_pred = sk_DecisionTreeClassifier(X_train, y_train, X_test, y_test, min_samples_splitAux, min_impurity_decreaseAux)
  f1 = f1_score(y_test, y_pred, average="macro")
  print(classification_report(y_test, y_pred))
  print("Macro-F1 aplicando tecnica de particionamiento 80/20 del dataset para entrenar y testear: " + str(f1))

  # Ahora comparamos con los resultados que se obtienen al entrenar el calsificador mediante la tecnica 5-Cross-Validation
  clf = DecisionTreeClassifier(random_state=0, min_samples_split=min_samples_splitAux, min_impurity_decrease=min_impurity_decreaseAux)
  X = X_train + X_test #En 5-Cross-Validation usamos todos los datos para entrenar
  y = y_train + y_test
  scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
  print("Luego de aplicar 5-Cross-Validation:")
  print("Macro-F1 para cada fold:", scores)
  print("Media de la Macro-F1:", np.mean(scores))
  print("Desviación estándar de la Macro-F1:", np.std(scores))
  return




#################################################
############## PARTE Random Forests #############
#################################################

def sk_RandomForestClassifier(X_train, y_train, X_test, y_test, min_samples, min_impurity):
  # Entrenamos el modelo
  clf = RandomForestClassifier(random_state=0, min_samples_split=min_samples, min_impurity_decrease=min_impurity)
  clf.fit(X_train, y_train)
  # Realizamos la prediccion
  y_pred = clf.predict(X_test)
  return y_pred


def evaluar_RandomForest(X_train, y_train, X_test, y_test, rangoSamples, rangoSplit):
  print("Evaluación para Random Forest:")
  for sample in rangoSamples:
    for splitGain in rangoSplit:
      y_pred1 = sk_RandomForestClassifier(X_train, y_train, X_test, y_test, sample, splitGain)
      f1 = f1_score(y_test, y_pred1, average="macro")
      print("Macro-F1  para min_samples_split=" + str(sample) + " y min_impurity_decrease=" + str(splitGain) + " :     ",f1)
  return


def reporte_RandomForest(X_train, y_train, X_test, y_test, min_samples_splitAux, min_impurity_decreaseAux):
  y_pred = sk_RandomForestClassifier(X_train, y_train, X_test, y_test, min_samples_splitAux, min_impurity_decreaseAux)
  f1 = f1_score(y_test, y_pred, average="macro")
  print(classification_report(y_test, y_pred))
  print("Macro-F1 aplicando tecnica de particionamiento 80/20 del dataset para entrenar y testear: " + str(f1))

  # Ahora comparamos con los resultados que se obtienen al entrenar el calsificador mediante la tecnica 5-Cross-Validation
  clf = RandomForestClassifier(random_state=0, min_samples_split=min_samples_splitAux, min_impurity_decrease=min_impurity_decreaseAux)
  X = X_train + X_test #En 5-Cross-Validation usamos todos los datos para entrenar
  y = y_train + y_test
  scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
  print("Luego de aplicar 5-Cross-Validation:")
  print("Macro-F1 para cada fold:", scores)
  print("Media de la Macro-F1:", np.mean(scores))
  print("Desviación estándar de la Macro-F1:", np.std(scores))
  return





###############################################################
################## RESUMEN Y ANALISIS GRÁFICO #################
###############################################################
def grafica(x_values, y_values, z_values, z_label , x_label, y_label, title):
        plt.figure(figsize=(7, 3))
        for i, z_value in enumerate(z_values):
            plt.plot(x_values, y_values[i], marker='o', label=f"{z_label}={z_value}")

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(title=z_label)
        plt.grid(True)
        plt.show()

def graficar_mejor_macro_f1(algoritmos, mejor_macro):
        plt.figure(figsize=(7, 3))
        plt.bar(algoritmos, mejor_macro, color='skyblue')
        plt.title('Mejor Macro-F1 por Algoritmo')
        plt.xlabel('Algoritmo')
        plt.ylabel('Mejor Macro-F1')
        plt.ylim(0.7, 0.85)  # Ajusta los límites del eje y según tus datos
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

def crear_analisis_grafico():
    #Datos
    min_split_gain_values = [0, 0.001, 0.01, 0.1]
    min_samples_split_id3 = [1, 10, 20, 30, 40, 50]
    min_samples_split_values = [2, 10, 20, 30, 40, 50]






    #ID3
    macro_f1_id3 = [
    [0.7791, 0.7841, 0.7929, 0.7889, 0.7889, 0.7790],
    [0.7791, 0.7841, 0.7929, 0.7889, 0.7889, 0.7790],
    [0.7791, 0.7841, 0.7929, 0.7889, 0.7889, 0.7790],
    [0.7900, 0.7858, 0.7910, 0.7870, 0.7870, 0.7790]
    ]

    #Decision Tree
    macro_f1_dt = [
        [0.7704, 0.7872, 0.7928, 0.7986, 0.8013, 0.8024],
        [0.8080, 0.8080, 0.8080, 0.8080, 0.8096, 0.8035],
        [0.7955, 0.7955, 0.7955, 0.7955, 0.7955, 0.7955],
        [0.7804, 0.7804, 0.7804, 0.7804, 0.7804, 0.7804]
    ]
    
    #Random Forest
    macro_f1_rf = [
        [0.8302, 0.8197, 0.8208, 0.8205, 0.8205, 0.8236],
        [0.8259, 0.8205, 0.8160, 0.8186, 0.8152, 0.8205],
        [0.7860, 0.7860, 0.7860, 0.7860, 0.7860, 0.7891],
        [0.3978, 0.3978, 0.3978, 0.3978, 0.3978, 0.3978]
    ]
    colors = ['b', 'c', 'g', 'r']

    # Grafica para id3
    plt.figure(figsize=(10, 5))
    for i, min_split_gain in enumerate(min_split_gain_values):
        plt.plot(min_samples_split_id3, macro_f1_id3[i], marker='o', label=f"min_split_gain={min_split_gain}", color=colors[i], alpha=0.7, markersize=8)

    plt.title('ID3')
    plt.xlabel('min_samples_split')
    plt.ylabel('Macro-F1')
    plt.legend(title='min_split_gain')
    plt.grid(True)
    plt.show()

    

    grafica(min_samples_split_values, macro_f1_dt, min_split_gain_values, 'min_impurity_decrease', 'min_samples_split_values', 'Macro-F1', 'Decision Tree')
    grafica(min_samples_split_values, macro_f1_rf, min_split_gain_values, 'min_impurity_decrease', 'min_samples_split_values', 'Macro-F1', 'Random Forest')


    

    # Datos
    algoritmos = ["ID3", "Decision Tree", "Random Forest"]
    mejor_macro = [0.7929487179487178, 0.8096002020157007 , 0.8302770522179805 ]

    graficar_mejor_macro_f1(algoritmos, mejor_macro)



def main():

    print("\nPreprocesando corpus de entrada...\n")


    print("\n#####################################################")
    print("\nObtenemos las columnas no discretas y codificamos las etiquetas")
    
    
    # Codificamos las etiquetas y obtenemos las columnas no discretas 
    detectar_columnas_no_discretas("data.csv")



    print("\n#####################################################")
    # Cargamos el archivo en la lista de listas "D"
    
    D = cargar_archivo("data.csv")
    X_train,y_train,X_test,y_test = crear_conjuntos_evaluacion(D)

    #######################################################

    X_train = preprocesar_corpus([6,12,25,31,33,34,35],X_train)
    X_test = preprocesar_corpus([6,12,25,31,33,34,35],X_test)

    #######################################################


    
    # Defimos parametros
    min_split_gain_range = [0, 0.001, 0.01, 0.1]
    rango = [1, 10, 20, 30, 40, 50]
    print("\n#####################################################")
    print("Laboratorio 1 Grupo 6 - Aprendizaje automático 2023")
    print("#####################################################\n")

    print("Entrega de código, presentamos este menú para ejecutar cada caso de forma sencilla.")
    print("Por favor, agregue el archivo 'data.csv' proporcionado en el mismo directorio que este script.\n")


    print("\nSeleccione una opción:")
    while True:
        
        print("1. Evaluar ID3 con variaciones de parámetros (Sección 3.1.1)")
        print("2. Evaluar ID3 con los mejores parámetros observados (Sección 3.1.1)")
        print("3. Evaluar ID3 usando 5-Cross-Validation")
        print("4. Evaluar DecisionTree dividiendo el conjunto")
        print("5. Evaluar DecisionTree usando 5-Cross-Validation")
        print("6. Evaluar RandomForest dividiendo el conjunto")
        print("7. Evaluar RandomForest usando 5-Cross-Validation")
        print("8. Salir")
        eleccion = input("Ingrese su elección: ")

        if eleccion == '1':
            print("\n#####################################################")
            print("Evaluación de ID3 en conjunto de testing 80-20 con parámetros:")
            print("\n")
            print("min_split_gain: " + str(min_split_gain_range))
            print("min_samples_split: " + str(rango))
            print("\n")
            print("#####################################################\n")
            evaluar_id3_intercambiando_parametros(X_train, y_train,X_test,y_test,range(36),min_split_gain_range,rango)
            pass
        elif eleccion == '2':
            print("\n#####################################################")
            print("Evaluación de ID3 en conjunto de testing 80-20 con parámetros óptimos:")
            print("\n")
            total_data = armar_data(X_train,y_train)

            treeOpt = id3(X_train, y_train,total_data, range(36),36, 20 , 0)
            imprimir_resultados(treeOpt, X_test, y_test)
            print("\n")
            print("#####################################################\n")
            pass
        elif eleccion == '3':
            print("\n#####################################################")
            print("Evaluación de ID3 mediante 5-Cross-Validation:")
            print("#####################################################\n")
            ID3_CrossValidation(X_train, y_train, X_test, y_test, 5, 40, 0)
            pass
        elif eleccion == '4':
            print("\n#####################################################")
            print("Evaluación de DecisionTree particionando el dataset:")
            print("#####################################################\n")
            evaluar_DecisionTree(X_train, y_train, X_test, y_test,[2,10,20,30,40,50],[0,0.001,0.01,0.1])
            pass
        elif eleccion == '5':
            print("\n#####################################################")
            print("Evaluación de DecisionTree mediante 5-Cross-Validation:")
            print("#####################################################\n")
            reporte_DecisionTree(X_train, y_train, X_test, y_test, 20, 0.001)

            pass
        elif eleccion == '6':
            print("\n#####################################################")
            print("Evaluación de RandomForest particionando el dataset:")
            print("#####################################################\n")
            evaluar_RandomForest(X_train, y_train, X_test, y_test,[2,10,20,30,40,50],[0,0.001,0.01,0.1])
            pass
        elif eleccion == '7':
            print("\n#####################################################")
            print("Evaluación de DecisionTree mediante 5-Cross-Validation:")
            print("#####################################################\n")
            reporte_RandomForest(X_train, y_train, X_test, y_test, 30, 0)
            pass
        elif eleccion == '8':
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Por favor, ingrese una opción válida.")

        print("\nFin de la evaluación. Ingrese otra opción")






if __name__ == "__main__":
    main()