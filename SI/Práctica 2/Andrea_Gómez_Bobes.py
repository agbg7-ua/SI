#Se han comentado las ejecuciones de las funciones ya que no me ha dado tiempo a hacer la función main con cada una de las partes del trabajo
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from keras.datasets import mnist as keras_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
semilla = int(time.time())
np.random.seed(20)
## Clase para introducir MNIST
def MNIST_adaboost():
    (X_train,Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data()
    X_train= X_train.reshape((X_train.shape[0], 28*28)).astype("float32")/255.0
    X_test= X_test.reshape((X_test.shape[0], 28*28)).astype("float32")/255.0
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")
    
    Y_train_binary = np.where(Y_train == 9,1,-1)
    Y_test_binary = np.where(Y_test == 9,1,-1)
    return X_train, Y_train_binary, X_test, Y_test_binary
##Clase de los clasificadores débiles
class DecisionStump:
    def __init__(self, n_features):
        self.feature = np.random.randint(0,n_features-1)
        self.umbral = np.random.rand()
        self.polaridad = np.random.choice([-1,1])
        self.error = 0.0
    def predict(self, X):
        resultado = np.ones(X.shape[0])
        ##Si la polaridad es -1 y la característica mayor o igual que el umbral entonces el resultado es -1
        if self.polaridad == 1:
            resultado[X[:,self.feature] < self.umbral] = -1
        ##Si la polaridad es 1 y la característica menor que el umbral entonces el resultado es 1
        else:
            resultado[X[:,self.feature] >= self.umbral] = -1
            
        return resultado
##Clase para crear el clasificador adaboost
class Adaboost:
 def __init__(self, T=5, A=20):
     self.T = T
     self.A = A
     self.clasificadores=[]
 def fit(self, X, Y, verbose = False):
     n_observaciones,n_caracteristicas = X.shape
     pesos = np.ones(n_observaciones)/n_observaciones
     for ent in range(self.T):
        mejorError=1
        for clas in range(self.A):
            #Creamos el clasificador débil y calculamos su error
            clasificador = DecisionStump(n_caracteristicas)
            predicciones = clasificador.predict(X)
            error = np.sum(pesos*(predicciones != Y))
            #Si el error es menor que el mejor error encontrado hasta el momento se considera este clasificador como el mejor y se sigue
            if error < mejorError:
                mejorClas =clasificador
                clasificador.error = error
                mejorError = error
        alfa = 0.5 * np.log((1 - mejorError) / (mejorError + 1e-10))
        
        pesos = pesos * np.exp(-alfa* Y * predicciones)
        Z = np.sum(pesos)
        pesos /= Z
        mejorClas.alfa = alfa
        self.clasificadores.append(mejorClas)
        print(f"Añadido clasificador {ent +1}: Características: {mejorClas.feature}, Umbral: {mejorClas.umbral}, Polaridad: {mejorClas.polaridad}, Error:{mejorError}")
        print(f"Iteración {ent+1}/{self.T}, Error:{mejorError}")
 def predict(self, X):
     prediccion = [clas.alfa * clas.predict(X) for clas in self.clasificadores]
     pred = np.sum(prediccion,axis=0)
     pred = np.sign(pred)
     return pred
def prueba(T,A):
    (X_train,Y_train_binary,X_test,Y_test_binary) = MNIST_adaboost()
    class_adaboost = Adaboost(T=T,A=A)
    print(f"Entrenando clasificador Adaboost para T = {T}, A={A}")
    inicio = time.time()
    class_adaboost.fit(X_train,Y_train_binary,True)
    final = time.time()
    total = final - inicio
    
    y_test_pr = class_adaboost.predict(X_test)
    y_train_pr = class_adaboost.predict(X_train)
    
    precision_test = accuracy_score(Y_test_binary, y_test_pr)*100
    precision_train = accuracy_score(Y_train_binary, y_train_pr)*100
    
    print(f"Tasas acierto(train,test) y tiempo: {precision_train:.2f}%, {precision_test:.2f}%, {total} s.")
    return
#prueba(T=6,A=6)
def grafica_adaboost():
    valores_T = [5,10,15,20,25,30]
    valores_A = [5,10,15,20,25,30]
    precision_resultados = []
    
    for t in valores_T:
       for a in valores_A:
           X_train, Y_train_binary, X_test, Y_test_binary = MNIST_adaboost()
           
           adaboost = Adaboost(T=t, A = a)
           inicio = time.time()
           adaboost.fit(X_train,Y_train_binary)
           final = time.time()
           tiempo = final - inicio
           
           test_pr = adaboost.predict(X_test)
           train_pr = adaboost.predict(X_train)
           precision_test = accuracy_score(Y_test_binary,test_pr)
           test = precision_test*100
           precision_train = accuracy_score(Y_train_binary, train_pr)*100
           print(f"Tasas acierto(train,test) y tiempo: {precision_train:.2f}%, {test:.2f}%, {tiempo} s.")
           
           precision_resultados.append((t,a,precision_test,tiempo))    
    
    fig, ax1 = plt.subplots()
    color = 'tab:pink'
    color2 = 'black'
    ax1.set_xlabel('T y A')
    ax1.set_ylabel('Porcentaje de acierto', color=color)
    ax1.plot(range(len(precision_resultados)), [result[2] for result in precision_resultados], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color2)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Tiempo', color=color)
    ax2.plot(range(len(precision_resultados)), [result[3] for result in precision_resultados], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.show()
#grafica_adaboost()
def MNIST_sklearn():
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser = 'auto')
    datos = mnist.data.astype("float32") / 255.0
    etiquetas = mnist.target.astype("int8")
    etiquetas_bin = np.where(etiquetas == 9, 1, -1)
    datos_tr, datos_test, etiquetas_tr, etiquetas_test = train_test_split(datos, etiquetas_bin, test_size=0.2, random_state=40)
    return datos_tr, datos_test, etiquetas_tr, etiquetas_test

def adaboost_sklearn(clasificadores):
    tiempos = []
    tasas = []
    datos_tr, datos_test, etiquetas_tr, etiquetas_test = MNIST_sklearn()
    for clasificador in range(1, clasificadores+1):
        inicio = time.time()
        adaboost = AdaBoostClassifier(n_estimators=clasificador, random_state=semilla)
        adaboost.fit(datos_tr, etiquetas_tr)
        fin = time.time()
        tiempo = fin - inicio

        predicciones = adaboost.predict(datos_test)
        precision = accuracy_score(etiquetas_test, predicciones)
        tiempos.append(tiempo)
        tasas.append(precision)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, clasificadores+1), tasas, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de clasificadores')
    plt.ylabel('Tasa de Acierto')
    plt.title('Clasificador AdaBoost con sklearn')
    plt.grid(True)
    plt.show()
    print(f"Tiempo de ejecución {tiempo} s.")

clasificadores = 15
#adaboost_sklearn(clasificadores)
def MNIST_mlp():
    (X_train, Y_train), (X_test, Y_test) = keras_mnist.load_data()
    
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')/255.0
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')/255.0
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    return (X_train,X_test,Y_train,Y_test)

def clas_mlp():
    (X_train,X_test,Y_train,Y_test) = MNIST_mlp()
    modelo = Sequential()
    modelo.add(Flatten(input_shape=(28,28,1)))
    modelo.add(Dense(128, activation = 'relu'))
    modelo.add(Dense(10,activation='softmax'))
    
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    historial = modelo.fit(X_train, Y_train, validation_data=(X_test,Y_test), validation_split = 0.3, epochs = 10,verbose=2)
    
    evaluacion = modelo.evaluate(X_test, Y_test, verbose=2)
    precision = evaluacion[1]*100
    
    plt.plot(historial.history['loss'], label='Pérdida de entrenamiento',color = 'pink', marker='o')
    plt.plot(historial.history['val_loss'], label='Pérdida de validación', color = 'green', marker='o')
    plt.title('Pérdida por epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.show()

    plt.plot(historial.history['accuracy'], label='Tasa de precisión de entrenamiento', color = 'pink', marker='o')
    plt.plot(historial.history['val_accuracy'], label='Tasa de precisión de validación', color = 'green', marker= 'o')
    plt.title('Tasa de precisión por epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Tasa de precisión')
    plt.show()
#clas_mlp()
