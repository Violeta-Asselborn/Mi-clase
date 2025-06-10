"""

El módulo contiene clases para realizar ciertos análisis estadísticos como:
  -Estimacion de densidades mediante histogramas y núcleos
  -Generación y/o simulación de datos con distinta distribución
  -Regresión lineal
  -Regresión logística

Los nombres de las clases para cada funcionalidad son los siguientes:
  -AnalisisDescriptivo: Para estimar y visualizar densidades 
  -GeneradoraDeDatos: Para generar y/o simular datos
  -Regresion: Para armar modelos de regresión
  -RegresionLineal: Para aplicar regresión lineal simple y múltiple
  -RegresionLogistica: Para aplicar regresión logística 

"""


#Librerías necesarias
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import random

class AnalisisDescriptivo:
  """

  Clase para realizar estimaciones de densidades

  Atributos:
    - datos: Datos muestrales
    - x: Puntos donde evaluar la densidad

  """
  def __init__(self, datos=None, x=None):
    if datos is not None:
      self.datos = np.array(datos)
    if x is not None:
      self.x = x 

  def evalua_histograma(self, h):
    """

    Estima la función de densidad mediante histograma

    Argumentos:
      - h: Ancho de bins 

    Retorna: 
      - Valores de densidad estiamada en los puntos x

    """
    bins = np.arange(min(self.datos), max(self.datos) + h, h)

    frec_abs = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
      for j in range(len(self.datos)):
        if bins[i] <= self.datos[j] < bins[i+1]:
          frec_abs[i] += 1

    frec_rel = frec_abs / len(self.datos)
    densidad_estimada = frec_rel / h

    densidad_estimada_x = np.zeros(len(self.x))
    for i in range(len(self.x)):
      for j in range(len(bins) - 1):
        if bins[j] <= self.x[i] < bins[j + 1]:
          densidad_estimada_x[i] = densidad_estimada[j]

    return densidad_estimada_x 

  #Núcleos
  def kernel_gaussiano(self, u):
    return (1 / np.sqrt(2 * np.pi)) * np.e**(-1/2 * u**2)

  def kernel_uniforme(self, u):
    return (u > -1/2) & (u <= 1/2)

  def cuadratico(self, u):
    return (3 / 4) * (1 - u ** 2) * ((u >= -1) & (u <= 1))

  def triangular(self, u):
    return ((1 + u) * ((u >= -1) & (u < 0))) + ((1 - u) * ((u >= 0) & (u <= 1)))

  def densidad_nucleo(self, h, nucleo):
    """

    Estima la densidad usando núcleos con una técnica suave y flexible 
    que promedia núcleos centrados en cada punto de datos

    Argumentos:
      - h: Ancho de bins
      - nucleo: Tipo de núcleo ('uniforme', 'gaussiano', 'triangular' o 'cuadratico')

    Retorna:
      - Valores estimados de densidad

    """
    densidad = np.zeros(len(self.x))

    for i in range(len(self.x)):
      u = (self.datos - self.x[i]) / h #Estandariza la distancia entre cada dato (x[i]), al dividir ajusta la escala (h)

      if nucleo == 'uniforme':
        valores_nucleo = self.kernel_uniforme(u)
      elif nucleo == 'gaussiano':
        valores_nucleo = self.kernel_gaussiano(u)
      elif nucleo == 'triangular':
        valores_nucleo = self.triangular(u)
      else:
        valores_nucleo = self.cuadratico(u)

      densidad[i] = np.sum(valores_nucleo) / (len(self.datos) * h)
      return densidad 

  #Cuanto más bajo, mejor es la estimación
  def margen_error_ECM(self, densidad_estimada, densidad_teorica):
    """

    Calcula el Error Cuadrático Medio entre la densidad estimada y la real

    Retorna:
      - Valor del ECM

    """
    return float(np.mean((densidad_estimada - densidad_teorica) ** 2))

  def miqqplot(self):
    """

    Grafica la comparación de los cuantiles empíricos con los teóricos de una normal estándar 

    """
    #Estandarizando los datos
    media_x = np.mean(self.datos)
    desvio_x = np.std(self.datos)
    cuantiles_muestrales = (self.datos - media_x) / desvio_x
    #Ordenando los datos estandarizados
    cuantiles_muestrales = np.sort(cuantiles_muestrales)
    #Generando los cuantiles teóricos de la normal estándar
    n = len(self.datos)
    p = (np.arange(1, n + 1) -0.5)/ n
    cuantiles_teoricos = norm.ppf(p)
    #Graficando los cuantiles muestrales versus los cuantiles teóricos
    plt.scatter(cuantiles_teoricos, cuantiles_muestrales, color='blue', marker='o')
    plt.xlabel('Cuantiles teóricos')
    plt.ylabel('Cuantiles muestrales')
    plt.plot(cuantiles_teoricos,cuantiles_teoricos , linestyle='-', color='red')
    plt.show()
  pass

class GeneradoraDeDatos:
  """

  Genera datos simulados con distribución normal y BS(Bart Simpson)

  Atributos:
    - N: Tamaño de la muestra a generar

  """
  
  def __init__(self,N):
    self.N = N 

  def datos_normal(self,a,b):
    """

    Función para generar datos con distribución normal

    Argumentos:
      - a: Media de los datos a estimar
      - b: Desviación estándar de los datos a estimar

    Retorna:
      - Datos con distribucion normal de tamaño N

    """
    return np.random.normal(a, b,self.N) 

  def densidad_normal(self,x,a,b):
    """

    Calcula la densidad teórica de una distribucion normal

    Argumentos:
      - x: Datos 
      - a: Media de los datos a estimar
      - b: Desviación estándar de los datos a estimar

    Retorna:
      - Densidad teórica de una distribucion normal

    """
    return norm.pdf(x, loc=a, scale=b) 

  def densidad_BS(self,x):
    """

    Calcula la densidad teórica de una distribucion normal con una distribucion BS(Bart Simpson)

    Argumentos:
      - x: Datos

    Retorna:
      - Densidad teórica de una distribución BS

    """
    return 1/2 * self.densidad_normal(x,0,1) + 1/10 * sum(self.densidad_normal(x,j/2 - 1, 1/10) for j in range(5)) 
  
  def datos_BS(self):
    """

    Función para generar datos con BS(Bart Simpson)

    Retorna:
      - Datos con distribucion BS de tamaño N

    """
    u = np.random.uniform(size=(self.N,))
    datos_BS = u.copy()
    ind = np.where(u > 0.5)[0]
    datos_BS[ind] = np.random.normal(0, 1, size=len(ind))
    for j in range(5):
      ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
      datos_BS[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind)) #Función para generar datos con distribución BS
    return datos_BS

  pass

class Regresion:
  """

  Clase base para generar cada modelo de regresión

  Atributos:
    - x: Variables predictoras
    - y: Variable respuesta

  """
  def __init__(self, x, y):
    self.x = np.array(x) 
    self.y = np.array(y) 

  
  def ajustar_modelo(self):
    """

    Entrena el modelo con los datos

    Retorna:
      - Modelo ajustado de regresión lineal (OLS)

    """
    x_const = sm.add_constant(self.x)
    modelo = sm.OLS(self.y, x_const)
    resultados = modelo.fit()
    return resultados


  def evalua_histograma(self, h, x):
    """

    Grafica un histograma para las variables predictoras

    Argumentos:
      - h: Ancho de bins
      - x: Valores de variables predictoras

    """
    plt.hist(x, bins=h, alpha=0.7)
    plt.title("Histograma")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.show()

  def dividir_train_test(self,n, porcentaje_train=0.8, seed=None):
    """

    Divide los datos en dos conjutos, uno de entrenamiento y otro de prueba

    Argumentos:
      - n: Tamaño en que se desea dividir los conjutos
      - porcentaje_train: Porcentaje que se desea tomar de n
      - seed: Semilla

    Retorna:
      - x_train: Datos entrenamiento de variables predictoras 
      - y_train: Datos entrenamiento de variable respuesta 
      - x_test: Datos testeo de variables predictoras 
      - y_test: Datos testeo de variable respuesta 
    """
    if porcentaje_train == 1:
      n_train = int(n)

    else:
      n_train = int(n * porcentaje_train)
      indices = list(range(n))

    if seed is not None:
      random.seed(seed)
      train_indices = random.sample(indices, n_train)
      test_indices = list(set(indices) - set(train_indices))

      x_train = self.x[train_indices]
      y_train = self.y[train_indices]
      x_test = self.x[test_indices]
      y_test = self.y[test_indices]

    else:
      x_train = self.x[:n_train]
      y_train = self.y[:n_train]
      x_test = self.x[n_train:]
      y_test = self.y[n_train:]

    return x_train, y_train, x_test, y_test

  pass

class RegresionLineal(Regresion):
  """

  Aplica un modelo de regresión lineal simple o múltiple
  Hereda de la clase 'Regresion'

  Atributos:
    - x: Variables predictoras
    - y: Variable respuesta

  """
  
  def __init__(self, x, y):
    x = np.array(x)
    if x.ndim == 1:
      x = x.reshape(-1, 1)
    super().__init__(x, y)

  def predecir(self, new_x):
    """

    Realiza predicciones sobre un nuevo conjunto de datos especifico 

    Argumento:
      - new_x: Nuevo conjuto de datos

    Retorna:
      - Predicción de valores de respuesta para los nuevos datos

    """
    new_x = np.array(new_x)
    if new_x.ndim == 1:
      new_x = new_x.reshape(-1, 1)
    new_x_const = sm.add_constant(new_x)
    resultados = self.ajustar_modelo()
    return resultados.predict(new_x_const)

  
  def graficar_recta_ajustada(self):
    """

    Grafica la recta de regresion ajustada

    """
    if self.x.shape[1] == 1:
      resultados = self.ajustar_modelo()
      x_ordenado = np.sort(self.x, axis=0)
      y_pred = resultados.predict(sm.add_constant(x_ordenado))
      plt.scatter(self.x, self.y, label='Datos')
      plt.plot(x_ordenado, y_pred, color='red', label='Recta ajustada')
      plt.xlabel("x")
      plt.ylabel("y")
      plt.legend()
      plt.grid(True)
      plt.show()

  def obtener_estadisticas(self):
    """

    Ajusta el modelo entrenado previamente en Regresion 
    
    Retorna: 
      Un diccionario que contiene las claves y muestra:
      - betas
      - errores_estandar
      - t_obs
      - p_valores

    """
    resultados = self.ajustar_modelo()
    return {'betas': resultados.params,
        'errores_estandar': resultados.bse,
        't_obs': resultados.tvalues,
        'p_valores': resultados.pvalues}

  def calcular_residuos(self):
    """

    Calcula los residuos del modelo ajustado 

    Retorna:
      - Los residuos entre los valores predichos y observados

    
    """
    resultados = self.ajustar_modelo()
    predichos = resultados.predict(sm.add_constant(self.x))
    residuos = self.y - predichos
    return residuos

  def graficar_residuos(self):
    """

    Grafica los residuos versus los valores predichos

    """
    residuos = self.calcular_residuos()
    predichos = self.ajustar_modelo().predict(sm.add_constant(self.x))
    plt.scatter(predichos, residuos)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.title("Residuos vs Valores predichos")
    plt.grid(True)
    plt.show()

  
  def calcular_intervalos_confianza(self, alpha=0.05):
    """

    Calcula intervalos de confianza para los betas

    Argumento:
      - alpha: Valor del alfa, por defecto 0.05
    
    Retorna:
      - Intervalo de confianza

    """
    resultados = self.ajustar_modelo()
    return resultados.conf_int(alpha=alpha)

  def calcular_r2(self):
    """

    Retorna: 
      Un diccionario que contiene las claves y muestra:
      - R2
      - R2_ajustado

    """
    resultados = self.ajustar_modelo()
    return {'R2': resultados.rsquared, 'R2_ajustado': resultados.rsquared_adj}

  pass


class RegresionLogistica(Regresion):
  """

  Modelo de regresión logística
  Hereda de la clase 'Regresion' y utiliza sus atributos

  """
  def ajustar_modelo(self):
    """

    Ajusta el modelo de regresión logística

    Retorna:
      - Modelo ajustado de regresión logística (Logit)

    """
    x_const = sm.add_constant(self.x)
    modelo = sm.Logit(self.y, x_const)
    resultados = modelo.fit()
    return resultados

  def predecir_probabilidades(self, new_x):
    """

    Estima las probabilidadades de un conjuto de datos especifico de pertenecer a la clase 1

    Argumento:
      - new_x: Nuevo conjunto de datos
    
    Retorna:
      - Predicción de la probabilidad de que pertenecezca a la clase 1

    """
    new_x = np.array(new_x)
    if new_x.ndim == 1:
      new_x = new_x.reshape(-1, 1)
    new_x_const = sm.add_constant(new_x)
    resultados = self.ajustar_modelo()
    return resultados.predict(new_x_const)
  
  def predecir_clases(self, new_x, umbral=0.5):
    probabilidades = self.predecir_probabilidades(new_x)
    return (probabilidades >= umbral).astype(int)

  def evaluar_desempenio(self, x_test, y_test, umbral=0.5):
    """
    Evalua el desempeño del modelo de regresión logística

    Argumentos:
      - x_test: Datos testeo de variables predictoras 
      - y_test: Datos testeo de variable respuesta
      - umbral: Por defecto 0.5

    Retorna:
      Un diccionario que contiene las claves y muestra: 
      - matriz_confusion
      - sensibilidad
      - especificidad
      - error_total

    """
    y_pred = self.predecir_clases(x_test, umbral)
    VP = np.sum((y_test == 1) & (y_pred == 1))
    VN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))
    sensibilidad = VP / (VP + FN)
    especificidad = VN / (VN + FP)
    error_total = (FP + FN) / len(y_test)
    matriz_confusion = np.array([[VP, FP], [FN, VN]])
    return {'matriz_confusion': matriz_confusion,
      'sensibilidad': sensibilidad,
      'especificidad': especificidad,
      'error_total': error_total} 

  def obtener_estadisticas(self):
    """

    Ajusta el modelo entrenado previamente en 'ajustar_modelo'

    Retorna:
      Un diccionario que contiene las claves y muestra:
      - betas
      - errores_estandar
      - z_obs
      - p_valores

    """
    resultados = self.ajustar_modelo()
    return {'betas': resultados.params,
        'errores_estandar': resultados.bse,
        'z_obs': resultados.tvalues,
        'p_valores': resultados.pvalues}

  def graficar_curva_ROC(self, x_test, y_test):
    """

    Grafica la curva ROC y calcula el AUC(Área Bajo la Curva) 
    Representa la capacidad discriminativa del modelo 

    Argumentos:
      - x_test: Datos testeo de variables predictoras 
      - y_test: Datos testeo de variable respuesta

    Retorna:
      - AUC
    """
    from sklearn.metrics import roc_curve, auc
    probs = self.predecir_probabilidades(x_test)
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.show()
    return auc_score

  pass
