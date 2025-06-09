# Librerías necesarias
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import random

class AnalisisDescriptivo:
  def __init__(self, datos=None, x=None):
    if datos is not None:
      self.datos = np.array(datos) #Atributo
    if x is not None:
      self.x = x

  #Estimación de la función de densidad usando un histograma
  #La frecuencia relativa en cada bin se divide por su ancho para obtener una densidad
  def evalua_histograma(self, h):
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
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2)

  def kernel_uniforme(self, u):
    return (np.abs(u) <= 0.5).astype(float) #Devuelve 1(True) cuando u está dentro del intervalo (-1/2,1/2) y cero(False) cuando no lo está

  def cuadratico(self, u):
    return (3 / 4) * (1 - u ** 2) * ((np.abs(u) <= 1).astype(float))

  def triangular(self, u):
    return ((1 - np.abs(u)) * (np.abs(u) <= 1).astype(float))

  #Estima la densidad usando núcleos
  #Técnica suave y flexible que promedia funciones/núcleos centrados en cada punto de datos
  def densidad_nucleo(self, h, nucleo):
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

      densidad[i] = np.mean(valores_nucleo) / h
      return densidad

  #Error Cuadrático Medio entre la densidad estimada y la real
  #Cuanto más bajo, mejor es la estimación
  def margen_error_ECM(self, densidad_estimada, densidad_teorica):
    return float(np.mean((densidad_estimada - densidad_teorica) ** 2))

  #Compara cuantiles empíricos con los teóricos de una normal estándar
  def miqqplot(self):
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
  def __init__(self,N):
    self.N = N

  def datos_normal(self,a,b):
    return np.random.normal(a, b,self.N) #Función para generar datos con distribución normal

  def densidad_normal(self,x,a,b):
    return norm.pdf(x, loc=a, scale=b) #Teórica

  def densidad_BS(self,x):
    return 1/2 * self.densidad_normal(x,0,1) + 1/10 * sum(self.densidad_normal(x,j/2 - 1, 1/10) for j in range(5)) #Teórica

  def datos_BS(self):
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
  def __init__(self, x, y):
    self.x = np.array(x)
    self.y = np.array(y)

  def ajustar_modelo(self):
    x_const = sm.add_constant(self.x)
    modelo = sm.OLS(self.y, x_const)
    resultados = modelo.fit()
    return resultados

  def evalua_histograma(self, h, x):
    plt.hist(x, bins=h, alpha=0.7)
    plt.title("Histograma")
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    plt.show()

  def dividir_train_test(self, porcentaje_train=0.8, seed=10):
    n = self.x.shape[0]
    n_train = int(n * porcentaje_train)
    indices = list(range(n))
    random.seed(seed)
    train_indices = random.sample(indices, n_train)
    test_indices = list(set(indices) - set(train_indices))
    x_train = self.x[train_indices]
    y_train = self.y[train_indices]
    x_test = self.x[test_indices]
    y_test = self.y[test_indices]
    return x_train, y_train, x_test, y_test

  pass

class RegresionLineal(Regresion):

  def __init__(self, x, y):
    x = np.array(x)
    if x.ndim == 1:
      x = x.reshape(-1, 1)
    super().__init__(x, y)

  def predecir(self, new_x):
    new_x = np.array(new_x)
    if new_x.ndim == 1:
      new_x = new_x.reshape(-1, 1)
    new_x_const = sm.add_constant(new_x)
    resultados = self.ajustar_modelo()
    return resultados.predict(new_x_const)

  def graficar_recta_ajustada(self):
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
    resultados = self.ajustar_modelo()
    return {'betas': resultados.params,
        'errores_estandar': resultados.bse,
        't_obs': resultados.tvalues,
        'p_valores': resultados.pvalues}

  def calcular_residuos(self):
    resultados = self.ajustar_modelo()
    predichos = resultados.predict(sm.add_constant(self.x))
    residuos = self.y - predichos
    return residuos

  def graficar_residuos(self):
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
    resultados = self.ajustar_modelo()
    return resultados.conf_int(alpha=alpha)

  def calcular_r2(self):
    resultados = self.ajustar_modelo()
    return {'R2': resultados.rsquared, 'R2_ajustado': resultados.rsquared_adj}

  pass

class RegresionLogistica(Regresion):
  def ajustar_modelo(self):
    x_const = sm.add_constant(self.x)
    modelo = sm.Logit(self.y, x_const)
    resultados = modelo.fit()
    return resultados

  def predecir_probabilidades(self, new_x):
    new_x = np.array(new_x)
    if new_x.ndim == 1:
      new_x = new_x.reshape(-1, 1)
    new_x_const = sm.add_constant(new_x)
    resultados = self.ajustar_modelo()
    return resultados.predict(new_x_const)


  def evaluar_desempenio(self, x_test, y_test, umbral=0.5):
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
      'error_total': error_total} #Error de clasificaci

  def obtener_estadisticas(self):
    resultados = self.ajustar_modelo()
    return {'betas': resultados.params,
        'errores_estandar': resultados.bse,
        'z_obs': resultados.tvalues,
        'p_valores': resultados.pvalues}

  def graficar_curva_ROC(self, x_test, y_test):
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