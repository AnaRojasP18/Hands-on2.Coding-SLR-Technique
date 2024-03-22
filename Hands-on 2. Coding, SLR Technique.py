class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

class DataSet:
    def __init__(self, puntos):
        self.puntos = puntos

    def media(self, eje):
        return sum(getattr(punto, eje) for punto in self.puntos) / len(self.puntos)

    def varianza(self, eje):
        media_eje = self.media(eje)
        return sum((getattr(punto, eje) - media_eje) ** 2 for punto in self.puntos) / len(self.puntos)

    def covarianza(self, eje_x, eje_y):
        media_x = self.media(eje_x)
        media_y = self.media(eje_y)
        return sum((getattr(punto, eje_x) - media_x) * (getattr(punto, eje_y) - media_y) for punto in self.puntos) / len(self.puntos)

class RegresionLineal:
    def __init__(self, data_set):
        self.data_set = data_set
        self._calcular_parametros()

    def _calcular_parametros(self):
        n = len(self.data_set.puntos)
        sum_x = sum(punto.x for punto in self.data_set.puntos)
        sum_y = sum(punto.y for punto in self.data_set.puntos)
        sum_xy = sum(punto.x * punto.y for punto in self.data_set.puntos)
        sum_x2 = sum(punto.x ** 2 for punto in self.data_set.puntos)

        self.beta_1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        self.beta_0 = (sum_y - self.beta_1 * sum_x) / n

    def predecir(self, x):
        return self.beta_0 + self.beta_1 * x

    def imprimir_ecuacion(self):
        print(f"Ecuación de regresión: y = {self.beta_0:.16f}+ {self.beta_1:.16f}x")

    def predecir_valor(self, x):
        y_predicho = self.predecir(x)
        print(f"Para x = {x}, el valor y predicho es: {y_predicho:.16f}")

    def calcular_coeficientes(self):
        sum_residuos2 = sum((punto.y - self.predecir(punto.x)) ** 2 for punto in self.data_set.puntos)
        sum_total2 = sum((punto.y - self.data_set.media("y")) ** 2 for punto in self.data_set.puntos)

        r2 = 1 - (sum_residuos2 / sum_total2)
        r = (r2)**0.5

        return r, r2

    def imprimir_coeficientes(self):
        r, r2 = self.calcular_coeficientes()
        print(f"Coeficiente de correlación (r): {r:.16f}")
        print(f"Coeficiente de determinación (r^2): {r2:.16f}")

data_set = DataSet([
    Punto(x, y) for x, y in [(23, 651), (26, 762), (30, 856), (34, 1063), (43, 1190), (48,1298), (52,1421), (57,1440), (58,1518)]
])

# Crear una nueva Regresion Lineal con el DataSet existente
regresion_existente = RegresionLineal(data_set)

# Imprimir la ecuación de regresión con los valores óptimos calculados
regresion_existente.imprimir_ecuacion()

# Imprimir los coeficientes de correlación y determinación
regresion_existente.imprimir_coeficientes()

# Realizar e imprimir la predicción para un valor X específico
regresion_existente.predecir_valor(18)
regresion_existente.predecir_valor(37)
regresion_existente.predecir_valor(60)
regresion_existente.predecir_valor(67)
regresion_existente.predecir_valor(80)
