# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:17:21 2025

@author: jperezr
"""

import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.markdown("**Universidad Panamericana**")  
st.sidebar.markdown("**Aprendizaje Máquina Aplicado**")  
st.sidebar.markdown("**Profesor:** Omar Velázquez López")  
st.sidebar.markdown("**Estudiante:** Javier Horacio Pérez Ricárdez")  
st.sidebar.markdown("**26 de marzo del 2025**")


# Título de la aplicación
st.title("Modelo de Regresión Logística: Predicción de la Calidad del Vino Tinto")

# Inicializar el estado de sesión para controlar el progreso
if "step" not in st.session_state:
    st.session_state.step = 1

# ----------------------------
# 1. Mostrar librerías importadas
# ----------------------------
if st.session_state.step >= 1:
    st.header("1. Librerías Importadas")
    st.code("""
    import streamlit as st
    import pandas as pd
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    """, language="python")

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_1"):
        st.session_state.step = 2

# ----------------------------
# 2. Mostrar código para cargar el archivo
# ----------------------------
if st.session_state.step >= 2:
    st.header("2. Código para Cargar el Archivo")
    st.code("""
    file_name = "winequality-red.xlsx"

    # Verificar si el archivo existe
    if os.path.exists(file_name):
        try:
            # Cargar el archivo Excel
            df = pd.read_excel(file_name)  # Cargar el archivo .xlsx

            # Mostrar el DataFrame en Streamlit con un formato mejorado
            st.subheader("Datos del archivo 'winequality-red.xlsx'")
            st.dataframe(df)  # Usamos st.dataframe para una visualización interactiva

            # Mostrar estadísticas descriptivas
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())  # Usamos st.dataframe para mejorar la visualización
        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo: {e}")
    else:
        st.error(f"El archivo '{file_name}' no se encontró en el directorio actual.")
        st.info("Por favor, asegúrate de que el archivo 'winequality-red.xlsx' esté en el mismo directorio que este script.")
    """, language="python")

    # Botón para ejecutar la carga del archivo
    if st.button("Ejecutar carga del archivo", key="button_2"):
        st.session_state.step = 3

# ----------------------------
# 3. Ejecutar la carga del archivo
# ----------------------------
if st.session_state.step >= 3:
    st.header("3. Ejecución: Carga del Archivo")
    file_name = "winequality-red.xlsx"

    # Verificar si el archivo existe
    if os.path.exists(file_name):
        try:
            # Cargar el archivo Excel
            df = pd.read_excel(file_name)  # Cargar el archivo .xlsx

            # Mostrar el DataFrame en Streamlit con un formato mejorado
            st.subheader("Datos del archivo 'winequality-red.xlsx'")
            st.dataframe(df)  # Usamos st.dataframe para una visualización interactiva

            # Mostrar estadísticas descriptivas
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe())  # Usamos st.dataframe para mejorar la visualización

            st.write("**Resumen:**")
            st.write("- Se cargaron los datos del archivo `winequality-red.xlsx`.")
            st.write("- Se mostraron las primeras filas y estadísticas descriptivas.")

            # Guardar el DataFrame en el estado de sesión para usarlo en pasos posteriores
            st.session_state.df = df

        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo: {e}")
    else:
        st.error(f"El archivo '{file_name}' no se encontró en el directorio actual.")
        st.info("Por favor, asegúrate de que el archivo 'winequality-red.xlsx' esté en el mismo directorio que este script.")

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_3"):
        st.session_state.step = 4

# ----------------------------
# 4. Mostrar código para preprocesamiento de datos
# ----------------------------
if st.session_state.step >= 4:
    st.header("4. Código para Preprocesamiento de Datos")
    st.code("""
    # Separar características y variable objetivo
    X = df.drop("quality", axis=1)  # Características (todas las columnas excepto 'quality')
    y = df["quality"]  # Variable objetivo ('quality')

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalización de los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    """, language="python")

    # Botón para ejecutar el preprocesamiento
    if st.button("Ejecutar preprocesamiento de datos", key="button_4"):
        st.session_state.step = 5

# ----------------------------
# 5. Ejecutar el preprocesamiento de datos
# ----------------------------
if st.session_state.step >= 5:
    st.header("5. Ejecución: Preprocesamiento de Datos")

    # Obtener el DataFrame del estado de sesión
    df = st.session_state.df

    # Separar características y variable objetivo
    X = df.drop("quality", axis=1)  # Características (todas las columnas excepto 'quality')
    y = df["quality"]  # Variable objetivo ('quality')

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalización de los datos
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    # Mostrar los conjuntos de entrenamiento y prueba
    st.subheader("Conjunto de Entrenamiento (80%)")
    st.write("**Características (X_train):**")
    st.dataframe(X_train)  # DataFrame de características de entrenamiento
    st.write("**Variable objetivo (y_train):**")
    st.dataframe(y_train)  # DataFrame de la variable objetivo de entrenamiento

    st.subheader("Conjunto de Prueba (20%)")
    st.write("**Características (X_test):**")
    st.dataframe(X_test)  # DataFrame de características de prueba
    st.write("**Variable objetivo (y_test):**")
    st.dataframe(y_test)  # DataFrame de la variable objetivo de prueba

    # Mostrar las características normalizadas
    st.subheader("Características Normalizadas")
    st.write("**Conjunto de Entrenamiento Normalizado (X_train_normalized):**")
    st.dataframe(pd.DataFrame(X_train_normalized, columns=X.columns))  # DataFrame de características normalizadas de entrenamiento
    st.write("**Conjunto de Prueba Normalizado (X_test_normalized):**")
    st.dataframe(pd.DataFrame(X_test_normalized, columns=X.columns))  # DataFrame de características normalizadas de prueba

    st.write("**Resumen:**")
    st.write("- Se separaron las características (`X`) y la variable objetivo (`y`).")
    st.write("- Los datos se dividieron en conjuntos de entrenamiento (80%) y prueba (20%).")
    st.write("- Las características se normalizaron para que tengan la misma escala.")

    # Guardar los datos preprocesados en el estado de sesión
    st.session_state.X_train = X_train_normalized
    st.session_state.X_test = X_test_normalized
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_5"):
        st.session_state.step = 6

# ----------------------------
# 6. Mostrar código para definir el modelo
# ----------------------------
if st.session_state.step >= 6:
    st.header("6. Código para Definir el Modelo")
    st.code("""
    # Definición del modelo de Regresión Logística
    model = LogisticRegression(max_iter=1000)  # Se aumenta max_iter para asegurar la convergencia
    """, language="python")

    # Botón para ejecutar la definición del modelo
    if st.button("Ejecutar definición del modelo", key="button_6"):
        st.session_state.step = 7

# ----------------------------
# 7. Ejecutar la definición del modelo
# ----------------------------
if st.session_state.step >= 7:
    st.header("7. Ejecución: Definición del Modelo")

    # Definición del modelo de Regresión Logística
    model = LogisticRegression(max_iter=1000)  # Se aumenta max_iter para asegurar la convergencia
    st.write("El modelo de Regresión Logística se define con los siguientes parámetros:")
    st.write(model)

    st.write("**Resumen:**")
    st.write("- Se definió un modelo de Regresión Logística utilizando `LogisticRegression` de Scikit-learn.")
    st.write("- Se aumentó `max_iter` a 1000 para asegurar la convergencia del modelo.")

    # Guardar el modelo en el estado de sesión
    st.session_state.model = model

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_7"):
        st.session_state.step = 8

# ----------------------------
# 8. Mostrar código para entrenar el modelo
# ----------------------------
if st.session_state.step >= 8:
    st.header("8. Código para Entrenar el Modelo")
    st.code("""
    # Entrenamiento del modelo
    model.fit(X_train, y_train)
    """, language="python")

    # Botón para ejecutar el entrenamiento del modelo
    if st.button("Ejecutar entrenamiento del modelo", key="button_8"):
        st.session_state.step = 9

# ----------------------------
# 9. Ejecutar el entrenamiento del modelo
# ----------------------------
if st.session_state.step >= 9:
    st.header("9. Ejecución: Entrenamiento del Modelo")

    # Obtener el modelo del estado de sesión
    model = st.session_state.model

    # Entrenamiento del modelo
    model.fit(st.session_state.X_train, st.session_state.y_train)
    st.write("El modelo ha sido entrenado con los datos de entrenamiento.")

    st.write("**Resumen:**")
    st.write("- El modelo se entrenó utilizando los datos de entrenamiento (`X_train` y `y_train`).")
    st.write("- El método `fit()` ajustó los coeficientes del modelo para minimizar la función de costo.")

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_9"):
        st.session_state.step = 10

# ----------------------------
# 10. Mostrar código para evaluar el modelo
# ----------------------------
if st.session_state.step >= 10:
    st.header("10. Código para Evaluar el Modelo")
    st.code("""
    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Mostrar métricas
    st.subheader("Métricas de Evaluación")
    st.write(f"Exactitud: {accuracy:.2f}")
    st.write(f"Precisión: {precision:.2f}")
    st.write(f"Exhaustividad: {recall:.2f}")
    st.write(f"Medida F1: {f1:.2f}")

    # Mostrar matriz de confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=range(3, 9), columns=range(3, 9))
    st.dataframe(cm_df)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)
    """, language="python")

    # Botón para ejecutar la evaluación del modelo
    if st.button("Ejecutar evaluación del modelo", key="button_10"):
        st.session_state.step = 11

# ----------------------------
# 11. Ejecutar la evaluación del modelo
# ----------------------------
if st.session_state.step >= 11:
    st.header("11. Ejecución: Evaluación del Modelo")

    # Obtener el modelo y los datos del estado de sesión
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Mostrar métricas
    st.subheader("Métricas de Evaluación")
    st.write(f"Exactitud: {accuracy:.2f}")
    st.write(f"Precisión: {precision:.2f}")
    st.write(f"Exhaustividad: {recall:.2f}")
    st.write(f"Medida F1: {f1:.2f}")

    # Mostrar matriz de confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=range(3, 9), columns=range(3, 9))
    st.dataframe(cm_df)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    st.write("**Resumen:**")
    st.write("- Se evaluó el modelo utilizando el conjunto de prueba (`X_test` y `y_test`).")
    st.write("- Se calcularon métricas como exactitud, precisión, exhaustividad y medida F1.")
    st.write("- Se visualizó la matriz de confusión para analizar las predicciones correctas e incorrectas.")

    # Botón para continuar
    if st.button("Mostrar siguiente paso", key="button_11"):
        st.session_state.step = 12

# ----------------------------
# 12. Mostrar código para predicción
# ----------------------------
if st.session_state.step >= 12:
    st.header("12. Código para Predicción")
    st.code("""
    # Predicción en nuevos datos (ejemplo)
    nuevos_datos = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]  # Ejemplo de nuevos datos
    nuevos_datos_escalados = scaler.transform(nuevos_datos)  # Escalar los nuevos datos
    prediccion = model.predict(nuevos_datos_escalados)
    st.write(f"Predicción de calidad para los nuevos datos: {prediccion[0]}")
    """, language="python")

    # Botón para ejecutar la predicción
    if st.button("Ejecutar predicción", key="button_12"):
        st.session_state.step = 13

# ----------------------------
# 13. Ejecutar la predicción
# ----------------------------
if st.session_state.step >= 13:
    st.header("13. Ejecución: Predicción")

    # Obtener el modelo y el scaler del estado de sesión
    model = st.session_state.model
    scaler = st.session_state.scaler

    # Predicción en nuevos datos (ejemplo)
    st.subheader("Predicción en Nuevos Datos")
    nuevos_datos = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]  # Ejemplo de nuevos datos
    nuevos_datos_escalados = scaler.transform(nuevos_datos)  # Escalar los nuevos datos
    prediccion = model.predict(nuevos_datos_escalados)
    st.write(f"Predicción de calidad para los nuevos datos: {prediccion[0]}")

    st.write("**Resumen:**")
    st.write("- Se simularon nuevos datos para predecir la calidad del vino.")
    st.write("- Los nuevos datos se escalaron utilizando el mismo `scaler` que se usó en el entrenamiento.")
    st.write("- El modelo predijo la calidad del vino para los nuevos datos.")
