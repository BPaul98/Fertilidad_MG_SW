import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import io

# =====================================================================
# CONFIGURACIÓN Y CARGA DE COMPONENTES
# =====================================================================
st.set_page_config(page_title="Geoquímica Predictiva", layout="wide")
st.title("Clasificador de Fertilidad Magmática multivariado")
st.markdown("Plataforma de procesamiento masivo para bases de datos litogeoquímicas.")

@st.cache_resource
def cargar_modelos():
    scaler = joblib.load('scaler_geoquimico.pkl')
    modelo = load_model('modelo_fertilidad.h5')
    return scaler, modelo

scaler, modelo_nn = cargar_modelos()

# =====================================================================
# INTERFAZ DE CARGA DE DATOS
# =====================================================================
st.subheader("1. Ingesta de Datos Analíticos")
archivo_subido = st.file_uploader("Sube tu matriz geoquímica (formato CSV o Excel)", type=['csv', 'xlsx'])

if archivo_subido is not None:
    # Leer el archivo dependiendo de su extensión
    if archivo_subido.name.endswith('.csv'):
        df_input = pd.read_csv(archivo_subido)
    else:
        df_input = pd.read_excel(archivo_subido)
        
    st.write(f"Archivo cargado exitosamente. Total de muestras detectadas: {len(df_input)}")
    
    # =====================================================================
# MOTOR DE INFERENCIA
# =====================================================================
    st.subheader("2. Procesamiento y Predicción")
    if st.button("Ejecutar Red Neuronal"):
        with st.spinner('Evaluando firmas multivariadas...'):
            try:
    # Obtener el orden de variables utilizado durante el entrenamiento
    if hasattr(scaler, "feature_names_in_"):
        columnas_modelo = list(scaler.feature_names_in_)
    else:
        st.error(
            "El scaler no conserva los nombres de las variables del entrenamiento. "
            "Debes declarar manualmente la lista columnas_modelo."
        )
        st.stop()

    # Limpiar espacios en los nombres de las columnas del archivo
    df.columns = df.columns.astype(str).str.strip()

    # Comprobar columnas faltantes
    columnas_faltantes = [
        columna
        for columna in columnas_modelo
        if columna not in df.columns
    ]

    if columnas_faltantes:
        st.error(
            "Faltan las siguientes columnas requeridas por el modelo: "
            + ", ".join(columnas_faltantes)
        )
        st.stop()

    # Seleccionar y ordenar exactamente como en el entrenamiento
    datos_modelo = df[columnas_modelo].copy()

    # Convertir las variables a formato numérico
    datos_modelo = datos_modelo.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Convertir infinitos en valores vacíos
    datos_modelo = datos_modelo.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Validar valores vacíos o no numéricos
    if datos_modelo.isna().any().any():
        columnas_invalidas = datos_modelo.columns[
            datos_modelo.isna().any()
        ].tolist()

        st.error(
            "Existen valores vacíos o no numéricos en: "
            + ", ".join(columnas_invalidas)
        )
        st.stop()

    # Corregir valores iguales o menores que cero antes de log10
    datos_corregidos = datos_modelo.copy()

    for columna in columnas_modelo:
        valores_positivos = datos_corregidos.loc[
            datos_corregidos[columna] > 0,
            columna
        ]

        if valores_positivos.empty:
            st.error(
                f"La columna '{columna}' no contiene valores positivos."
            )
            st.stop()

        # Sustituir <= 0 por la mitad del menor valor positivo
        valor_reemplazo = valores_positivos.min() / 2

        datos_corregidos.loc[
            datos_corregidos[columna] <= 0,
            columna
        ] = valor_reemplazo

    # Aplicar log10 sin ceros, negativos, infinitos ni NaN
    datos_log = np.log10(datos_corregidos)

    # Conservar nombres y orden después de la transformación
    datos_log = pd.DataFrame(
        datos_log,
        columns=columnas_modelo,
        index=datos_corregidos.index
    )

    # Escalar y predecir
    datos_escalados = scaler.transform(datos_log)
    predicciones = modelo_nn.predict(datos_escalados)

except Exception as e:
    st.error(f"Error durante el procesamiento: {e}")
