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
                # Extraer solo las variables numéricas (asumiendo que las columnas son los elementos químicos)
                # En producción, deberías definir explícitamente tu lista de 38 'elements'
                # Recuperar los nombres y el orden de las variables usadas
                # durante el entrenamiento del StandardScaler
                if hasattr(scaler, "feature_names_in_"):
                    elementos_esperados = list(scaler.feature_names_in_)
                else:
                    st.error(
                        "El scaler no conserva los nombres de las variables utilizadas "
                        "durante el entrenamiento."
                    )
                    st.stop()
                
                # Limpiar espacios accidentales en los encabezados del archivo
                df_input.columns = df_input.columns.astype(str).str.strip()
                
                # Identificar diferencias entre el archivo y el entrenamiento
                columnas_faltantes = [
                    columna
                    for columna in elementos_esperados
                    if columna not in df_input.columns
                ]
                
                columnas_adicionales = [
                    columna
                    for columna in df_input.select_dtypes(include=[np.number]).columns
                    if columna not in elementos_esperados
                ]
                
                # Detener el procesamiento si faltan elementos químicos
                if columnas_faltantes:
                    st.error(
                        "El archivo no contiene todos los elementos químicos requeridos."
                    )
                    st.write("Columnas faltantes:", columnas_faltantes)
                    st.stop()
                # Seleccionar exactamente las variables utilizadas durante el entrenamiento
                # y conservar el mismo orden
                datos_numericos = df_input[elementos_esperados].copy()
                
                # Convertir valores a formato numérico
                datos_numericos = datos_numericos.apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                
                # Comprobar valores vacíos o no numéricos
                if datos_numericos.isna().any().any():
                    columnas_con_problemas = datos_numericos.columns[
                        datos_numericos.isna().any()
                    ].tolist()
                
                    st.error(
                        "Existen valores vacíos o no numéricos en las variables químicas."
                    )
                    st.write("Columnas con problemas:", columnas_con_problemas)
                    st.stop()
                
                # Comprobar valores incompatibles con la transformación logarítmica
                if (datos_numericos + 1e-5 <= 0).any().any():
                    st.error(
                        "Existen concentraciones negativas incompatibles con log10(x + 1e-5)."
                    )
                    st.stop()
                
                # Aplicar el mismo preprocesamiento del entrenamiento
                datos_log = np.log10(datos_numericos + 1e-5)
                
                # Mantener nombres y orden antes de transformar
                datos_log = datos_log[elementos_esperados]
                
                datos_escalados = scaler.transform(datos_log)
                
                # Predicción del modelo
                probabilidades = modelo_nn.predict(datos_escalados)
                
                # Añadir los resultados al DataFrame original
                df_input['Probabilidad_Fertilidad'] = probabilidades.flatten()
                df_input['Clasificacion_IA'] = np.where(df_input['Probabilidad_Fertilidad'] > 0.5, 'Fértil', 'Estéril/Artefacto')
                
                st.success("¡Clasificación completada con éxito!")
                
                # Mostrar la tabla interactiva en la web
                st.dataframe(df_input.head(15))
                
                # =====================================================================
                # EXPORTACIÓN DE RESULTADOS
                # =====================================================================
                st.subheader("3. Descarga de Resultados")
                
                # Preparar el CSV en memoria para la descarga
                csv_buffer = df_input.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Descargar Base de Datos Clasificada (CSV)",
                    data=csv_buffer,
                    file_name="predicciones_fertilidad.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error de dimensionalidad: Asegúrate de que el archivo contenga los mismos elementos químicos del entrenamiento. Detalles: {e}")
