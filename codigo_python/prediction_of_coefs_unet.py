import os
import numpy as np # type: ignore
import scipy.io
from scipy.io import loadmat # type: ignore
import tensorflow as tf # Para red neuronal profunda
import numpy as np
import matplotlib.pyplot as plt
import time # Para tomar el tiempo de entrenamiento de la red
import math

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Definir la metrica NMSE ajustada para utilizar la varianza de los valores verdaderos
def nmse(y_true, y_pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
    var_true = tf.keras.backend.var(y_true)
    return mse / var_true

persona = 'SANOS/1_HEMU'
sector = 'derecho'
lado = ''

if sector == 'derecho':
    lado = 'vscd'
else:
    lado = 'vsci'

#CARGAR UN MODELO ESPECÍFICO. OBJETIVO: PREDECIR COEFICIENTES DE LA SENAL VSC ORIGINAL PARA ANALIZAR CALIDAD DE LA ESTIMACION DE LA RED
modelo_cargado = tf.keras.models.load_model('D:/TT/Memoria/Ubuntu_Windows_Sharing/data/signals_LDS/' + persona + '/unet_model_' + lado + '_' + sector + '.keras',
                                           custom_objects={'nmse': nmse})

#CARGAR COEFICIENTES (MATRIZ COMPLEJA) DE LA SEÑALES PAM ORIGINALES T OBTENER CANTIDAD DE MATRICES COMPLEJAS ENCONTRADAS
input_matrix_complex_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona +  '/PAMoriginal_matrixcomplex' # coeficientes de senal PAM original (sin ruido) en formato .mat
output_matrix_complex_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona +  '/PAMoriginal_matrixcomplex' # coeficientes de senal PAM original (sin ruido) en formato .npy
os.makedirs(output_matrix_complex_pam_dir, exist_ok=True) # crear directorio "output_matrix_complex_pam_dir" si no existe

# Se leen la cantidad de coeficientes o matrices complejas de senales PAM encontradas en el directorio "input_matrix_complex_pam_dir"
total_matrix_complex_pam = sum(1 for filename in os.listdir(input_matrix_complex_pam_dir) if filename.endswith('.mat')) + 1
print("Total de matrices complejas PAM encontradas -> ", total_matrix_complex_pam - 1)


#TRANSFORMAR A FORMATO .npy LAS MATRICES COMPLEJAS ASOCIADAS A LA SEÑAL ORIGINAL PAM
# Funcion para convertir una matriz compleja de .mat a .npy
def convert_mat_to_npy_original_signal(input_dir, output_dir, prefix):
    mat_file = os.path.join(input_dir, f'{prefix}.mat')
    npy_file = os.path.join(output_dir, f'{prefix}_npy.npy')
    
    # Cargar el archivo .mat
    mat_data = loadmat(mat_file)
    
    # Extraer la matriz compleja
    matrix_key = [key for key in mat_data.keys() if not key.startswith('__')][0]
    matrix = mat_data[matrix_key]
    
    # Guardar la matriz en formato .npy
    np.save(npy_file, matrix)


# Convertir la matriz compleja de la senal PAM original de formato .mat a .npy
convert_mat_to_npy_original_signal(input_matrix_complex_pam_dir, output_matrix_complex_pam_dir, 'matrix_complex_original_pam')

#PREDECIR COEFICIENTES DE UNA SEÑAL DE VSC A PARTIR DE COEFICIENTES DE UNA SEÑAL PAM
# Funcion para predecir con el modelo entrenado
def predecir_coefs(modelo_cargado, input_data):
    coefs_predicted = modelo_cargado.predict(input_data)
    return coefs_predicted

# Listar todos los archivos en el directorio
archivos_npy_dir = os.listdir(output_matrix_complex_pam_dir)
# Filtrar solo el archivo .npy
archivos_npy = [f for f in archivos_npy_dir if f.endswith('.npy')]

# Cargar una matriz de entrada para hacer una prediccion 
# Leer el primer archivo .npy
nombre_archivo_pam_npy = archivos_npy[0]
archivo_pam_npy_dir = os.path.join(output_matrix_complex_pam_dir, nombre_archivo_pam_npy) # ELEGIR ARCHIVO NPY A LEER 

#######################
# ENTRADA PARA LA RED:
######################
input_matrix_pam = np.load(archivo_pam_npy_dir)
print("Archivo cargado:", archivo_pam_npy_dir)
print("formato matrix complex input: ",input_matrix_pam.shape)

tensor_input_matrix_pam = np.stack((input_matrix_pam.real, input_matrix_pam.imag), axis=-1)
print("formato matrix complex input como tensor: ",tensor_input_matrix_pam.shape)

# Expandir dimensiones para que coincidan con la forma esperada por el modelo
tensor_input_matrix_pam = np.expand_dims(tensor_input_matrix_pam, axis=0)
print("Formato matrix complex input con dimensión adicional:", tensor_input_matrix_pam.shape)



#REALIZAR PREDICCIÓN (OBTENCIÓN DE COEFICIENTES DE SEÑAL VSC ESTIMADA)
# Realizar la predicción
predicted_output = predecir_coefs(modelo_cargado, tensor_input_matrix_pam)

# Mostrar la predicción
print("Prediccion de la primera muestra de entrada:")
#print(predicted_output)
print("Formato de los coeficientes de la senal VSC estimada: ", predicted_output.shape)

#TRANSFORMAR LA SALIDA ESTIMADA A UN FORMATO (36, 1024) Y LUEGO DE .npy a .mat
# Transformar la matriz tensor VSC a una matriz compleja de formato .mat
# El tensor tiene la forma (1, 36, 1024, 2) y necesitamos transformarlo a (36, 1024) a matriz compleja
complex_matrix_vsc = predicted_output[0, :, :, 0] + 1j * predicted_output[0, :, :, 1]
print("Formato nuevo:",complex_matrix_vsc.shape)

#GUARDADO DE LA MATRIZ COMPLEJA DE LA VSC ESTIMADA EN UNA CARPETA EN EL DIRECTORIO DE MATLAB EN FORMATO .mat

# Nombre que tendra el archivo de la matriz compleja d la VSC estimada .mat. Se concatena el numero del archivo estimado
lado_coefs = ''
#mat_file = ''
mat_dir = ''
if sector == 'derecho':
    lado_coefs = 'VSCd'
    # Directorio donde se guardara la matriz compleja de la VSC estimada en formato .mat
    mat_dir = 'D:/TT/Memoria/MemoriaCodigofuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/CoefficientsPredicted_' + lado_coefs
    os.makedirs(mat_dir, exist_ok=True) # crear directorio "mat_dir" si no existe
    mat_filename = f'matrix_complex_vscd_predicted.mat'
else:
    lado_coefs = 'VSCi'
    # Directorio donde se guardara la matriz compleja de la VSC estimada en formato .mat
    mat_dir = 'D:/TT/Memoria/MemoriaCodigofuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/CoefficientsPredicted_' + lado_coefs
    os.makedirs(mat_dir, exist_ok=True) # crear directorio "mat_dir" si no existe
    mat_filename = f'matrix_complex_vsci_predicted.mat'
print(mat_filename)

# Ruta completa del archivo .mat
mat_path = os.path.join(mat_dir, mat_filename)


try:
    data = scipy.io.loadmat(mat_path)
    print("El archivo es un archivo .mat válido.")
    print(data.keys())
except Exception as e:
    print(f"Error al cargar el archivo .mat: {e}")


# Guardar la matriz compleja en un archivo .mat {nombre archivo: contenido archivo}
scipy.io.savemat(mat_path, {mat_filename: complex_matrix_vsc})

print(f"Archivo guardado en: {mat_path}")