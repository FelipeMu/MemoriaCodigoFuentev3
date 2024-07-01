
"""
**Librerias para trabajar con la red neuronal y procesamiento de datos**
"""
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


"""
**CONVERTIR MATRICES DE: .mat  --> .npy**
"""


"""
Directorios de entrada y salida: ***SUJETOS SANOS Y PACIENTES TEC***

**(1) sujetos sanos:** 1_HEMU - 2_DAOC - 3_DASI - 4_DABA - 5_HEFU - 6_JOBO - 7_ROMI - 8_FEGA - 9_GAGO - 10_MIMO - 11_JULE - 12_NIGA - 13_BYLA - 
14_ARVA - 15_CLSE - 16_PAAR - 17_VATO - 18_FEBE - 19_VINA - 20_CLHE - 21_MAIN - 22_ALSA - 23_MIRA - 24_LACA - 25_GOAC - 26_ANGL - 27_HC036101
___________________________________________________________________________________________________________________________________________________________________________________

**(2) pacientes tec:** 1_DENI1005 - 2_KNOW1001 - 3_ALI0 - 4_BUTL - 5_HAGG - 6_HASTI007 - 7_BOAM - 8_DANE0005 - 9_GREG - 10_AITK - 11_RANS0000 - 12_JONES004 - 13_PERR 
- 14_SLAC - 15_HEPPL010 - 16_RICHS010 - 17_KENT0007 - 18_STAN1002 - 19_MCDON022 - 20_PULL - 21_MORR1002 - 22_PARK - 23_HIGH - 24_NOBL - 25_COWL - 26_KHAN - 27_NOLA
"""





# Directorios de las matrices complejas (coeficientes) de las senales PAM ,VSCd y VSCi de suejtos sanos o pacientes con tec. Se debe modificar estos directorios para ir
# generando los respectivos modelos de cada individuo.

# DIRECTORIOS: SUJETO SANO

# SUJETO SANO O PACIENTE TEC POR ANALIZAR
persona = '/1_HEMU'

 # INPUT PARA LA RED
# INPUT PAM
input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_mat'
# INPUT VSCd
input_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_mat'
# INPUT VSCI
input_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_mat'

# OUTPUT O SALIDAS ESPERADAS PARA LA RED
# OUTPUT PAM
output_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_npy'
# OUTPUT PAM
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_npy'
# OUTPUT PAM
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_npy'


#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################


# DIRECTORIOS: PACIENTE TEC
#paciente_tec = '/1_DENI1005'


 # INPUT PARA LA RED
# INPUT PAM
#input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/PAMnoises_matrixcomplex_mat'
# INPUT VSCd
#input_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/VSCdnoises_matrixcomplex_mat'
# INPUT VSCI
#input_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/VSCinoises_matrixcomplex_mat'

# OUTPUT O SALIDAS ESPERADAS PARA LA RED
# OUTPUT PAM
#output_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/PAMnoises_matrixcomplex_npy'
# OUTPUT PAM
#output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/VSCdnoises_matrixcomplex_npy'
# OUTPUT PAM
#output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/TEC' + persona + '/VSCinoises_matrixcomplex_npy'


print(input_pam_dir)
print(input_vscd_dir)
print(input_vsci_dir)


"""
Crear los directorios de salida si no existen
"""


os.makedirs(output_pam_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE PAM EN FORMATO .npy
os.makedirs(output_vscd_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCd EN FORMATO .npy
os.makedirs(output_vsci_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCi EN FORMATO .npy


"""
Funcion para convertir archivos .mat a .npy
"""


total_files=sum(1 for filename in os.listdir(input_pam_dir) if filename.endswith('.mat')) + 1
print("Total de archivos a analizar -> ",total_files-1)


def convert_mat_to_npy(input_dir, output_dir, prefix):
    for i in range(1, total_files):
        mat_file = os.path.join(input_dir, f'{prefix}_noise_{i}.mat')
        npy_file = os.path.join(output_dir, f'{prefix}_noise_{i}.npy')
        
        # Cargar el archivo .mat
        mat_data = loadmat(mat_file)
        
        # Extraer la matriz compleja
        matrix_key = [key for key in mat_data.keys() if not key.startswith('__')][0]
        matrix = mat_data[matrix_key]
        
        # Guardar la matriz en formato .npy
        np.save(npy_file, matrix)


"""
Convertir archivos .mat a .npy para PAM y VSC
"""


convert_mat_to_npy(input_pam_dir, output_pam_dir, 'matrix_complex_pam')
convert_mat_to_npy(input_vscd_dir, output_vscd_dir, 'matrix_complex_vscd')
convert_mat_to_npy(input_vsci_dir, output_vsci_dir, 'matrix_complex_vsci')


"""
**Conversion a tensor tridimensional** (Estructura adecuada para entrenar la red U-net)
"""


"""
Directorios salida para matrices con estructura tensor tridimensional
"""


input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_npy'
input_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_npy'
input_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_npy'

output_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'


"""
Crear directorios de salida si no existen (matrices complejas en forma de tensor tridimensional)
"""


os.makedirs(output_pam_dir, exist_ok=True)
os.makedirs(output_vscd_dir, exist_ok=True)
os.makedirs(output_vsci_dir, exist_ok=True)


"""
**PROCESAR MATRICES SEGÚN UNA NORMALIZACIÓN Y ORGANIZAR DATOS PARA LA RED**
"""



#Entrada: matriz_compleja (array bidimensional)
#Salida: datos_organizados (tensor de datos de la matriz pam)
#Descripcion: funcion encargada de normalizar cada matriz pam y definir el input para la red u-net
def procesar_matriz_compleja_pam(matriz_compleja):
    
    # Aplicacion de normalziacion Z-CORE || z = (x - u) / desv
    #norm_real, norm_imag = normalizacion_pam(matriz_compleja)

    # Aplicacion de normalziacion MIN-MAX || Ni = (Xi - Xmin) / (Xmax - Xmin)
    #norm_real, norm_imag = normalizacion_minmax_pam(matriz_compleja)

    # Crear input adecuado
    #datos_organizados = np.stack((norm_real, norm_imag), axis=-1)
    datos_organizados = np.stack((matriz_compleja.real, matriz_compleja.imag), axis=-1)
    return datos_organizados


#Entrada: matriz_compleja (array bidimensional)
#Salida: datos_organizados (tensor de datos de la matriz vsc)
#Descripcion: funcion encargada de normalizar cada matriz vsc y definir el input para la red u-net
def procesar_matriz_compleja_vsc(matriz_compleja):
    
    # Aplicacion de normalziacion Z-CORE || z=(x - u)/desv
    #norm_real, norm_imag = normalizacion_vsc(matriz_compleja)

    # Aplicacion de normalziacion MIN-MAX || Ni = (Xi-Xmin)/(Xmax-Xmin)
    #norm_real, norm_imag = normalizacion_minmax_vsc(matriz_compleja)
    
    # Crear input adecuado
    #datos_organizados = np.stack((norm_real, norm_imag), axis=-1)
    datos_organizados = np.stack((matriz_compleja.real, matriz_compleja.imag), axis=-1)
    return datos_organizados


"""
**PROCESAR MATRICES COMPLEJAS EN LA CARPETA input_pam_dir y procesar matrices mediante una normalización**
"""


for filename in os.listdir(input_pam_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_pam_dir, filename)
        output_path = os.path.join(output_pam_dir, filename)
        
        # Cargar la matriz compleja
        matriz_compleja = np.load(input_path)
        
        # Procesar la matriz compleja segun una normalizacion
        datos_organizados = procesar_matriz_compleja_pam(matriz_compleja)
        
        # Guardar los datos procesados
        np.save(output_path, datos_organizados)




"""
**PROCESAR MATRICES COMPLEJAS EN LA CARPETA input_vscd_dir**
"""


for filename in os.listdir(input_vscd_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_vscd_dir, filename)
        output_path = os.path.join(output_vscd_dir, filename)
        
        # Cargar la matriz compleja
        matriz_compleja = np.load(input_path)
        
        # Procesar la matriz compleja segun una normalizacion
        datos_organizados = procesar_matriz_compleja_vsc(matriz_compleja)
        
        # Guardar los datos procesados
        np.save(output_path, datos_organizados)
        
print("Procesamiento completado.")


"""
**PROCESAR MATRICES COMPLEJAS EN LA CARPETA input_vsci_dir**
"""


for filename in os.listdir(input_vsci_dir):
    if filename.endswith('.npy'):
        input_path = os.path.join(input_vsci_dir, filename)
        output_path = os.path.join(output_vsci_dir, filename)
        
        # Cargar la matriz compleja
        matriz_compleja = np.load(input_path)
        
        # Procesar la matriz compleja segun una normalizacion
        datos_organizados = procesar_matriz_compleja_vsc(matriz_compleja)
        
        # Guardar los datos procesados
        np.save(output_path, datos_organizados)
        
print("Procesamiento completado.")


"""
**Verificacion de "shape" - matrices pam y vsc**
"""


"""
Directorios de salida a verificar
"""


output_pam_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
output_vscd_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
output_vsci_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'


"""
Funcion para verificar la forma de una matriz
"""


def verificar_shape(directorio, nombre_archivo):
    path = os.path.join(directorio, nombre_archivo)
    matriz = np.load(path)
    return matriz.shape


"""
Verificar la forma de un archivo de ejemplo en output_pam_dir_check
"""


ejemplo_pam = os.listdir(output_pam_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_pam = verificar_shape(output_pam_dir_check, ejemplo_pam)
print(f"Shape de {ejemplo_pam} en {output_pam_dir_check}: {shape_pam}")


"""
Verificar la forma de un archivo de ejemplo en output_vscd_dir_check
"""


ejemplo_vscd = os.listdir(output_vscd_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_vscd = verificar_shape(output_vscd_dir_check, ejemplo_vscd)
print(f"Shape de {ejemplo_vscd} en {output_vscd_dir_check}: {shape_vscd}")


"""
Verificar la forma de un archivo de ejemplo en output_vsci_dir_check
"""


ejemplo_vsci = os.listdir(output_vsci_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_vsci = verificar_shape(output_vsci_dir_check, ejemplo_vsci)
print(f"Shape de {ejemplo_vsci} en {output_vsci_dir_check}: {shape_vsci}")


#####################################################################################################################################################
##############################################          **|||Red Neuronal Profunda: U-net|||**            ###########################################
#####################################################################################################################################################


"""
Directorios de entrada
"""


# SENAL PAM
input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
# VSC LADO DERECHO
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
# VSC LADO IZQUIERDO
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'


"""
Funcion para cargar los archivos .npy
"""


def load_npy_files(input_dir):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])
     # verificar orden con que entrar los archivos en X e Y
    file_names = [os.path.basename(f) for f in files]
    print(f"Archivos: {file_names}\n")
    data = [np.load(f) for f in files]
    return np.array(data)


################################################################################################
########## CARGA DE DATOS DE ENTRADAS Y SALIDAS PARA LA RED (X: INPUTS; Y: OUTPUTS) ############
################################################################################################


X = load_npy_files(input_pam_dir) # inputs
Y = load_npy_files(output_vscd_dir) # outputs


"""
Verificar las formas de los datos cargados (# entradas, filas, columnas, canales)
"""


print(f"Shape de los inputs (X): {X.shape}")
print(f"Shape de los outputs (Y): {Y.shape}")


"""
Definir la U-Net con regularizacion L2
"""

def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Regularizer
    #l2_reg = tf.keras.regularizers.l2(l2_lambda)
    
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs) #filtro original=64
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1) #filtro original=64
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1) #filtro original=128
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2) #filtro original=128
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2) #filtro original=256
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3) #filtro original=256
    
    # Decoder
    u4 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3) #filtro original=128
    u4 = tf.keras.layers.concatenate([u4, c2])
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4) #filtro original=128
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4) #filtro original=128
    
    u5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4) #filtro original=64
    u5 = tf.keras.layers.concatenate([u5, c1])
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5) #filtro original=64
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5) #filtro original=64
    
    outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='linear')(c5)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


"""
Definir la metrica NMSE ajustada para utilizar la varianza de los valores verdaderos
"""


def nmse(y_true, y_pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
    var_true = tf.keras.backend.var(y_true)
    return mse / var_true


###########################################
########## **HIPERPARAMETROS** ############
###########################################


max_epoch = 300
batchsize = 8
learning_rate = 0.001
#l2_lambda = 0.01
validation_split = 0.3 # 70% entrenamiento & 30% validacion


 # alpha: el lr min al que llegara el decaimiento sera el 10% del lr inicia
#alpha = 0.1
# decay steps: Numero de pasos de entrenamiento tras los cuales el learning rate decaera desde su valor inicial hasta el valor final determinado por alpha
#decay_steps = 300#(int(X.shape[0]/batchsize))*max_epoch 
#print("Total pasos de decaimiento ->",decay_steps, "pasos.")


"""
**CREACIÓN DEL MODELO U-NET**
"""


input_shape = X.shape[1:]  # forma del input a entrar. en este caso esta forma debe coincidir con las matrices que entran a la red tensor X = [#inputs, columnas, filas, canales]. Se omite #inputs
model = unet_model(input_shape)


"""
**DEFINICION DE LA FUNCION DE DECAIMIENTO, ALGORITMO OPTIMIZADOR, FUNCION DE PERDIDA Y METRICA**
"""

'''
#funcion decaimiento de coseno
decay_cosine = tf.keras.experimental.CosineDecay(learning_rate, decay_steps)
def lr_schedule(X):
    return float(decay_cosine(X))
    

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
'''

optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[nmse])


"""
#########################################################################################################<br>
#########################################################################################################<br>
#########################################################################################################
"""



#### ENTRENAMIENTO DE LA RED ####

start_time = time.time()
#history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, callbacks=[lr_scheduler], validation_split=validation_split)
history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, validation_split=validation_split)
end_time = time.time()
total_time = end_time - start_time
min_time = total_time / 60
print(f'Tiempo total de entrenamiento: {min_time:.2f} minutos.')

"""
#########################################################################################################<br>
#########################################################################################################<br>
#########################################################################################################
"""


"""
Visualizar el NMSE
"""


plt.plot(history.history['nmse'], label='NMSE (entrenamiento)')
plt.plot(history.history['val_nmse'], label='NMSE (validacion)')
plt.xlabel('Epoca')
plt.ylabel('NMSE')
plt.legend()
plt.show()


"""
**GUARDAR UN MODELO ESPECÍFICO**
"""


# Directorio en donde se almacenara el modelo
save_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/SANOS' + persona
os.makedirs(save_dir, exist_ok=True)  # Crear el directorio si no existe

# Nombre del archivo del modelo
model_name = 'unet_model_vscd.keras'

# Ruta completa del archivo
model_path = os.path.join(save_dir, model_name)

# Guardar el modelo entrenado
model.save(model_path)





'''



"""
**CARGAR UN MODELO ESPECÍFICO**
"""


modelo_cargado = tf.keras.models.load_model('D:/TT/Memoria/waveletycnn/codigo_python/modelos_generados/unet_model_7.keras',
                                           custom_objects={'nmse': nmse})


"""
**CARGAR COEFICIENTES (MATRIZ COMPLEJA) DE LA SEÑALES PAM ORIGINALES T OBTENER CANTIDAD DE MATRICES COMPLEJAS ENCONTRADAS**
"""


input_matrix_complex_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/inputs_coeficientes' # coeficientes de senal PAM original (sin ruido) en formato .mat
output_matrix_complex_pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/inputs_coeficientes_npy' # coeficientes de senal PAM original (sin ruido) en formato .npy
os.makedirs(output_matrix_complex_pam_dir, exist_ok=True) # crear directorio "output_matrix_complex_pam_dir" si no existe

# Se leen la cantidad de coeficientes o matrices complejas de senales PAM encontradas en el directorio "input_matrix_complex_pam_dir"
total_matrix_complex_pam = sum(1 for filename in os.listdir(input_matrix_complex_pam_dir) if filename.endswith('.mat')) + 1
print("Total de matrices complejas PAM encontradas -> ", total_matrix_complex_pam - 1)


"""
**TRANSFORMAR A FORMATO .npy LAS MATRICES COMPLEJAS ASOCIADAS A LA SEÑAL ORIGINAL PAM**
"""


# Funcion para convertir una matriz compleja de .mat a .npy
def convert_mat_to_npy_original_signal(input_dir, output_dir, prefix):
    for i in range(1, total_matrix_complex_pam):
        mat_file = os.path.join(input_dir, f'{prefix}_{i}.mat')
        npy_file = os.path.join(output_dir, f'{prefix}_npy_{i}.npy')
        
        # Cargar el archivo .mat
        mat_data = loadmat(mat_file)
        
        # Extraer la matriz compleja
        matrix_key = [key for key in mat_data.keys() if not key.startswith('__')][0]
        matrix = mat_data[matrix_key]
        
        # Guardar la matriz en formato .npy
        np.save(npy_file, matrix)


# Convertir las matrices comlejas de senales PAM originales de formato .mat a .npy
convert_mat_to_npy_original_signal(input_matrix_complex_pam_dir, output_matrix_complex_pam_dir, 'matrix_complex_pam_to_predict')


"""
**PREDECIR COEFICIENTES DE UNA SEÑAL DE VSC A PARTIR DE COEFICIENTES DE UNA SEÑAL PAM**
"""


# Funcion para predecir con el modelo entrenado
def predecir_coefs(modelo_cargado, input_data):
    coefs_predicted = modelo_cargado.predict(input_data)
    return coefs_predicted



# Listar todos los archivos en el directorio
archivos_npy_dir = os.listdir(output_matrix_complex_pam_dir)
# Filtrar solo los archivos .npy
archivos_npy = [f for f in archivos_npy_dir if f.endswith('.npy')]

# Cargar una matriz de entrada para hacer una prediccion 
# Leer el primer archivo .npy
nombre_archivo_pam_npy = archivos_npy[0]
archivo_pam_npy_dir = os.path.join(output_matrix_complex_pam_dir, nombre_archivo_pam_npy) # ELEGIR ARCHIVOS NPY A LEER (0, 1, 2, 3, 4, ...)

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



"""
**REALIZAR PREDICCIÓN (OBTENCIÓN DE COEFICIENTES DE SEÑAL VSC ESTIMADA)**
"""


# Realizar la predicción
predicted_output = predecir_coefs(modelo_cargado, tensor_input_matrix_pam)

# Mostrar la predicción
print("Prediccion de la primera muestra de entrada:")
print(predicted_output)
print("Formato de los coeficientes de la senal VSC estimada: ", predicted_output.shape)


"""
**TRANSFORMAR LA SALIDA ESTIMADA A UN FORMATO (36, 1024) Y LUEGO DE .npy a .mat**
"""


# Transformar la matriz tensor VSC a una matriz compleja de formato .mat
# El tensor tiene la forma (1, 36, 1024, 2) y necesitamos transformarlo a (36, 1024) a matriz compleja
complex_matrix_vsc = predicted_output[0, :, :, 0] + 1j * predicted_output[0, :, :, 1]
print("Formato nuevo:",complex_matrix_vsc.shape)


# Especificar la ruta completa del archivo, incluyendo el nombre y la extensión .mat
ruta_archivo = "D:\TT\Memoria\waveletycnn\codigo_matlab\codigo_fuente\coefs_vsc_predicted\coefs_vsc_predicted.mat"

# Guardar la matriz compleja en el archivo especificado
scipy.io.savemat(ruta_archivo, {'complex_matrix_vsc': complex_matrix_vsc})



"""
**TRANSFORMACIÓN DE COEFICIENTES DE LA SENAL VSC ESTIMADA DE FORMATO .mat a .npy**
"""


"""
**GUARDADO DE LA MATRIZ COMPLEJA DE LA VSC ESTIMADA EN UNA CARPETA EN EL DIRECTORIO DE MATLAB EN FORMATO .mat**
"""


# Directorio donde se guardara la matriz compleja de la VSC estimada en formato .mat
mat_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/coefs_vsc_predicted'
os.makedirs(mat_dir, exist_ok=True) # crear directorio "mat_dir" si no existe

# Extraer el número como cadena
numero_como_cadena = nombre_archivo_pam_npy.split('_')[-1].split('.')[0]

# Nombre que tendra el archivo de la matriz compleja d la VSC estimada .mat. Se concatena el numero del archivo estimado
mat_filename = f'matrix_complex_vsc_predicted_{numero_como_cadena}.mat'
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




'''