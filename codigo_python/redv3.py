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

#CONVERTIR MATRICES DE: .mat  --> .npyDirectorios de entrada y salida: SUJETOS SANOS Y PACIENTES TEC

#(1) sujetos sanos: 1_HEMU - 2_DAOC - 3_DASI - 4_DABA - 5_HEFU - 6_JOBO - 7_ROMI - 8_FEGA - 9_GAGO - 10_MIMO - 11_JULE - 12_NIGA - 13_BYLA - 14_ARVA
#  - 15_CLSE - 16_PAAR - 17_VATO - 18_FEBE - 19_VINA - 20_CLHE - 21_MAIN - 22_ALSA - 23_MIRA - 24_LACA - 25_GOAC - 26_ANGL - 27_HC036101

#(2) pacientes tec: 1_DENI1005 - 2_KNOW1001 - 3_ALI0 - 4_BUTL - 5_HAGG - 6_HASTI007 - 7_BOAM - 8_DANE0005 - 9_GREG - 10_AITK - 11_RANS0000 - 12_JONES004 - 13_PERR - 14_SLAC
#  - 15_HEPPL010 - 16_RICHS010 - 17_KENT0007 - 18_STAN1002 - 19_MCDON022 - 20_PULL - 21_MORR1002 - 22_PARK - 23_HIGH - 24_NOBL - 25_COWL - 26_KHAN - 27_NOLA

# Directorios de las matrices complejas (coeficientes) de las senales PAM ,VSCd y VSCi de suejtos sanos o pacientes con tec. Se debe modificar estos directorios para ir
# generando los respectivos modelos de cada individuo.

# DIRECTORIOS

# SUJETO SANO O PACIENTE TEC POR ANALIZAR
# DIRECTORIO ejemplo: PACIENTE TEC --> 'TEC/1_DENI1005'
persona = 'SANOS/7_ROMI'
# Sector de la VSC por analizar (derecho o izquierdo):
sector = 'derecho'

 # INPUT PARA LA RED
# INPUT PAM
input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_mat'
# INPUT VSCd
input_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_mat'
# INPUT VSCI
input_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_mat'

# OUTPUT O SALIDAS ESPERADAS PARA LA RED
# OUTPUT PAM
output_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy'
# OUTPUT PAM
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy'
# OUTPUT PAM
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy'

#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################
print(input_pam_dir)
print(input_vscd_dir)
print(input_vsci_dir)

#Crear los directorios de salida si no existen
os.makedirs(output_pam_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE PAM EN FORMATO .npy
os.makedirs(output_vscd_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCd EN FORMATO .npy
os.makedirs(output_vsci_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCi EN FORMATO .npy

total_files=sum(1 for filename in os.listdir(input_pam_dir) if filename.endswith('.mat')) + 1
print("Total de archivos a analizar -> ",total_files-1)

#Funcion para convertir archivos .mat a .npy
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

#Convertir archivos .mat a .npy para PAM y VSC
convert_mat_to_npy(input_pam_dir, output_pam_dir, 'matrix_complex_pam')
convert_mat_to_npy(input_vscd_dir, output_vscd_dir, 'matrix_complex_vscd')
convert_mat_to_npy(input_vsci_dir, output_vsci_dir, 'matrix_complex_vsci')

#Conversion a tensor tridimensional (Estructura adecuada para entrenar la red U-net)
#Directorios salida para matrices con estructura tensor tridimensional
input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy'
input_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy'
input_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy'

output_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'

#Crear directorios de salida si no existen (matrices complejas en forma de tensor tridimensional)
os.makedirs(output_pam_dir, exist_ok=True)
os.makedirs(output_vscd_dir, exist_ok=True)
os.makedirs(output_vsci_dir, exist_ok=True)

# PROCESAR MATRICES SEGÚN UNA NORMALIZACIÓN Y ORGANIZAR DATOS PARA LA RED

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


#PROCESAR MATRICES COMPLEJAS EN LA CARPETA input_pam_dir y procesar matrices mediante una normalización
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

       
#PROCESAR MATRICES COMPLEJAS EN LA CARPETA input_vscd_dir o input_vsci_dir
if sector == 'derecho':
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
            
    print("Procesamiento completado: VSCd lado derecho.")
    
else:    
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
            
    print("Procesamiento completado: VSCi lado izquierdo.")


#Verificacion de "shape" - matrices pam y vsc
#Directorios de salida a verificar:
output_pam_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
output_vscd_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
output_vsci_dir_check = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'

#Funcion para verificar la forma de una matriz
def verificar_shape(directorio, nombre_archivo):
    path = os.path.join(directorio, nombre_archivo)
    matriz = np.load(path)
    return matriz.shape

#Verificar la forma de un archivo de ejemplo en output_pam_dir_check
ejemplo_pam = os.listdir(output_pam_dir_check)[0]  # Obtener el primer archivo de la carpeta
shape_pam = verificar_shape(output_pam_dir_check, ejemplo_pam)
print(f"Shape de {ejemplo_pam} en {output_pam_dir_check}: {shape_pam}")


#Verificar la forma de un archivo de ejemplo en output_vscd_dir_check o output_vsci_dir_check
if sector == 'derecho':
    ejemplo_vscd = os.listdir(output_vscd_dir_check)[0]  # Obtener el primer archivo de la carpeta
    shape_vscd = verificar_shape(output_vscd_dir_check, ejemplo_vscd)
    print(f"Shape de {ejemplo_vscd} en {output_vscd_dir_check}: {shape_vscd}")
else:   
    ejemplo_vsci = os.listdir(output_vsci_dir_check)[0]  # Obtener el primer archivo de la carpeta
    shape_vsci = verificar_shape(output_vsci_dir_check, ejemplo_vsci)
    print(f"Shape de {ejemplo_vsci} en {output_vsci_dir_check}: {shape_vsci}")

#*********************************************************************************************************************************************************************************
#************************************************************************ Red Neuronal Profunda: U-net ***************************************************************************
#*********************************************************************************************************************************************************************************

#Directorios de entrada
# SENAL PAM
input_pam_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/PAMnoises_matrixcomplex_npy_tensor3d'
# VSC LADO DERECHO
output_vscd_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCdnoises_matrixcomplex_npy_tensor3d'
# VSC LADO IZQUIERDO
output_vsci_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona + '/VSCinoises_matrixcomplex_npy_tensor3d'


#Funcion para cargar los archivos .npy
def load_npy_files(input_dir):
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])
     # verificar orden con que entrar los archivos en X e Y
    file_names = [os.path.basename(f) for f in files]
    print(f"Archivos: {file_names}\n")
    data = [np.load(f) for f in files]
    return np.array(data)

#CARGA DE DATOS DE ENTRADAS Y SALIDAS PARA LA RED (X: INPUTS; Y: OUTPUTS)
# Se identifica que sector del cerebro se desea analizar:
lado = ''
if sector == 'derecho':
    X = load_npy_files(input_pam_dir) # inputs
    Y = load_npy_files(output_vscd_dir) # outputs
    lado = 'vscd'
    print('Modelo para VSC: sector derecho del cerebro.\n')
else:
    X = load_npy_files(input_pam_dir) # inputs
    Y = load_npy_files(output_vsci_dir) # outputs
    lado = 'vsci'
    print('Modelo para VSC: sector izquierdo del cerebro.\n')
print('Abreviacion de sector a estudiar:', lado)


#Verificar las formas de los datos cargados (# entradas, filas, columnas, canales):
print(f"Shape de los inputs (X): {X.shape}")
print(f"Shape de los outputs (Y): {Y.shape}")

#Definir la U-Net
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


#Definir la metrica NMSE ajustada para utilizar la varianza de los valores verdaderos
def nmse(y_true, y_pred):
    mse = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred))
    var_true = tf.keras.backend.var(y_true)
    return mse / var_true

#****************
#HIPERPARAMETROS
#****************
max_epoch = 200
batchsize = 8
learning_rate = 0.0001
#l2_lambda = 0.01
validation_split = 0.2 # 80% entrenamiento & 20% validacion
# alpha: el lr min al que llegara el decaimiento sera el 10% del lr inicia
#alpha = 0.1
# decay steps: Numero de pasos de entrenamiento tras los cuales el learning rate decaera desde su valor inicial hasta el valor final determinado por alpha
#decay_steps = (int(X.shape[0]/batchsize))*max_epoch 
#print("Total pasos de decaimiento ->",decay_steps, "pasos.")


#CREACIÓN DEL MODELO U-NET
input_shape = X.shape[1:]  # forma del input a entrar. en este caso esta forma debe coincidir con las matrices que entran a la red tensor X = [#inputs, columnas, filas, canales]. Se omite #inputs
model = unet_model(input_shape)


#DEFINICION DE ALGORITMO OPTIMIZADOR, FUNCION DE PERDIDA Y METRICA
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[nmse])

#########################################################################################################
#########################################################################################################
#########################################################################################################
#ENTRENAMIENTO DE LA RED:
start_time = time.time()
#history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, callbacks=[lr_scheduler], validation_split=validation_split)
history = model.fit(X, Y, epochs=max_epoch, batch_size=batchsize, validation_split=validation_split)
end_time = time.time()
total_time = end_time - start_time
min_time = total_time / 60
print(f'Tiempo total de entrenamiento: {min_time:.2f} minutos.')
#########################################################################################################
#########################################################################################################
#########################################################################################################

#Visualizar el NMSE
plt.plot(history.history['nmse'], label='NMSE (entrenamiento)')
plt.plot(history.history['val_nmse'], label='NMSE (validacion)')
plt.xlabel('Epoca')
plt.ylabel('NMSE')
plt.legend()

# Guardar grafica de curvas NMSE

# Directorio y nombre del archivo
output_dir_graphic = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona  # Reemplaza con tu directorio
output_file_graphic =  'unet_model_' + lado + '_' + sector + '_graphic.png'  # Reemplaza con tu nombre de archivo

# Guardar el gráfico
output_path_graphic = f'{output_dir_graphic}/{output_file_graphic}'
plt.savefig(output_path_graphic, format='png')
plt.show()

#GUARDAR UN MODELO ESPECIFICO
# Directorio en donde se almacenara el modelo
save_dir = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/signals_LDS/' + persona
os.makedirs(save_dir, exist_ok=True)  # Crear el directorio si no existe

# Nombre del archivo del modelo
model_name = 'unet_model_' + lado + '_' + sector + '.keras'

# Ruta completa del archivo
model_path = os.path.join(save_dir, model_name)

# Guardar el modelo entrenado
model.save(model_path)




