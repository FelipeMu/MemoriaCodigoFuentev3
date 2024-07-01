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

'''
CONVERTIR MATRICES DE: .mat  --> .npy

Directorios de entrada y salida: SUJETOS SANOS Y PACIENTES TEC

(1) sujetos sanos: 1_HEMU - 2_DAOC - 3_DASI - 4_DABA - 5_HEFU - 6_JOBO - 7_ROMI - 8_FEGA - 9_GAGO - 10_MIMO - 11_JULE - 12_NIGA - 13_BYLA - 14_ARVA - 15_CLSE - 16_PAAR - 17_VATO - 18_FEBE - 19_VINA - 20_CLHE - 21_MAIN - 22_ALSA - 23_MIRA - 24_LACA - 25_GOAC - 26_ANGL - 27_HC036101

(2) pacientes tec: 1_DENI1005 - 2_KNOW1001 - 3_ALI0 - 4_BUTL - 5_HAGG - 6_HASTI007 - 7_BOAM - 8_DANE0005 - 9_GREG - 10_AITK - 11_RANS0000 - 12_JONES004 - 13_PERR - 14_SLAC - 15_HEPPL010 - 16_RICHS010 - 17_KENT0007 - 18_STAN1002 - 19_MCDON022 - 20_PULL - 21_MORR1002 - 22_PARK - 23_HIGH - 24_NOBL - 25_COWL - 26_KHAN - 27_NOLA
'''




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

#Crear los directorios de salida si no existen

os.makedirs(output_pam_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE PAM EN FORMATO .npy
os.makedirs(output_vscd_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCd EN FORMATO .npy
os.makedirs(output_vsci_dir, exist_ok=True) # DIRECTORIO PARA GUARDAR MATRICES COMPLEJAS DE VSCi EN FORMATO .npy


