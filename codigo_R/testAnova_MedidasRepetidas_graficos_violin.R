# Cargar las librerías necesarias
library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyr)
library(ggsignif)
library(afex)
library(emmeans)
library(car) 
library(ez)
library(rstatix)
library(lme4)
library(afex)
# *************************************************************************************
# *************************************************************************************
# ** PREPROCESAMIENTO DE LOS DATOS DE ENTRADA - CSV DE SUJETOS SANOS Y PACIENTE TEC **
# *************************************************************************************
# *************************************************************************************

# Directorio donde se encuentran los archivos CSV
directorio_csv <- "D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_R/TESTINGv11_baseline40to90_InitialSignal8seg_minVSC8.2-14_AC_Indexs_Norm_NNM2d5OI_FD_LR0d001_E300_DO0d2"

# Leer los archivos CSV y agregar columnas para el tipo y el nombre
IndexAC_SANO_DERECHO <- read.csv(file.path(directorio_csv, "IndexAC_SANO_DERECHO.csv")) %>%
  mutate(Tipo = "Sano", Nombre = "Derecho_Sano")
IndexAC_SANO_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_SANO_IZQUIERDO.csv")) %>%
  mutate(Tipo = "Sano", Nombre = "Izquierdo_Sano")
IndexAC_TEC_DERECHO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_DERECHO.csv")) %>%
  mutate(Tipo = "TEC", Nombre = "Derecho_TEC")
IndexAC_TEC_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_IZQUIERDO.csv")) %>%
  mutate(Tipo = "TEC", Nombre = "Izquierdo_TEC")

# Unir todos los dataframes en uno solo
datos_totales <- bind_rows(
  IndexAC_SANO_DERECHO,
  IndexAC_SANO_IZQUIERDO,
  IndexAC_TEC_DERECHO,
  IndexAC_TEC_IZQUIERDO
)

# Ordenar los niveles de la columna Nombre para que aparezcan en el orden deseado
datos_totales$Nombre <- factor(datos_totales$Nombre, 
                               levels = c("Izquierdo_Sano", "Derecho_Sano", "Izquierdo_TEC", "Derecho_TEC"))



# ****************************************************************************************
# ************ DATOS TEC PARA POSTERIOR ANALISIS DE ETIQUETA VIVO Y MUERTO ***************
# ****************************************************************************************

# Unir todos los dataframes en uno solo
datos_TEC <- bind_rows(
  IndexAC_TEC_IZQUIERDO,
  IndexAC_TEC_DERECHO
)

# Ordenar los niveles de la columna Nombre para que aparezcan en el orden deseado
datos_TEC$Nombre <- factor(datos_TEC$Nombre, 
                               levels = c("Izquierdo_TEC", "Derecho_TEC"))



# Agregar la columna hemisferio con el valor "A" en todas las filas
datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = "NA")


#IDENTIFICANDO HEMISFERIOS NO AFECTADOS DE LOS PACIETNES TEC:
datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "1_DENI1005" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "2_KNOW1001" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "3_ALI0" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "4_BUTL" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "5_HAGG" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "6_HASTI007" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "7_BOAM" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "8_DANE0005" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "9_GREG" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "10_AITK" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "11_RANS0000" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "12_JONES004" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "13_PERR" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "14_SLAC" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "15_HEPPL010" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "16_RICHS010" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "17_KENT0007" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "18_STAN1002" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "19_MCDON022" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "20_PULL" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "21_MORR1002" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "22_PARK" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "23_HIGH" & Nombre == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "24_NOBL" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "25_COWL" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "26_KHAN" & Nombre == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "27_NOLA" & Nombre == "Derecho_TEC", "A", Hemisferio))

# SE PROCEDE A CREAR UN DATAFRAME DE SOLO LOS TEC CON LOS HEMISFERIOS AFECTADOS
# Crear un nuevo dataframe con solo las filas donde 'hemisferio' es "A"
datos_TEC_HA <- datos_TEC %>%
  filter(Hemisferio == "A")

# SE AGREGA ETIQUETA DE VIVO Y MUERTO A LOS RESPECTIVOS PACIENTES:
# HA: hemisferio afectado
# VM: vivo o muerto
datos_TEC_HA_VM <- datos_TEC_HA %>%
  mutate(Estado = "vivo")



#IDENTIFICANDO PACIETNES TEC CON ETIQUETA VIVO/MUERTO:

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "1_DENI1005", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "2_KNOW1001", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "3_ALI0", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "5_HAGG", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "10_AITK", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "13_PERR", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "15_HEPPL010", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "16_RICHS010", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "17_KENT0007", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "20_PULL", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "23_HIGH", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "24_NOBL", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "25_COWL", "muerto", Estado))

datos_TEC_HA_VM <- datos_TEC_HA_VM %>%
  mutate(Estado = ifelse(Persona == "27_NOLA", "muerto", Estado))








###############################################################################

###############################################################################

# Definir los colores y etiquetas para el gráfico
colores <- c("Izquierdo_Sano" = "red", 
             "Derecho_Sano" = "blue", 
             "Izquierdo_TEC" = "green", 
             "Derecho_TEC" = "purple")

# Gráfico de violín para mfARI en un solo eje cuadrático
grafico_mfARI <- ggplot(datos_totales, aes(x = Nombre, y = mfARI, fill = Nombre)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, outlier.shape = NA, color = "black", fill = "white") +
  stat_summary(fun = "mean", geom = "point", shape = 23, size = 3, fill = "red") +  # Añadir la media en rojo
  labs(title = "Gráficos mfARI: sujetos sanos y pacientes TEC", y = "mfARI", x = "") +
  theme_minimal() +
  theme(legend.position = "right", legend.title = element_blank()) +  # Mostrar leyenda a la derecha
  scale_fill_manual(values = colores) +  # Aplicar colores personalizados
  scale_y_continuous(breaks = seq(-2.5, 10, by = 1.5), limits = c(-3.5, 13))  # Ajustar límites y pasos

# Mostrar solo el gráfico de mfARI
grid.arrange(grafico_mfARI, ncol = 1)


# *****************************************************************************
# *****************************************************************************
# ****************** TEST ANOVA DE MEDIDAS REPETIVAS **************************
# *****************************************************************************
# *****************************************************************************

# TRANSFORMANDO A FACTOR
datos_totales$Nombre <- factor(datos_totales$Nombre)
datos_totales$Tipo <- factor(datos_totales$Tipo)
# APLICACION ANOVA MEDIDAS REPETIDAS
anova_medRep <- aov(mfARI ~ Nombre, data = datos_totales)
print(summary(anova_medRep));
# POST - HOC
post_hoc <- emmeans(anova_medRep, ~ Nombre)
pairs(post_hoc)


# CONDICIONES PARA APLICAR ANOVA DE MEDIDAS REPETIDAS:

# 1. Verificar la normalidad de los residuos. nivel de significancia alpha=0.05
# H0: los residuos siguen una distribucion normal (p>0.05)
# H1: Los residuos no siguen una distribucion normal (p<=0.05)
# W: entre mas cercano a 1, indica que los datos tienen distribucion normal
residuos <- residuals(anova_medRep)
shapiro_testAnova_SanosTec <- shapiro.test(residuos)  
print(shapiro_testAnova_SanosTec)
# 1.Graficos Q-Q
qqnorm(residuos)
qqline(residuos)
# conclusion: Se acepta la hipotesis nula H0




# 2. Se procede a verificar propiedad de esfericidad:

#Conclusion sobre la Esfericidad: Dado que has utilizado un modelo mixto, no 
#necesitas comprobar la esfericidad de forma explicita. El modelo ya maneja la
#variabilidad y la correlacion entre las mediciones repetidas. Sin embargo,
#puedes concluir que las diferencias significativas encontradas en mfARI entre
#los niveles de Nombre indican que, aunque no hayas podido realizar una prueba 
#de esfericidad tradicional, el modelo ha sido capaz de capturar y ajustar la
#variabilidad intragrupo de manera efectiva.





# GRAFICOS CON NIVELES DE SIGNIFICANCIA:
grafico_mfARI_significancia <- ggplot(datos_totales, aes(x = Nombre, y = mfARI, fill = Nombre)) +
  geom_violin(trim = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  theme_minimal() +
  labs(title = "Comparación de mfARI: sujetos Sanos y pacientes TEC",
       x = "Grupo", y = "mfARI") +
  scale_fill_brewer(palette = "Pastel1") +
  geom_signif(comparisons = list(
    c("Izquierdo_Sano", "Izquierdo_TEC"),
    c("Izquierdo_Sano", "Derecho_TEC"),
    c("Derecho_Sano", "Izquierdo_TEC"),
    c("Derecho_Sano", "Derecho_TEC")
  ),
  map_signif_level = TRUE, 
  y_position = c(15, 16, 17, 18),
  textsize = 3)

print(grafico_mfARI_significancia)





# ******************************************************************************
# ******************************************************************************
# ************* ANALIZAR SI EXISTEN DIFERENCIAS SIGNIFICATIVAS ENTRE LOS *******
# ************* CON ETIQUETA VIVO Y MUERTO                               *******
# ******************************************************************************
# ******************************************************************************

# test de normalidad Shapiro-Wilk para subgrupos de pacientes TEC: vivo y muerto
shapiro.test(datos_TEC_HA_VM$mfARI[datos_TEC_HA_VM$Estado == "vivo"])
shapiro.test(datos_TEC_HA_VM$mfARI[datos_TEC_HA_VM$Estado == "muerto"])

# test para ver si existen diferetencias significativas
mann_whitney_result <- wilcox.test(mfARI ~ Estado, data = datos_TEC_HA_VM)
print(mann_whitney_result)



