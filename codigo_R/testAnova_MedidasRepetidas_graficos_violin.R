# Cargar las librerías necesarias
library(ggplot2)
library(gridExtra)
library(dplyr)
library(tidyr)
library(ggsignif)
library(afex)
library(emmeans);

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


#######################################################
#### TEST ANOVA #######################################
#######################################################

# Convertir a factores si no lo son
datos_totales$Nombre <- factor(datos_totales$Nombre)
datos_totales$Tipo <- factor(datos_totales$Tipo)


resultado_anova <- aov(mfARI ~ Nombre + Tipo, data = datos_totales)

# Muestra el resumen del ANOVA
print(summary(resultado_anova))

# Comparaciones post hoc
emmeans_result <- emmeans(resultado_anova, ~ Nombre)
pairs(emmeans_result)
 


# Gráfico de violín con comparación de grupos y significancia
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
  y_position = c(15, 16, 17, 18), # Ajusta las posiciones según sea necesario
  textsize = 3)

print(grafico_mfARI_significancia)




