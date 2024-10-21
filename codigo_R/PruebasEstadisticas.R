# Cargar las librerias necesarias
library(ggplot2)
library(gridExtra)

# Directorio donde se encuentran los archivos CSV
directorio_csv <- "D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_R/AC_Indexs_Norm_NNM2d5OI_FD_LR0d001_E300_DO0d2"

# Leer los archivos CSV y almacenarlos en dataframes
IndexAC_SANO_DERECHO <- read.csv(file.path(directorio_csv, "IndexAC_SANO_DERECHO.csv"))
IndexAC_SANO_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_SANO_IZQUIERDO.csv"))
IndexAC_TEC_DERECHO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_DERECHO.csv"))
IndexAC_TEC_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_IZQUIERDO.csv"))

# Crear una lista de dataframes y sus nombres
dataframes <- list(
  IndexAC_SANO_IZQUIERDO = IndexAC_SANO_IZQUIERDO,
  IndexAC_SANO_DERECHO = IndexAC_SANO_DERECHO,
  IndexAC_TEC_IZQUIERDO = IndexAC_TEC_IZQUIERDO,
  IndexAC_TEC_DERECHO = IndexAC_TEC_DERECHO
  
)

# Colores para cada tipo
colores_mfARI <- c("skyblue", "lightgreen", "salmon", "gold")
colores_ARI <- c("salmon", "lightblue", "lightgreen", "skyblue")

# Crear gráficos de violín para mfARI y ARI
graficos_mfARI <- lapply(names(dataframes), function(nombre) {
  ggplot(dataframes[[nombre]], aes(x = "", y = mfARI)) +
    geom_violin(trim = FALSE, fill = colores_mfARI[which(names(dataframes) == nombre)]) +
    geom_boxplot(width = 0.07, outlier.shape = NA, color = "black", fill = "white") +
    labs(title = paste("mfARI -", nombre), y = "mfARI", x = "") +
    theme_minimal() +
    theme(legend.position = "none")
})

graficos_ARI <- lapply(names(dataframes), function(nombre) {
  ggplot(dataframes[[nombre]], aes(x = "", y = ARI)) +
    geom_violin(trim = FALSE, fill = colores_ARI[which(names(dataframes) == nombre)]) +
    geom_boxplot(width = 0.07, outlier.shape = NA, color = "black", fill = "white") +
    labs(title = paste("ARI -", nombre), y = "ARI", x = "") +
    theme_minimal() +
    theme(legend.position = "none")
})

# Mostrar todos los gráficos de mfARI en una sola imagen
grid.arrange(grobs = graficos_mfARI, ncol = 2, top = "Gráficos de mfARI")

# Mostrar todos los gráficos de ARI en una sola imagen
grid.arrange(grobs = graficos_ARI, ncol = 2, top = "Gráficos de ARI")
