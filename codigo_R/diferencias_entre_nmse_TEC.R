# Instalar y cargar paquetes necesarios
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("reshape2")) install.packages("reshape2")
library(ggplot2)
library(reshape2)

# Crear el dataframe con los valores proporcionados
NMSE_Hemisferio_Afectado <- c(0.0513, 0.0349, 0.0553, 0.0308, 0.0297, 0.0332, 0.0221, 0.0176, 
                              0.0127, 0.0168, 0.0222, 0.0187, 0.1247, 0.1123, 0.0559, 0.0188, 
                              0.0413, 0.0174, 0.0102, 0.0920, 0.0146, 0.1300, 0.0201, 0.0110, 
                              0.0115, 0.0068, 0.0240)
NMSE_Hemisferio_No_Afectado <- c(0.0400, 0.1290, 0.0166, 0.0173, 0.0623, 0.0192, 0.0217, 0.0258, 
                                 0.0124, 0.0212, 0.0293, 0.0065, 0.0119, 0.0133, 0.0108, 0.0200, 
                                 0.0121, 0.0462, 0.0208, 0.0154, 0.0180, 0.0300, 0.0122, 0.0140, 
                                 0.0561, 0.0126, 0.0129)
df <- data.frame(NMSE_Hemisferio_Afectado, NMSE_Hemisferio_No_Afectado)

# Prueba de normalidad (Shapiro-Wilk)
shapiro_afectado <- shapiro.test(df$NMSE_Hemisferio_Afectado)
shapiro_no_afectado <- shapiro.test(df$NMSE_Hemisferio_No_Afectado)

# Imprimir resultados de la prueba de normalidad
print("Prueba de normalidad para Hemisferio Afectado:")
print(shapiro_afectado)
print("Prueba de normalidad para Hemisferio No Afectado:")
print(shapiro_no_afectado)

# Decidir qué prueba estadística aplicar
if (shapiro_afectado$p.value > 0.05 && shapiro_no_afectado$p.value > 0.05) {
  # Si ambos grupos son normales, aplicar t-test pareado
  test_result <- t.test(df$NMSE_Hemisferio_Afectado, 
                        df$NMSE_Hemisferio_No_Afectado, 
                        paired = TRUE)
  print("Resultado del t-test pareado:")
  print(test_result)
} else {
  # Si alguno no es normal, aplicar Wilcoxon signed-rank test
  test_result <- wilcox.test(df$NMSE_Hemisferio_Afectado, 
                             df$NMSE_Hemisferio_No_Afectado, 
                             paired = TRUE)
  print("Resultado del Wilcoxon signed-rank test:")
  print(test_result)
}

# Transformar el dataframe a formato largo para ggplot2
df_long <- melt(df, variable.name = "Grupo", value.name = "NMSE")

# Cambiar etiquetas para reflejar los hemisferios
df_long$Grupo <- factor(df_long$Grupo, 
                        levels = c("NMSE_Hemisferio_Afectado", "NMSE_Hemisferio_No_Afectado"), 
                        labels = c("Hemisferio Afectado", "Hemisferio No Afectado"))

# Crear el gráfico combinado (violín + boxplot)
ggplot(df_long, aes(x = Grupo, y = NMSE, fill = Grupo)) +
  geom_violin(alpha = 0.5, color = "black") +  # Gráfico de violín
  geom_boxplot(width = 0.2, color = "black", outlier.shape = NA, alpha = 0.7) +  # Agregar boxplot
  geom_jitter(width = 0.1, size = 2, color = "black") +  # Puntos individuales en gris
  theme_minimal() +
  labs(title = "Distribución de NMSE por Hemisferio - Grupo TEC", 
       x = "Hemisferio", 
       y = "NMSE") +
  scale_fill_manual(values = c("#e41a1c", "#377eb8"))  # Colores para violines
