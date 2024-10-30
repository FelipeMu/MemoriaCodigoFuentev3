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
  mutate(Tipo = "Sano", Grupo = "Derecho_Sano")
IndexAC_SANO_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_SANO_IZQUIERDO.csv")) %>%
  mutate(Tipo = "Sano", Grupo = "Izquierdo_Sano")
IndexAC_TEC_DERECHO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_DERECHO.csv")) %>%
  mutate(Tipo = "TEC", Grupo = "Derecho_TEC")
IndexAC_TEC_IZQUIERDO <- read.csv(file.path(directorio_csv, "IndexAC_TEC_IZQUIERDO.csv")) %>%
  mutate(Tipo = "TEC", Grupo = "Izquierdo_TEC")

# Unir todos los dataframes en uno solo
datos_totales <- bind_rows(
  IndexAC_SANO_DERECHO,
  IndexAC_SANO_IZQUIERDO,
  IndexAC_TEC_DERECHO,
  IndexAC_TEC_IZQUIERDO
)

# Ordenar los niveles de la columna Nombre para que aparezcan en el orden deseado
datos_totales$Grupo <- factor(datos_totales$Grupo, 
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
datos_TEC$Grupo <- factor(datos_TEC$Grupo, 
                               levels = c("Izquierdo_TEC", "Derecho_TEC"))



# Agregar la columna hemisferio con el valor "A" en todas las filas
datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = "NA")


#IDENTIFICANDO HEMISFERIOS NO AFECTADOS DE LOS PACIETNES TEC:
datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "1_DENI1005" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "2_KNOW1001" & Grupo == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "3_ALI0" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "4_BUTL" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "5_HAGG" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "6_HASTI007" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "7_BOAM" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "8_DANE0005" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "9_GREG" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "10_AITK" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "11_RANS0000" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "12_JONES004" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "13_PERR" & Grupo == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "14_SLAC" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "15_HEPPL010" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "16_RICHS010" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "17_KENT0007" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "18_STAN1002" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "19_MCDON022" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "20_PULL" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "21_MORR1002" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "22_PARK" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "23_HIGH" & Grupo == "Derecho_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "24_NOBL" & Grupo == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "25_COWL" & Grupo == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "26_KHAN" & Grupo == "Izquierdo_TEC", "A", Hemisferio))

datos_TEC <- datos_TEC %>%
  mutate(Hemisferio = ifelse(Persona == "27_NOLA" & Grupo == "Derecho_TEC", "A", Hemisferio))

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
grafico_mfARI <- ggplot(datos_totales, aes(x = Grupo, y = mfARI, fill = Grupo)) +
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

# ************************
# ****PREPROCESAMIENTO****
# ************************
DatasNuevos <- subset(datos_totales, select = -c(ARI, RoR, Tipo))


# Crear la columna Par para identificar los grupos pareados
datos_largo <- DatasNuevos %>%
  mutate(Par = sub("_.*", "", Persona))

# Dividir dataframe total en 2: para sanos y tec
sanos <- datos_largo %>%
  filter(grepl("Sano", Grupo)) %>%
  pivot_wider(names_from = Grupo, values_from = mfARI) %>%
  select(-Persona)  # Opcional: Eliminar columna Persona si ya no es necesaria

tec <- datos_largo %>%
  filter(grepl("TEC", Grupo)) %>%
  pivot_wider(names_from = Grupo, values_from = mfARI) %>%
  select(-Persona)  # Opcional: Eliminar columna Persona si ya no es necesaria

# Combinar dataframes para obtener la estructura adecuada y los sujetos pareados segun estado
datos_pareados <- sanos %>%
  inner_join(tec, by = "Par", suffix = c("_Sano", "_TEC"))


# Crear columna de pares para identificar los grupos correspondientes
datos_ancho <- DatasNuevos %>%
  mutate(Par = sub("_.*", "", Persona)) %>% 
  pivot_wider(names_from = Grupo, values_from = mfARI) 



#==========================================================
#==========================================================


# APLICACION ANOVA MEDIDAS REPETIDAS (no permite comprobar la esfericidad)
anova <- aov(formula = mfARI ~ Grupo + Error(Persona/Tipo), data = datos_totales)
print(summary(anova));

#==========================================================
# CONDICIONES PARA APLICAR ANOVA DE MEDIDAS REPETIDAS #
#==========================================================

# SE PROCEDE A REALIZAR UN ANALISIS DE VARIANZA MAS EXHAUSTIVO CON Anova():
datos_pareados_sin1fila <- as.matrix(datos_pareados[-1])
# Crear un modelo lineal multivariable. Los coeficientes estimados coinciden con la media de cada grupo.

modelo_lm <- lm(datos_pareados_sin1fila ~ 1)
# Se define el diseno de estudio: definir grupos
Grupo <- factor(c("Izquierdo_Sano", "Derecho_Sano", "Izquierdo_TEC", "Derecho_TEC"))

# **** TEST de ANOVA DE MUESTRAS REPETIDAS ****
anova_pareado <- Anova(modelo_lm, idata = data.frame(Grupo),
                       idesign = ~ Grupo, type = "III")
summary(anova_pareado, multivariate = F)


# 1. Verificar la normalidad de los residuos. nivel de significancia alpha=0.05
# H0: los residuos siguen una distribucion normal (p>0.05)
# H1: Los residuos no siguen una distribucion normal (p<=0.05)
# W: entre mas cercano a 1, indica que los datos tienen distribucion normal
residuos <- residuals(modelo_lm)
shapiro_test <- shapiro.test(residuos)
print(shapiro_test)
qqnorm(residuos)
qqline(residuos)
# conclusion: Se acepta la hipotesis nula H0. (p=0.4108)

# 2. Verificar esfericidad (p=0.039594 --> correccion Greenhouse.Geisser y Huyinh-Feldt --> p=7.017e-07)
# Comparaciones multiples (POST-HOC)
datos_pareados_tabla_larga <- gather(data = datos_pareados, key = "Grupo", value = "mfARI", 2:5)
posthoc <- pairwise.t.test(x = datos_pareados_tabla_larga$mfARI, g = datos_pareados_tabla_larga$Grupo,
                p.adjust.method = "holm", paired = TRUE, alternative = "two.sided")
print(posthoc);


# GRAFICOS VIOLIN Y DIFERENCIAS ENTRE GRUPOS
signif_results <- as.data.frame(as.table(posthoc$p.value)) %>%
  filter(!is.na(Freq)) %>%
  rename(Grupo1 = Var1, Grupo2 = Var2, p.value = Freq) %>%
  mutate(sig_level = case_when(
    p.value < 0.001 ~ "***",
    p.value < 0.01 ~ "**",
    p.value < 0.05 ~ "*",
    TRUE ~ ""
  )) %>%
  filter(sig_level != "")

datos_pareados_tabla_larga$Grupo <- factor(
  datos_pareados_tabla_larga$Grupo,
  levels = c("Izquierdo_Sano", "Derecho_Sano", "Izquierdo_TEC", "Derecho_TEC")
)


# Generar el gráfico de violín con niveles de significancia
grafico_mfARI_significancia <- ggplot(datos_pareados_tabla_larga, aes(x = Grupo, y = mfARI, fill = Grupo)) +
  geom_violin(trim = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.5, color = "black") +
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
  y_position = c(15, 16, 17, 18), # Ajusta estas posiciones según tu rango de datos
  textsize = 3)

print(grafico_mfARI_significancia)
#==========================================================
#==========================================================






# ******************************************************************************
# ******************************************************************************
# ************* ANALIZAR SI EXISTEN DIFERENCIAS SIGNIFICATIVAS ENTRE LOS *******
# ************* CON ETIQUETA VIVO Y MUERTO                               *******
# ******************************************************************************
# ******************************************************************************

# test de normalidad Shapiro-Wilk para subgrupos de pacientes TEC: vivo y muerto
shapiro.test(datos_TEC_HA_VM$mfARI[datos_TEC_HA_VM$Estado == "vivo"])
shapiro.test(datos_TEC_HA_VM$mfARI[datos_TEC_HA_VM$Estado == "muerto"])

# test para ver si existen diferencias significativas entre los etiqueta vivo y muerto:
mann_whitney_result <- wilcox.test(mfARI ~ Estado, data = datos_TEC_HA_VM)
print(mann_whitney_result)

medias_y_desv_tecs <- datos_TEC_HA_VM %>%
  group_by(Estado) %>%
  summarise(
    Media = mean(mfARI, na.rm = TRUE),
    Desviacion_Estandar = sd(mfARI, na.rm = TRUE)
  )

# Graficas violin
grafico_mfARI_estado <- ggplot(datos_TEC_HA_VM, aes(x = Estado, y = mfARI, fill = Estado)) +
  geom_violin(trim = FALSE) +
  geom_jitter(width = 0.2, alpha = 0.5, color = "black") +
  theme_minimal() +
  labs(title = "Comparación de mfARI entre estado vivo y muerto",
       x = "Estado", y = "mfARI") +
  scale_fill_brewer(palette = "Pastel1")

# Se agrega valores de media y dv a la imagen, esto por cada grafico violin
for(i in 1:nrow(medias_y_desv_tecs)) {
  grafico_mfARI_estado <- grafico_mfARI_estado +
    annotate("text", x = medias_y_desv_tecs$Estado[i], 
             y = max(datos_TEC_HA_VM$mfARI, na.rm = TRUE) - i, 
             label = paste("Media:", round(medias_y_desv_tecs$Media[i], 2), 
                           "\nSD:", round(medias_y_desv_tecs$Desviacion_Estandar[i], 2)),
             size = 4, color = "black", vjust = 4)
}

print(grafico_mfARI_estado)
