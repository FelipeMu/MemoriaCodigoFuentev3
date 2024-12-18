% Inicializar una matriz para almacenar las señales
struct_type = StructPredTecs_NNM2d5OI_FD_LR0d001_E300_DO0d2; % *** modificar ***
num_senales = numel(struct_type); % Número total de señales
largo_senal = 1024; % Largo de cada señal
matriz_senales = zeros(num_senales, largo_senal); % Matriz para almacenar las señales

% Extraer las señales y almacenarlas en la matriz
for i = 1:num_senales
    matriz_senales(i, :) = struct_type(i).VSCi_Response; % IR CAMBIANDO HEMISFERIO
end

% Calcular el promedio a lo largo de las filas
senal_promedio = mean(matriz_senales, 1);
senal_promedio_rec = senal_promedio(762:end);

% Calcular la desviación estándar para cada punto temporal (por columna)
desviacion_estandar_por_punto = std(matriz_senales(:, 762:end), 0, 1);

% Crear los límites superior e inferior usando esta desviación estándar
limite_superior = senal_promedio_rec + desviacion_estandar_por_punto;
limite_inferior = senal_promedio_rec - desviacion_estandar_por_punto;

% Crear el eje temporal
delta_t = 0.2; % Intervalo de muestreo (segundos)
tiempo = (0:length(senal_promedio_rec)-1) * delta_t; % Eje temporal en segundos

signal_step_pre = struct_step_sanos_norm_noisemax2dot5onlyinput(1).signal_step(762:end);
signal_step = signal_step_pre';

% Graficar los resultados
figure;
plot(tiempo, senal_promedio_rec, 'k', 'LineWidth', 1.5); hold on;
plot(tiempo, limite_superior, 'LineStyle', '--', 'Color', [0.5, 0, 0.5], 'LineWidth', 0.6); % Línea límite superior
plot(tiempo, limite_inferior, 'b--', 'LineWidth', 0.6); % Línea límite inferior
plot(tiempo, signal_step, 'r', 'LineWidth', 1); % step
xlabel('Tiempo (seg)');
ylabel('Amplitud');
title('Respuesta vSC promedio \pm DE. Grupo TEC HI');
legend('Respuesta vSCi Promedio', 'Respuesta vSCi + DE', 'Respuesta vSCi - DE', 'Escalón inverso');
grid on;
% Limitar la visualización del gráfico hasta los 35 segundos
xlim([5 35]);
hold off;
