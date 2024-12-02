% Inicializar una matriz para almacenar las señales

struct_type = StructPredSanos_NNM2d5OI_FD_LR0d001_E300_DO0d2; %*** modificar ***
num_senales = numel(struct_type); % Número total de señales
largo_senal = 1024; % Largo de cada señal
matriz_senales = zeros(num_senales, largo_senal); % Matriz para almacenar las señales

% Extraer las señales y almacenarlas en la matriz
for i = 1:num_senales
    matriz_senales(i, :) = struct_type(i).VSCd_Response; % IR CAMBIANDO HEMISFERIO
end

% Calcular el promedio a lo largo de las filas
senal_promedio = mean(matriz_senales, 1);
senal_promedio_rec = senal_promedio(762:end);

% Calcular la desviación estándar de la señal promedio
desviacion_estandar_total = std(senal_promedio_rec);

disp('DE:');
disp(desviacion_estandar_total);

% Crear los límites superior e inferior usando esta desviación estándar
limite_superior = senal_promedio_rec + desviacion_estandar_total;
limite_inferior = senal_promedio_rec - desviacion_estandar_total;




% Crear el eje temporal
delta_t = 0.2; % Intervalo de muestreo (segundos)
tiempo = (0:length(senal_promedio_rec)-1) * delta_t; % Eje temporal en segundos


signal_step_pre = struct_step_sanos_norm_noisemax2dot5onlyinput(1).signal_step(762:end);
signal_step = signal_step_pre';

figure;
plot(tiempo, senal_promedio_rec, 'k', 'LineWidth', 1.5); hold on;
plot(tiempo, limite_superior, 'LineStyle', '--', 'Color', [0.5, 0, 0.5], 'LineWidth', 0.6); % Línea límite superior
plot(tiempo, limite_inferior, 'b--', 'LineWidth', 0.6); % Línea límite inferior
plot(tiempo, signal_step, 'r', 'LineWidth', 1); % step
xlabel('Tiempo (seg)');
ylabel('Amplitud');
title('Respuesta vSC promedio \pm DE. Grupo sano HD');
legend('Respuesta vSCd Promedio', 'Respuesta vSCd + DE', 'Respuesta vSCd - DE', 'Escalón inverso');
grid on;
% Limitar la visualización del gráfico hasta los 35 segundos
xlim([5 35]);
hold off;
