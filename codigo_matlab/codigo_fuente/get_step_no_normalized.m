function signal_escalon_unitario_inverso_final = get_step_no_normalized(Ts, order, cut_freq, PAM)
    escalonInverso=ones(150/Ts,1);
    escalonInverso(1:length(escalonInverso)/2)=escalonInverso(1:length(escalonInverso)/2)*max(PAM);%*max(PAM);
    escalonInverso(1+length(escalonInverso)/2:length(escalonInverso))=escalonInverso(1+length(escalonInverso)/2:length(escalonInverso))*min(PAM);%*min(PAM);
    [b,a] = butter(order,cut_freq);
    escalonInverso = filter(b,a,escalonInverso);
    escalonInverso=[escalonInverso,zeros(length(escalonInverso),1)];
    signal_eui = escalonInverso(:,1);
    
    % crear vector columna de 1024 instancias
    last_value = signal_eui(end); % ultimo valor a repetir para alcanzar las 1024 instancias necesarias para calcular la cwt
    rep_value = 1024-length(signal_eui); %cantidad necesario para alcanzar 1024 instancias en la senal del escalon inverso
    extras_values = last_value * ones(rep_value, 1); % se eliminaron los primeros 4 seg para que el escalon empezara desde una recta en y=1
    %extras_values = last_value * ones(999, 1); 
    % Agregar el vector que contiene el ultimo valor del escalon unitario
    % inverso (repetido 974 veces) al final del vector original signal_eui.
    % los 4 seg equivalen a 20 instacias, es por ello que repite el ultimo
    % valor 974 veces.
    escalon_inverso_unitario = [signal_eui; extras_values];
    %escalon_inverso_unitario = escalon_inverso_unitario(50:end);
    

    %GRAFICA
    % Se procede a cortar la senal de escalon inverso unitario a partir del
    % segundo 4:
    signal_escalon_unitario_inverso = escalon_inverso_unitario(326:end);%21
    expand_value = signal_escalon_unitario_inverso(end)*ones(325, 1);%20
    signal_escalon_unitario_inverso_final = [signal_escalon_unitario_inverso; expand_value];
    % Se corta la senal de escalon inverso en el segundo 8 aprox
    signal_escalon_unitario_inverso_final = signal_escalon_unitario_inverso_final(45:end);
    expand_value = signal_escalon_unitario_inverso_final(end)*ones(44, 1);
    signal_escalon_unitario_inverso_final = [signal_escalon_unitario_inverso_final; expand_value];
    

    % Crear el vector de tiempo
    t = (0:length(signal_escalon_unitario_inverso_final)-1) * Ts;
    
    
    % Graficar el escalon unitario inverso para ver el comportamiento teorico
    %figure;
    %hold on;
    %plot(t, signal_escalon_unitario_inverso_final, 'LineWidth', 2);
    %xlabel('Tiempo (s)');
    %ylabel('cm/s');
    %title('Escalón Unitario Inverso: Suavizado con Filtro Butterworth 2do orden - Freq de corte de 0.3 [Hz] y periodo de muestreo de 0.2 [seg]');
    %legend('Suavizado');
    %xlim([0 45]);
    %grid on;
    % Ajustar los marcadores del eje x para que vayan de 1 en 1 segundos
    %xticks(0:1:max(t));



end