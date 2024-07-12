function escalon_inverso_unitario = get_step(Ts, order, cut_freq)
    escalonInverso=ones(10/Ts,1);
    escalonInverso(1:length(escalonInverso)/2)=escalonInverso(1:length(escalonInverso)/2);%*max(PAM);
    escalonInverso(1+length(escalonInverso)/2:length(escalonInverso))=escalonInverso(1+length(escalonInverso)/2:length(escalonInverso))*0;%*min(PAM);
    [b,a] = butter(order,cut_freq);
    escalonInverso = filter(b,a,escalonInverso);
    escalonInverso=[escalonInverso,zeros(length(escalonInverso),1)];
    signal_eui = escalonInverso(:,1);
    
    % crear vector columna de 1024 instancias
    last_value = signal_eui(end); % ultimo valor a repetir para alcanzar las 1024 instancias necesarias para calcular la cwt
    extras_values = last_value * ones(974, 1); % se eliminaron los primeros 4 seg para que el escalon empezara desde una recta en y=1
    % Agregar el vector que contiene el ultimo valor del escalon unitario
    % inverso (repetido 974 veces) al final del vector original signal_eui.
    % los 4 seg equivalen a 20 instacias, es por ello que repite el ultima
    % valor 974 veces.
    escalon_inverso_unitario = [signal_eui; extras_values];
    
    % Crear el vector de tiempo
    t = (0:length(escalon_inverso_unitario)-1) * Ts;
    
    % Graficar el escalon unitario inverso 
    figure;
    hold on;
    plot(t, escalon_inverso_unitario, 'LineWidth', 2);
    xlabel('Tiempo (s)');
    ylabel('cm/s');
    title('Escalón Unitario Inverso: Suavizado con Filtro Butterworth 2do orden - Freq de corte de 0.3 [Hz] y periodo de muestreo de 0.2 [seg]');
    legend('Suavizado');
    ylim([-0.5, 1.1]);
    xlim([0 10]);
    grid on;
    % Ajustar los marcadores del eje x para que vayan de 0.5 en 0.5 segundos
    yticks(0:0.5:1);
    % Ajustar los marcadores del eje x para que vayan de 1 en 1 segundos
    xticks(0:1:max(t));
    
    
    % Se procede a cortar la senal de escalon inverso unitario a partir del
    % segundo 4:
    signal_escalon_unitario_inverso = escalon_inverso_unitario(21:end);
    expand_value = signal_escalon_unitario_inverso(end)*ones(20, 1);
    signal_escalon_unitario_inverso = [signal_escalon_unitario_inverso; expand_value];
    % Crear el vector de tiempo
    t = (0:length(signal_escalon_unitario_inverso)-1) * Ts;
    % Graficar el escalon unitario inverso para ver el comportamiento teorico
    figure;
    hold on;
    plot(t, signal_escalon_unitario_inverso, 'LineWidth', 2);
    xlabel('Tiempo (s)');
    ylabel('cm/s');
    title('Escalón Unitario Inverso: Suavizado con Filtro Butterworth 2do orden - Freq de corte de 0.3 [Hz] y periodo de muestreo de 0.2 [seg]');
    legend('Suavizado');
    ylim([-0.5, 1.1]);
    xlim([0 10]);
    grid on;
    % Ajustar los marcadores del eje y para que vayan de 0.5 en 0.5 segundos
    yticks(0:0.5:1);
    % Ajustar los marcadores del eje x para que vayan de 1 en 1 segundos
    xticks(0:1:max(t));



end