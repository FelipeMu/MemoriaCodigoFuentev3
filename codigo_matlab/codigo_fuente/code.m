%######################################################
%############## PRE-PROCESAMIENTO #####################
%######################################################
% **** linea 243 aprox, se selecciona al individuo a estudiar ****

% Periodo de muestreo
ts = 0.2; % segundos

% Frecuencia de muestreo
fs = 1.0 / ts; % Hz

% Mostrar el periodo y frecuencia de muestreo
fprintf('Periodo de muestreo de señales: %.2f [seg]\n', ts);
fprintf('Frecuencia de muestreo de señales: %.2f [Hz]\n', fs);

% Arreglos para almacenar datos de señales
pam = []; % PAM: Presión Arterial Media
vsc = []; % VSC: Velocidad Sanguínea Cerebral

% Directorio que contiene archivos CSV
folder_csv = 'D:/TT/Memoria/waveletycnn/codigo_matlab_codigo_fuente/signals';

% Listar archivos en el directorio
files_csv = dir(fullfile(folder_csv, '*.csv'));

% Extraer nombres de los archivos
file_names = {files_csv.name};

% Mostrar los nombres de los archivos encontrados
fprintf('Archivos encontrados:\n');
disp(file_names);

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Crear lista de estructuras. cada estructura le corresponde a un individuo
% Directorio donde están los archivos CSV
carpeta_csv = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/signals';

% Obtener lista de archivos CSV en la carpeta
files_csv = dir(fullfile(carpeta_csv, '*.csv'));


% Estructura de wavelet: AMOR
structure_amor = struct('name_wavelet', 'amor', 'error', 0.0, 'complex_coeffs_amor', [], 'matrix_real', [], 'matrix_imag', [], 'scals_coeffs_amor', [], 'psif_amor', [], 'signal_vsc_rec', []);

% Estructura de wavelet: MORSE
structure_morse = struct('name_wavelet', 'morse', 'error', 0.0, 'signal_vsc_rec', []);

% Estructura de wavelet: BUMP
structure_bump = struct('name_wavelet', 'bump', 'error', 0.0, 'signal_vsc_rec', []);


% Inicializar el arreglo de estructuras con los campos necesarios
num_files = numel(files_csv); % Se almacena la cantidad de archivos csv leidos
signals(num_files) = struct('name_file', '', 'signal_pam', [], 'signal_vsc', [], 'struct_amor', structure_amor, 'struct_morse', structure_morse, 'struct_bump', structure_bump);
% Definir la estructura con cada uno de sus atributos, tomando en cuenta la
% cantidad de archivos encontrados (senales de invididuos) en la carpeta.
for j = 1:num_files
    signals(j) = struct('name_file', '', 'signal_pam', [], 'signal_vsc', [], 'struct_amor', structure_amor, 'struct_morse', structure_morse, 'struct_bump', structure_bump);
end
% Procesar cada archivo CSV
for idx = 1:num_files
    archivo_csv = files_csv(idx).name; % Nombre del archivo CSV
    ruta_archivo = fullfile(carpeta_csv, archivo_csv); % Ruta completa del archivo CSV
    
    % Leer el contenido del archivo CSV
    data = readmatrix(ruta_archivo); % Lee datos del archivo CSV
    
    % Separa las señales PAM y VSC, eliminando la primera fila (suponiendo encabezados)
    pam = data(:, 1); % Presion Arterial Media
    vsc = data(:, 2); % Velocidad Sanguinea Cerebral
    
    % Convertir a double y recortar a 1024 puntos
    pam = double(pam(1:min(end, 1024))); % Asegurar que no exceda el tamano
    vsc = double(vsc(1:min(end, 1024))); 

    % Asignar a la estructura
    signals(idx).name_file = archivo_csv; % Guardar el nombre del archivo
    signals(idx).signal_pam = pam; % Guardar la senal PAM
    signals(idx).signal_vsc = vsc; % Guardar la senal VSC
end

% Mostrar el contenido del arreglo de estructuras
fprintf('\n**** Mostrando el arreglo de estructuras de CSV ****\n');

% Verificar el contenido del arreglo de estructuras
for idx = 1:num_files
    dicc = signals(idx);
    fprintf('Nombre del archivo: %s\n', dicc.name_file);
    fprintf('Señal PAM: %s - N° de instancias: %d\n', mat2str(dicc.signal_pam), numel(dicc.signal_pam));
    fprintf('Señal VSC: %s - N° de instancias: %d\n', mat2str(dicc.signal_vsc), numel(dicc.signal_vsc));
    fprintf('----------------------------------------\n');
end
fprintf('----------------------------------------\n');


for i = 1:num_files
    disp(signals(i)); % Muestra la estructura de cada elemento
end
%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Recorrer el arreglo de estructuras 'signals'
for idx = 1:numel(signals)
    dicc = signals(idx); % Acceder a la estructura actual

    % Verificar si las longitudes de las señales PAM y VSC son diferentes
    if numel(dicc.signal_pam) ~= numel(dicc.signal_vsc)
        fprintf('Arreglos PAM y VSC tienen diferente longitud en el archivo %s.\n', dicc.name_file);
    end

    % Mostrar el nombre del archivo y el número de muestras
    fprintf('%s || Número de muestras: %d\n', dicc.name_file, numel(dicc.signal_pam));

    % Verificar si la longitud de la señal PAM es potencia de dos
    is_power_of_two(numel(dicc.signal_pam), 'señal PAM');

    % Verificar si la longitud de la señal VSC es potencia de dos
    is_power_of_two(numel(dicc.signal_vsc), 'señal VSC');

    fprintf('\n'); % Espacio entre salidas para claridad
end

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Aplicar CWT y espectros para cada senal (por ahora a las senales de VSC)
% // esto es solo para ver la calidad de la reconstruccion usando wt y icwt
for i = 1:numel(signals)
    s = signals(i);
    disp("Analizando archivo:");
    disp(s.name_file);
    signal_to_analyze = s.signal_vsc; % Senal para analizar
    % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor) Wavelet
    fb_amor = cwtfilterbank(SignalLength=length(signal_to_analyze),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_amor = freqz(fb_amor,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    signals(i).struct_amor.psif_amor = psif_amor; % Se guarda las respuestas de las frecuencias en el atributo de la estructura
    [coefs_amor,freqs_amor,~,scalcfs_amor] = wt(fb_amor,signal_to_analyze); % se aplica la transformada continua a la senal
    signals(i).struct_amor.complex_coeffs_amor = coefs_amor; % Se guarda la matriz de coeficientes en su respectivo atributo de la estructura
    signals(i).struct_amor.matrix_real = real(coefs_amor); % Se guarda la parte real de la matrix compleja de coeficientes
    signals(i).struct_amor.matrix_imag = imag(coefs_amor); % Se guarda la parte imaginaria de la matrix compleja de coeficientes
    signals(i).struct_amor.scals_coeffs_amor = scalcfs_amor; % Se guarda el vector de escalas en su respectivo atributo de la estructura
    xrecAN_amor = icwt(coefs_amor,[],ScalingCoefficients=scalcfs_amor,AnalysisFilterBank=psif_amor); % se realiza la transformada inversa continua de la senal
    xrecAN_amor = xrecAN_amor(:); % la reconstruccion de la senal se pasa a formato vector columna
    signals(i).struct_amor.signal_vsc_rec = xrecAN_amor;  % Se guarda signal_rec en su respectiva estructura
    errorAN_amor = get_nmse(signal_to_analyze, signals(i).struct_amor.signal_vsc_rec); % se calcula el nmse
    signals(i).struct_amor.error = errorAN_amor; % se almacena el respectivo nmse en la estructura de la senal analizada

    %###########################################################################################
    % Crear vector que representa los tiempos en los que se toma una muestra
    tms = (0:numel(signal_to_analyze)-1)/fs;
    % Llamada a funcion para mostrar grafica de la senal y su respectivo
    % escalograma
    plot_signal_and_scalogram(tms, signal_to_analyze, freqs_amor, coefs_amor, 'amor',s.name_file)
 end 

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################


% Ejemplo de como crear una figura con multiples secciones para comparar senales y mostrar errores

% Crear una nueva figura
figure;
% Crear subplots para la señal original y las reconstrucciones con amor, morse, bump
% y mostrar el error NMSE en el costado
for i = 1:num_files
    s = signals(i); % Obtenemos la estructura correspondiente a la senal actual

    % Definir el índice base para los subplots (para separar las señales)
    base_idx = (i - 1) * 2;

    % Crear subplots para comparar las senales originales con las reconstruidas por amor
    subplot(num_files, 3, base_idx + 1);
    hold on;
    plot(s.signal_vsc, 'b'); % Senal original
    plot(s.struct_amor.signal_vsc_rec, 'r--'); % Senal reconstruida por amor
    title(sprintf('Señal VSC vs Amor (NMSE: %.2e) [%s]', s.struct_amor.error, s.name_file));
    xlabel('Tiempo');
    ylabel('Amplitud');
    legend('Original', 'Reconstruida (amor)');
    hold off;
end

%###########################################################################################################
%###########################################################################################################
%###########################################################################################################

% BUSQUEDA EL MINIMO NMSE PARA CADA WAVELET, TENIENDO EN CUENTA LOS
% HIPERPARAMETROS QUE UTILIZAN:
% Para encontrar el minimo error, se procede a utilizar las siguientes
% funciones. En ellas se realizan todas las combinaciones posibles de los
% hiperparametros TimeBandwidth y VoicesPerOctave para encontrar el error
% minimo de cada wavelet madre
min_error_amor_bump(signals(1).signal_vsc, "bump"); % para wavelet AMOR y BUMP
min_error_morse(signals(1).signal_vsc); % para wavelet MORSE





%###########################################################################################################
%####################### Aplicacion de ruido Gaussiano y filtro Butterworth ################################
%###########################################################################################################

% Se procede aplicar ruido gaussiano con un coeficiente de variacion entre
% [5%, 10%], y posteriormente un filtro Butterworth de octavo orden, con
% frecuencia de corte de 0.25 Hz. 
%=====================================================================================================
% apply_noise_and_filter(estructura signals, frecuencia de muestreo, [#cantidad] de senales con ruido)
%=====================================================================================================
apply_noise_and_filter(signals, fs, 50);







%###########################################################################################################
%############################ Preparacion de inputs para la red ############################################
%###########################################################################################################


% Por cada senal pam & vsc se debe aplicar CWT, esto con el fin de
% obtener los inputs para el entrenamiento de la red.

path_1 = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/signals_noises';
% Obtener los nombres de las carpetas dentro del directorio
folder_structs = dir(path_1);
folder_names = {folder_structs([folder_structs.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names = setdiff(folder_names, {'.', '..'});

%##############################
% Elegir al sujeto de prueba
%##############################
file_person = folder_names(1); % ************* se tiene el nombre de la carpeta: ejemplo G2x001 *************
fprintf("Individuo a analizar")
disp(file_person)
file_pam = fullfile(path_1, file_person); % Se concatena al path_1 (path general), el nombre de la carpeta del sujeto


% Obtener los nombres de las carpetas dentro del directorio del sujeto que
% se va a estudiar
folder_structs2 = dir(file_pam{1}); % Se obtiene la ruta2 como string
folder_names2 = {folder_structs2([folder_structs2.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names2 = setdiff(folder_names2, {'.', '..'}); % Se almacenan los nombres de PAMnoises y VSCnoises

% Se obtiene el directorio de la carpeta que almacena las senales PAM con
% ruido:
path_pam_noises = fullfile(file_pam{1}, folder_names2{1}); % directorio de PAMnoises

% Se obtiene el directorio de la carpeta que almacena las senales VSC con
% ruido:
path_vsc_noises = fullfile(file_pam{1}, folder_names2{2}); % directorio de PAMnoises


% Ahora se deben extraer todos los archivos .csv del directorio
% path_pam_noises:

% Obtener lista de archivos CSV en la carpeta de PAMnoises
pam_noises_csv = dir(fullfile(path_pam_noises, '*.csv'));

% Obtener lista de archivos CSV en la carpeta de VSCnoises
vsc_noises_csv = dir(fullfile(path_vsc_noises, '*.csv'));

% Cantidad de csvs encontrados en el directorio path_pam_noises
num_csv = numel(pam_noises_csv); % Se almacena la cantidad de archivos csv leidos



% Crear estructura que guardara cada par de PAM y VSC con ruido
struct_noises(num_csv) = struct('name_signal', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vsc_noise', [], 'matrix_complex_vsc', [], 'scalscfs_vsc_noise', [], 'psif_vsc_noise', []);

% Se crean tantas instancias de la estructura como archivos csv encontrados
% en la carpeta
for j = 1:num_csv
    struct_noises(j) = struct('name_signal', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vsc_noise', [], 'matrix_complex_vsc', [], 'scalscfs_vsc_noise', [], 'psif_vsc_noise', []);
end


% *********************************************************************
% Bucle para almacenar las senales con ruido en una estructura unica:
% *********************************************************************
for index = 1:num_csv

    file2_csv = pam_noises_csv(index).name; % Nombre del archivo  PAMnoises
    file2_csv_vsc = vsc_noises_csv(index).name; % Nombre del archivo VSCnoises
    
    path_pam_file2 = fullfile(path_pam_noises, file2_csv); % Ruta completa de la carpeta PAMnoises
    path_vsc_file2 = fullfile(path_vsc_noises, file2_csv_vsc); % Ruta completa de la carpeta VSCnoises

    % Leer el contenido del archivo CSV
    data2_pam_noises = readmatrix(path_pam_file2); % Lee datos del archivo CSV
    % Leer el contenido del archivo CSV
    data2_vsc_noises = readmatrix(path_vsc_file2); % Lee datos del archivo CSV
    
    % Separa las señales PAM y VSC, eliminando la primera fila (suponiendo encabezados)
    pam_noise = data2_pam_noises(:, 1); % Presion Arterial Media con ruido
    vsc_noise = data2_vsc_noises(:, 1); % Velocidad Sanguinea Cerebral con ruido

    % Asignar a la estructura
    struct_noises(index).name_signal = ['Ruido', num2str(index)]; % Guardar el nombre del archivo
    struct_noises(index).pam_noise = pam_noise; % Guardar la senal PAM con ruido
    struct_noises(index).vsc_noise = vsc_noise; % Guardar la senal VSC con ruido

    %########################################
    %######## Aplicacion de CWT #############
    %########################################

    % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor) Wavelet
    
    % [PAM NOISE - CWT]
    filters_bank_pam_noise = cwtfilterbank(SignalLength=length(struct_noises(index).pam_noise),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_pam_noise = freqz(filters_bank_pam_noise,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs_pam_noise,freqs_pam_noise,~,scalcfs_pam_noise] = wt(filters_bank_pam_noise,struct_noises(index).pam_noise); % se aplica la transformada continua a la senal
    
    % [VSC NOISE - CWT]
    filters_bank_vsc_noise = cwtfilterbank(SignalLength=length(struct_noises(index).vsc_noise),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_vsc_noise = freqz(filters_bank_vsc_noise,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs_vsc_noise,freqs_vsc_noise,~,scalcfs_vsc_noise] = wt(filters_bank_vsc_noise,struct_noises(index).vsc_noise); % se aplica la transformada continua a la senal
    
    
    % Almacenando nueva informacion en la respectiva estructura de senales
    % con ruido:
    
    % Almacenando coeficientes (matriz compleja)
    struct_noises(index).matrix_complex_pam = coefs_pam_noise; % pam
    struct_noises(index).matrix_complex_vsc = coefs_vsc_noise; % vsc

    % Almacenando escalas de coeficientes (vector 1D real en fila, largo 1024)
    struct_noises(index).scalscfs_pam_noise = scalcfs_pam_noise; % pam
    struct_noises(index).scalscfs_vsc_noise = scalcfs_vsc_noise; % vsc

    % Almacenando respuestas de filtros (matriz real 30x1024)
    struct_noises(index).psif_pam_noise = psif_pam_noise; % pam
    struct_noises(index).psif_vsc_noise = psif_vsc_noise; % vsc

end



% Almacenar matrices complejas pam y vsc en carpetas especificas para 
% luego trabajar con la red profunda en python. Para ello se importan 
% las matrices en formato.mat y luego en python se utiliza un script
% para transformar dicho formato a npy.

% Directorios para guardar los archivos .mat
pam_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_pam_mat';
vsc_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/matrices_complejas_vsc_mat';

% Crear los directorios si no existen
if ~exist(pam_dir, 'dir')
    mkdir(pam_dir);
end
if ~exist(vsc_dir, 'dir')
    mkdir(vsc_dir);
end
%################################################
% Guardar las matrices complejas en archivos .mat
%################################################
for i = 1:num_csv
    % Guardar matriz_complex_pam
    matrix_complex_pam = struct_noises(i).matrix_complex_pam;
    save(fullfile(pam_dir, sprintf('matrix_complex_pam_noise_%d.mat', i)), 'matrix_complex_pam');
    
    % Guardar matriz_complex_vsc
    matrix_complex_vsc = struct_noises(i).matrix_complex_vsc;
    save(fullfile(vsc_dir, sprintf('matrix_complex_vsc_noise_%d.mat', i)), 'matrix_complex_vsc');
end





%##################################################################
%##################################################################
% Bucle para obtener la matriz de coeficientes de cada senal
% original PAM (de cada individuo). Porteriormente se hara una 
% prediccion de la senal VSC por medio de la red ya entrenada
% (U-net)
%##################################################################
%##################################################################


% Directorios para guardar los archivos.mat asociados a los inputs de coeficientes 
inputs_coefs_dir = 'D:/TT/Memoria/waveletycnn/codigo_python/inputs_coeficientes';

% Crear los directorios si no existen
if ~exist(inputs_coefs_dir, 'dir')
    mkdir(inputs_coefs_dir);
end

% Aplicar CWT y obtener coeficientes (matriz compleja) para predecir los
% coeficientes de la senal VSC por medio de la red neuronal U-net
for index = 1:numel(signals)
    signal_to_predict = signals(index).signal_pam;
    filters_bank_pam_predict = cwtfilterbank(SignalLength=length(signal_to_predict),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif_pam_noise = freqz(filters_bank_pam_predict,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs_pam_to_predict,freqs_pam_to_predict,~,scalcfs_pam_to_predict] = wt(filters_bank_pam_predict,signal_to_predict); % se aplica la transformada continua a la senal
    
    % Guardar coeficientes (matriz compleja de senal PAM)
    save(fullfile(inputs_coefs_dir, sprintf('matrix_complex_pam_to_predict_%d.mat', index)), 'coefs_pam_to_predict');
end




%###################################################################
%###################################################################
% Se calcula la ICWT de la senal VSC con la matriz d compleja
% (coeficientes) que predijo la red u-net
%###################################################################
%###################################################################

% Se obtienes 2 de los 3 parametros para calcular ICWT de la senal VSC. El
% terer parametro (mas importante) corresponde a los coeficientes o
% matriz compleja que se obtiene de la prediccion de la red U-net
signal_vsc_to_predict = signals(1).signal_vsc;
filters_bank_vsc_to_predict = cwtfilterbank(SignalLength=length(signal_vsc_to_predict),Boundary="periodic", Wavelet="amor",SamplingFrequency=5,VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
psif_vsc_to_predict = freqz(filters_bank_vsc_to_predict,FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
[coefs_vsc_signal_original,freqs_vsc_original_to_predict,~,scalcfs_vsc_original_to_predict] = wt(filters_bank_vsc_to_predict, signal_vsc_to_predict); % se aplica la transformada continua a la senal


%###########################################################
%############# lectura coeficientes predichos ##############
%###########################################################
% Especificar el directorio donde esta el archivo .mat
coefs_predicted_dir = 'D:/TT/Memoria/waveletycnn/codigo_matlab/codigo_fuente/coefs_vsc_predicted';

% Nombre del archivo .mat
coefs_predicted_filename = 'coefs_vsc_predicted';

% Ruta completa del archivo .mat
coefs_predicted_path = fullfile(coefs_predicted_dir, coefs_predicted_filename);

% Cargar el archivo .mat y guardar el contenido en una variable
coefs_vsc_predicted_by_unet_struct = load(coefs_predicted_path);
% see extrae la matriz compleja
coefs_vsc_predicted_by_unet = cast(coefs_vsc_predicted_by_unet_struct(1).complex_matrix_vsc, 'double');


% Se procede a aplicar la ICWT con el uso de los coeficientes predichos por
% la red U-net:
 get_signal_vsc_estimated_with_coefs_unet = icwt(coefs_vsc_predicted_by_unet,[], ScalingCoefficients = scalcfs_vsc_original_to_predict, AnalysisFilterBank = psif_vsc_to_predict); % se realiza la transformada inversa continua de la senal
 get_signal_vsc_estimated_with_coefs_unet = get_signal_vsc_estimated_with_coefs_unet(:); % la reconstruccion de la senal estimada por los coeficientes predichos se pasa a formato vector columna
 error_signal_original_and_predicted = get_nmse(signal_vsc_to_predict, get_signal_vsc_estimated_with_coefs_unet); % se calcula el nmse


 % comparar senales
figure;
hold on;
plot(signal_vsc_to_predict, 'b'); % Senal original
plot( get_signal_vsc_estimated_with_coefs_unet, 'r--'); % Senal reconstruida por amor
title(sprintf('Señal VSC: Original vs Predicha (NMSE: %.2e)', error_signal_original_and_predicted));
xlabel('Tiempo');
ylabel('Amplitud');
legend('Original', 'Reconstruida');
hold off;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% codigo final 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPROCESAMIENTO DE LAS SENALES: PAM | VSC DERECHA | VSC IZQUIERDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Se establece la cantidad de senales con ruido que se crearan para el posterior entrenamiento de la red profunda U-NET 
num_of_noises_signals = 50;

%=============================
%=== LECTURA DE DLS: SANOS ===
%=============================

% Directorios y configuracion
sourceDirectory_sano = 'D:\TT\Memoria\Memoria_Codigo_Fuente\codigo_matlab\codigo_fuente\LDS\DATOS_SANOS_PAR';
destinationDirectory_sano = 'D:\TT\Memoria\Memoria_Codigo_Fuente\codigo_matlab\codigo_fuente\signals_LDS\SANOS';

% Obtener la lista de archivos .PAR en el directorio
fileList_sano = dir(fullfile(sourceDirectory_sano, '*.PAR'));
num_healthy_people = numel(fileList_sano);

% Inicializar las estructuras
structure_original_PAM(1) = struct('matrix_complex_original_pam', [], 'scalscfs_original_pam', [], 'psif_original_pam', [], 'freqs_original_pam', []);
structure_original_VSCd(1) = struct('matrix_complex_original_vscd', [], 'scalscfs_original_vscd', [], 'psif_original_vscd', [], 'freqs_original_vscd', []);
structure_original_VSCi(1) = struct('matrix_complex_original_vsci', [], 'scalscfs_original_vsci', [], 'psif_original_vsci', [], 'freqs_original_vsci', []);

structure_vscd_noises_sano(num_of_noises_signals) = struct('name_file_noise', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [], 'freqs_pam_noise', [], 'vscd_noise', [], 'matrix_complex_vscd', [], 'scalscfs_vscd_noise', [], 'psif_vscd_noise', [], 'freqs_vscd_noise', []);
structure_vsci_noises_sano(num_of_noises_signals) = struct('name_file_noise', '', 'pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [], 'freqs_pam_noise', [], 'vsci_noise', [], 'matrix_complex_vsci', [], 'scalscfs_vsci_noise', [], 'psif_vsci_noise', [], 'freqs_vsci_noise', []);
struct_lds_sano(num_healthy_people) = struct('name_file', '', 'signal_pam', [], 'signal_vscd', [], 'signal_vsci', [], 'struct_VSCd_noises', structure_vscd_noises_sano, 'struct_VSCi_noises', structure_vsci_noises_sano, 'struct_original_PAM', [], 'struct_original_VSCd', [], 'struct_original_VSCi', []);

temp = 0; % activador para considerar que la senal PAM es la columna 2 (solo para el sujeto 27_HC036101.PAR)
% Recorrer cada archivo
for i = 1:num_healthy_people
    % Obtener nombre del archivo y crear la ruta completa
    filename_sano = fileList_sano(i).name;
    filepath_sano = fullfile(sourceDirectory_sano, filename_sano);
    
    % Determinar el delimitador segun el nombre del archivo
    if strcmp(filename_sano, '22_ALSA.PAR')
        delimiter_sano = '\t';
        columnsToRead_sano = 3:5;
    elseif strcmp(filename_sano, '27_HC036101.PAR')
        delimiter_sano = ' ';
        columnsToRead_sano = 3:5;
        temp = 1; % se activa para considerar que la senal PAM es la columna 2 y no la 3 como en el resto de los sujetos sanos
    else
        delimiter_sano = ' ';
        columnsToRead_sano = 4:6;
    end
    
    % Leer el archivo de texto
    data_sano= readmatrix(filepath_sano, 'FileType', 'text', 'Delimiter', delimiter_sano);
    % Extraer las columnas especificas (primeras 1024 filas)
    columnas_sano = data_sano(1:1024, columnsToRead_sano);
    
    % Crear la carpeta para el archivo
    folderName_sano = fullfile(destinationDirectory_sano, erase(filename_sano, '.PAR'));
    if ~exist(folderName_sano, 'dir')
        mkdir(folderName_sano);
    end
    
    % Guardar los datos en la estructura
    struct_lds_sano(i).name_file = erase(filename_sano, '.PAR');

    if temp == 1
        struct_lds_sano(i).signal_pam = columnas_sano(:, 2); % PAM es la columna 2 para 23_ALSA
        struct_lds_sano(i).signal_vscd = columnas_sano(:, 1); % VSCd es la columna 1 23_ALSA
        struct_lds_sano(i).signal_vsci = columnas_sano(:, 3); % VSCi es la tercera columna para 23_ALSA
        temp = 0;
    else
        struct_lds_sano(i).signal_pam = columnas_sano(:, 3); % PAM es la columna 3 para los los sanos restantes
        struct_lds_sano(i).signal_vscd = columnas_sano(:, 1); % VSCd es la columna 1 para los sanos restantes
        struct_lds_sano(i).signal_vsci = columnas_sano(:, 2); % VSCi es la columna 2 para los sanos restantes
    end
    % Guardar las señales en archivos CSV dentro de la carpeta correspondiente
    pam_sano_csv_path = fullfile(folderName_sano, 'signal_pam.csv');
    vscd_sano_csv_path = fullfile(folderName_sano, 'signal_vscd.csv');
    vsci_sano_csv_path = fullfile(folderName_sano, 'signal_vsci.csv');
    
    writematrix(struct_lds_sano(i).signal_pam, pam_sano_csv_path);
    writematrix(struct_lds_sano(i).signal_vscd, vscd_sano_csv_path);
    writematrix(struct_lds_sano(i).signal_vsci, vsci_sano_csv_path);
end




%===========================
%=== LECTURA DE DLS: TEC ===
%===========================

% Directorios y configuracion
sourceDirectory_tec = 'D:\TT\Memoria\Memoria_Codigo_Fuente\codigo_matlab\codigo_fuente\LDS\DATOS_TEC_PAR';
destinationDirectory_tec = 'D:\TT\Memoria\Memoria_Codigo_Fuente\codigo_matlab\codigo_fuente\signals_LDS\TEC';

% Obtener la lista de archivos .PAR en el directorio
fileList_tec = dir(fullfile(sourceDirectory_tec, '*.PAR'));
num_tec_people = numel(fileList_tec);

% Inicializar las estructuras
structure_vscd_noises_tec(num_of_noises_signals) = struct('pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vscd_noise', [], 'matrix_complex_vscd', [], 'scalscfs_vscd_noise', [], 'psif_vscd_noise', []);
structure_vsci_noises_tec(num_of_noises_signals) = struct('pam_noise', [], 'matrix_complex_pam', [], 'scalscfs_pam_noise', [], 'psif_pam_noise', [],  'vsci_noise', [], 'matrix_complex_vsci', [], 'scalscfs_vsci_noise', [], 'psif_vsci_noise', []);
struct_lds_tec(num_tec_people) = struct('name_file', '', 'signal_pam', [], 'signal_vscd', [], 'signal_vsci', [], 'struct_VSCd_noises', structure_vscd_noises_tec, 'struct_VSCi_noises', structure_vsci_noises_tec, 'struct_original_PAM', [], 'struct_original_VSCd', [], 'struct_original_VSCi', []);

temp = 0; % activador para considerar que la senal que el archivo 6_HASTI007 tiene como delimitador un '\t' y no un ' ' como los otros archivos
% Recorrer cada archivo
for i = 1:num_tec_people
    % Obtener nombre del archivo y crear la ruta completa
    filename_tec = fileList_tec(i).name;
    filepath_tec = fullfile(sourceDirectory_tec, filename_tec);
    
    % Determinar el delimitador segun el nombre del archivo
    if strcmp(filename_tec, '6_HASTI007.PAR')
        delimiter_tec = '\t';
        columnsToRead_tec = 1:3;
    else
        delimiter_tec = ' ';
        columnsToRead_tec = 2:4;
    end
    
    % Leer el archivo de texto
    data_tec = readmatrix(filepath_tec, 'FileType', 'text', 'Delimiter', delimiter_tec);
    % Extraer las columnas especificas (primeras 1024 filas)
    columnas_tec = data_tec(1:1024, columnsToRead_tec);
    
    % Crear la carpeta para el archivo
    folderName_tec = fullfile(destinationDirectory_tec, erase(filename_tec, '.PAR'));
    if ~exist(folderName_tec, 'dir')
        mkdir(folderName_tec);
    end
    
    % Guardar los datos en la estructura
    struct_lds_tec(i).name_file = erase(filename_tec, '.PAR');

    % se guardan columnas especificas en la respectiva estructura del
    % paciente:
    struct_lds_tec(i).signal_pam = columnas_tec(:, 2); % PAM: 2 columna
    struct_lds_tec(i).signal_vscd = columnas_tec(:, 1); % VSCd derecho (chanel 1): columna 1
    struct_lds_tec(i).signal_vsci = columnas_tec(:, 3); % VSCi izquierdo (chanel 2): columna 3

    % Guardar las señales en archivos CSV dentro de la carpeta correspondiente
    pam_tec_csv_path = fullfile(folderName_tec, 'signal_pam.csv');
    vscd_tec_csv_path = fullfile(folderName_tec, 'signal_vscd.csv');
    vsci_tec_csv_path = fullfile(folderName_tec, 'signal_vsci.csv');
    
    writematrix(struct_lds_tec(i).signal_pam, pam_tec_csv_path);
    writematrix(struct_lds_tec(i).signal_vscd, vscd_tec_csv_path);
    writematrix(struct_lds_tec(i).signal_vsci, vsci_tec_csv_path);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APLICACIÓN DE RUIDO GAUSSIANO: PAM | VSC DERECHA | VSC IZQUIERDA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Periodo de muestreo
ts = 0.2; % segundos

% Frecuencia de muestreo
fs = 1.0 / ts; % Hz

% Aplicacion de ruido gaussinano y filtro octavo orden a las senales PAM,
% VSCd y VSCi tanto de sujetos sanos como de pacientes TEC:
apply_noise_and_filter_dls(struct_lds_sano, struct_lds_tec, fs, num_of_noises_signals);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% APLICACIÓN DE CWT Y OBTENCIÓN DE COEFICIENTES: PAM | VSC DERECHA | VSC IZQUIERDA %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%                                                   SUJETO SANO
%===============================================================================================================================
% Se obtienen los nombres de todas las carpetas existentes en el direccion
% path_sano: 27 sujetos en total
path_sano = 'D:/TT/Memoria/Memoria_Codigo_Fuente/codigo_matlab/codigo_fuente/signals_LDS/SANOS';
% Obtener los nombres de las carpetas dentro del directorio
folder_structs_sano = dir(path_sano);
folder_names_sano = {folder_structs_sano([folder_structs_sano.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names_sano = setdiff(folder_names_sano, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de sujetos sanos

% Se obtienen los nombres de todas las carpetas existentes en el direccion
% path_files_signals. Carpetas a seleccionar --> PAMnoises, VSCdnoises,
% VSCinoises
path_files_signals_sano = 'D:/TT/Memoria/Memoria_Codigo_Fuente/codigo_matlab/codigo_fuente/signals_LDS/SANOS/1_HEMU'; % Se elige 1_HEMU al azar, todos los sujetos tienen las mismas carpetas asociadas a las senales con ruido
% Obtener los nombres de las carpetas dentro del directorio
folder_signals_sano = dir(path_files_signals_sano);
folder_names_signals_sano = {folder_signals_sano([folder_signals_sano.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names_signals_sano = setdiff(folder_names_signals_sano, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de sujetos sanos
%===============================================================================================================================



%                                                   PACIENTE TEC
%===============================================================================================================================
% Se obtienen los nombres de todas las carpetas existentes en el direccion
% path_sano: 27 sujetos en total
path_tec = 'D:/TT/Memoria/Memoria_Codigo_Fuente/codigo_matlab/codigo_fuente/signals_LDS/TEC';
% Obtener los nombres de las carpetas dentro del directorio
folder_structs_tec = dir(path_tec);
folder_names_tec = {folder_structs_tec([folder_structs_tec.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names_tec = setdiff(folder_names_tec, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de sujetos sanos

% Se obtienen los nombres de todas las carpetas existentes en el direccion
% path_files_signals. Carpetas a seleccionar --> PAMnoises, VSCdnoises,
% VSCinoises
path_files_signals_tec = 'D:/TT/Memoria/Memoria_Codigo_Fuente/codigo_matlab/codigo_fuente/signals_LDS/TEC/1_DENI1005'; % Se elige 1_DENI1005 al azar, todos los sujetos tienen las mismas carpetas asociadas a las senales con ruido
% Obtener los nombres de las carpetas dentro del directorio
folder_signals_tec = dir(path_files_signals_tec);
folder_names_signals_tec = {folder_signals_tec([folder_signals_tec.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
folder_names_signals_tec = setdiff(folder_names_signals_tec, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de sujetos sanos
%===============================================================================================================================


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% PARA SUJETOS SANOS Y PACIENTES TEC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for index = 1:numel(folder_names_sano)
    %=========================================================================================================================
    % PARA SUJETO SANO
    % ************* se tiene el nombre de la carpeta: 10_MIMO ************* index=1
    file_person_sano = folder_names_sano(index); %SANO
    % directorio: carpeta del sujeto por analizar
    file_pam_sano = fullfile(path_sano, file_person_sano); %SANO
    %=========================================================================================================================
    % PARA PACIENTE TEC
    % ************* se tiene el nombre de la carpeta:  ************* index=1
    file_person_tec = folder_names_tec(index); %TEC
    % directorio: carpeta del sujeto por analizar
    file_pam_tec = fullfile(path_tec, file_person_tec); %TEC 


    %=========================================================================================================================
    % PARA SUJETO SANO
    % Por cada sujeto se accede a sus carpetas de PAMnoises, VSCdnoises y
    % VSCinoises (se obtienen los path de las carpetas). Ademas se obtienen
    % los archivos .csv existentes en cada una de estas carpetas:
    
    % Se obtiene el directorio de la carpeta que almacena las senales PAM con
    % ruido del sujeto index:
    path_pam_noises_sano = fullfile(file_pam_sano{1}, folder_names_signals_sano{1}); % directorio de PAMnoises
 
    % Se obtiene el directorio de la carpeta que almacena las senales VSCd con
    % ruido:
    path_vscd_noises_sano = fullfile(file_pam_sano{1}, folder_names_signals_sano{2}); % directorio de VSCdnoises
 
    % Se obtiene el directorio de la carpeta que almacena las senales VSCd con
    % ruido:
    path_vsci_noises_sano = fullfile(file_pam_sano{1}, folder_names_signals_sano{3}); % directorio de VSCinoises
    %=========================================================================================================================
    % PARA PACIENTE TEC
    % Por cada paciente se accede a sus carpetas de PAMnoises, VSCdnoises y
    % VSCinoises (se obtienen los path de las carpetas). Ademas se obtienen
    % los archivos .csv existentes en cada una de estas carpetas:
    
    % Se obtiene el directorio de la carpeta que almacena las senales PAM con
    % ruido del sujeto index:
    path_pam_noises_tec = fullfile(file_pam_tec{1}, folder_names_signals_tec{1}); % directorio de PAMnoises
 
    % Se obtiene el directorio de la carpeta que almacena las senales VSCd con
    % ruido:
    path_vscd_noises_tec = fullfile(file_pam_tec{1}, folder_names_signals_tec{2}); % directorio de VSCdnoises
 
    % Se obtiene el directorio de la carpeta que almacena las senales VSCd con
    % ruido:
    path_vsci_noises_tec = fullfile(file_pam_tec{1}, folder_names_signals_tec{3}); % directorio de VSCinoises


    %=========================================================================================================================
    % PARA SUJETO SANO
    % Ahora se deben extraer todos los archivos .csv del directorio
    % path_pam_noises:
    
    % Obtener lista de archivos CSV en la carpeta de PAMnoises
    pam_noises_csv_sano = dir(fullfile(path_pam_noises_sano, '*.csv'));
    
    % Obtener lista de archivos CSV en la carpeta de VSCnoises
    vscd_noises_csvd_sano = dir(fullfile(path_vscd_noises_sano, '*.csv'));
    
    % Obtener lista de archivos CSV en la carpeta de VSCnoises
    vsci_noises_csvi_sano = dir(fullfile(path_vsci_noises_sano, '*.csv'));
    

    % Cantidad de csvs encontrados en el directorio path_pam_noises (total esperado: 50 )
    num_csv = numel(pam_noises_csv_sano); % Se almacena la cantidad de archivos csv leidos
    %=========================================================================================================================
     % PARA PACIENTE TEC
    % Ahora se deben extraer todos los archivos .csv del directorio
    % path_pam_noises:
    
    % Obtener lista de archivos CSV en la carpeta de PAMnoises
    pam_noises_csv_tec = dir(fullfile(path_pam_noises_tec, '*.csv'));
    
    % Obtener lista de archivos CSV en la carpeta de VSCnoises
    vscd_noises_csvd_tec = dir(fullfile(path_vscd_noises_tec, '*.csv'));
    
    % Obtener lista de archivos CSV en la carpeta de VSCnoises
    vsci_noises_csvi_tec = dir(fullfile(path_vsci_noises_tec, '*.csv'));
    %=========================================================================================================================


    fprintf("[%i] Pareo sano y tec:\n", index);
    fprintf("%s <-> %s\n",file_person_sano{1}, file_person_tec{1});
    for j = 1:num_csv
        %%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% SUJETO SANO %%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%
        %==========================================================================================================================================
        name_csv_pam_sano = pam_noises_csv_sano(j).name; % Nombre del archivo.csv PAMnoises
        name_csv_vscd_sano = vscd_noises_csvd_sano(j).name; % Nombre del archivo.csv VSCdnoises
        name_csv_vsci_sano = vsci_noises_csvi_sano(j).name; % Nombre del archivo.csv VSCinoises
        % Verificar tripleta de datos vinculados (ej: 10_MIMO_ruidoPAM1.csv | 10_MIMO_ruidoVSCd1.csv | 10_MIMO_ruidoVSCi1.csv)
        %fprintf("%s | %s | %s", name_csv_pam_sano, name_csv_vscd_sano, name_csv_vsci_sano); 
    
        path_pam_noises_sanos = fullfile(path_pam_noises_sano, name_csv_pam_sano); % Ruta completa de la carpeta PAMnoises
        path_vscd_noises_sanos = fullfile(path_vscd_noises_sano, name_csv_vscd_sano); % Ruta completa de la carpeta VSCdnoises
        path_vsci_noises_sanos = fullfile(path_vsci_noises_sano, name_csv_vsci_sano); % Ruta completa de la carpeta VSCinoises


        % Leer el contenido del archivo CSV en carpeta PAMnoises
        data_pam_noises_sanos = readmatrix(path_pam_noises_sanos);
        % Leer el contenido del archivo CSV en la carpeta VSCdnoises
        data_vscd_noises_sanos = readmatrix(path_vscd_noises_sanos);
        % Leer el contenido del archivo CSV en la carpeta VSCinoises
        data_vsci_noises_sanos = readmatrix(path_vsci_noises_sanos); 

        % Asignar a la estructura
        struct_lds_sano(index).struct_VSCd_noises(j).name_file_noise = ['Ruido', num2str(j)]; % Guardar el nombre del archivo
        % Lado derecho cerebro
        struct_lds_sano(index).struct_VSCd_noises(j).pam_noise = data_pam_noises_sanos; % Guardar la senal PAM con ruido
        struct_lds_sano(index).struct_VSCd_noises(j).vscd_noise = data_vscd_noises_sanos; % Guardar la senal VSCd con ruido
        % Lado izquierdo cerebro
        struct_lds_sano(index).struct_VSCi_noises(j).pam_noise = data_pam_noises_sanos; % Guardar la senal PAM con ruido
        struct_lds_sano(index).struct_VSCi_noises(j).vsci_noise = data_vsci_noises_sanos; % Guardar la senal VSCi con ruido


        %########################################
        %######## Aplicacion de CWT #############
        %########################################
    
        % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor)
        % Wavelet:amor
        
        % [PAM NOISE - CWT]
        [coefs_pam_noise, freqs_pam_noise, scalcfs_pam_noise, psif_pam_noise] = cwt(struct_lds_sano(index).struct_VSCd_noises(j).pam_noise);
        % [VSC NOISE - CWT]
        [coefs_vscd_noise, freqs_vscd_noise, scalcfs_vscd_noise, psif_vscd_noise] = cwt(struct_lds_sano(index).struct_VSCd_noises(j).vscd_noise);
        % [VSC NOISE - CWT]
        [coefs_vsci_noise, freqs_vsci_noise, scalcfs_vsci_noise, psif_vsci_noise] = cwt(struct_lds_sano(index).struct_VSCi_noises(j).vsci_noise);

        % Almacenando nueva informacion en la respectiva estructura de senales
        % con ruido:
        
        % Almacenando coeficientes (matriz compleja)
        struct_lds_sano(index).struct_VSCd_noises(j).matrix_complex_pam = coefs_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCi_noises(j).matrix_complex_pam = coefs_pam_noise; % PAM lado izquierdo
        struct_lds_sano(index).struct_VSCd_noises(j).matrix_complex_vscd = coefs_vscd_noise; % VSCd
        struct_lds_sano(index).struct_VSCi_noises(j).matrix_complex_vsci = coefs_vsci_noise; % VSCi
    
        % Almacenando escalas de coeficientes (vector fila 1D real , largo 1024)
        struct_lds_sano(index).struct_VSCd_noises(j).scalscfs_pam_noise = scalcfs_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCi_noises(j).scalscfs_pam_noise = scalcfs_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCd_noises(j).scalscfs_vscd_noise = scalcfs_vscd_noise; % VSCd
        struct_lds_sano(index).struct_VSCi_noises(j).scalscfs_vscd_noise = scalcfs_vsci_noise; % VSCd
    
        % Almacenando respuestas de filtros (matriz real 30x1024)
        struct_lds_sano(index).struct_VSCd_noises(j).psif_pam_noise = psif_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCi_noises(j).psif_pam_noise = psif_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCd_noises(j).psif_vscd_noise = psif_vscd_noise; % VSCd
        struct_lds_sano(index).struct_VSCi_noises(j).psif_vsci_noise = psif_vsci_noise; % VSCi

        % Almacenando frecuencias (para mostrar escalograma)
        struct_lds_sano(index).struct_VSCd_noises(j).freqs_pam_noise = freqs_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCi_noises(j).freqs_pam_noise = freqs_pam_noise; % PAM lado derecho
        struct_lds_sano(index).struct_VSCd_noises(j).freqs_vscd_noise = freqs_vscd_noise; % VSCd
        struct_lds_sano(index).struct_VSCi_noises(j).freqs_vsci_noise = freqs_vsci_noise; % VSCi 
        %==========================================================================================================================================

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%% PACIENTE TEC %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        name_csv_pam_tec = pam_noises_csv_tec(j).name; % Nombre del archivo.csv PAMnoises
        name_csv_vscd_tec = vscd_noises_csvd_tec(j).name; % Nombre del archivo.csv VSCdnoises
        name_csv_vsci_tec = vsci_noises_csvi_tec(j).name; % Nombre del archivo.csv VSCinoises
        % Verificar tripleta de datos vinculados (ej: 1_DENI1005_ruidoPAM1.csv | 1_DENI1005_ruidoVSCd1.csv | 1_DENI1005_ruidoVSCi1.csv)
        %fprintf("%s | %s | %s", name_csv_pam_tec, name_csv_vscd_tec, name_csv_vsci_tec); 
    
        path_pam_noises_tecs = fullfile(path_pam_noises_tec, name_csv_pam_tec); % Ruta completa de la carpeta PAMnoises
        path_vscd_noises_tecs = fullfile(path_vscd_noises_tec, name_csv_vscd_tec); % Ruta completa de la carpeta VSCdnoises
        path_vsci_noises_tecs = fullfile(path_vsci_noises_tec, name_csv_vsci_tec); % Ruta completa de la carpeta VSCinoises


        % Leer el contenido del archivo CSV en carpeta PAMnoises
        data_pam_noises_tec = readmatrix(path_pam_noises_tecs);
        % Leer el contenido del archivo CSV en la carpeta VSCdnoises
        data_vscd_noises_tec = readmatrix(path_vscd_noises_tecs);
        % Leer el contenido del archivo CSV en la carpeta VSCinoises
        data_vsci_noises_tec = readmatrix(path_vsci_noises_tecs); 

        % Asignar a la estructura
        struct_lds_tec(index).struct_VSCd_noises(j).name_file_noise = ['Ruido', num2str(j)]; % Guardar el nombre del archivo
        % Lado derecho cerebro
        struct_lds_tec(index).struct_VSCd_noises(j).pam_noise = data_pam_noises_tec; % Guardar la senal PAM con ruido
        struct_lds_tec(index).struct_VSCd_noises(j).vscd_noise = data_vscd_noises_tec; % Guardar la senal VSCd con ruido
        % Lado izquierdo cerebro
        struct_lds_tec(index).struct_VSCi_noises(j).pam_noise = data_pam_noises_tec; % Guardar la senal PAM con ruido
        struct_lds_tec(index).struct_VSCi_noises(j).vsci_noise = data_vsci_noises_tec; % Guardar la senal VSCi con ruido


        %########################################
        %######## Aplicacion de CWT #############
        %########################################
    
        % WAVELET MADRE CONTINUA A UTILIZAR: Analytic Morlet (Gabor)
        % Wavelet:amor
        
        % [PAM NOISE - CWT]
        [coefs_pam_noise, freqs_pam_noise, scalcfs_pam_noise, psif_pam_noise] = cwt(struct_lds_tec(index).struct_VSCd_noises(j).pam_noise);
        % [VSC NOISE - CWT]
        [coefs_vscd_noise, freqs_vscd_noise, scalcfs_vscd_noise, psif_vscd_noise] = cwt(struct_lds_tec(index).struct_VSCd_noises(j).vscd_noise);
        % [VSC NOISE - CWT]
        [coefs_vsci_noise, freqs_vsci_noise, scalcfs_vsci_noise, psif_vsci_noise] = cwt(struct_lds_tec(index).struct_VSCi_noises(j).vsci_noise);

        
        % Almacenando nueva informacion en la respectiva estructura de senales
        % con ruido:
        
        % Almacenando coeficientes (matriz compleja)
        struct_lds_tec(index).struct_VSCd_noises(j).matrix_complex_pam = coefs_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCi_noises(j).matrix_complex_pam = coefs_pam_noise; % PAM lado izquierdo
        struct_lds_tec(index).struct_VSCd_noises(j).matrix_complex_vscd = coefs_vscd_noise; % VSCd
        struct_lds_tec(index).struct_VSCi_noises(j).matrix_complex_vsci = coefs_vsci_noise; % VSCi
    
        % Almacenando escalas de coeficientes (vector fila 1D real , largo 1024)
        struct_lds_tec(index).struct_VSCd_noises(j).scalscfs_pam_noise = scalcfs_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCi_noises(j).scalscfs_pam_noise = scalcfs_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCd_noises(j).scalscfs_vscd_noise = scalcfs_vscd_noise; % VSCd
        struct_lds_tec(index).struct_VSCi_noises(j).scalscfs_vscd_noise = scalcfs_vsci_noise; % VSCd
    
        % Almacenando respuestas de filtros (matriz real 30x1024)
        struct_lds_tec(index).struct_VSCd_noises(j).psif_pam_noise = psif_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCi_noises(j).psif_pam_noise = psif_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCd_noises(j).psif_vscd_noise = psif_vscd_noise; % VSCd
        struct_lds_tec(index).struct_VSCi_noises(j).psif_vsci_noise = psif_vsci_noise; % VSCi

        % Almacenando frecuencias (para mostrar escalograma)
        struct_lds_tec(index).struct_VSCd_noises(j).freqs_pam_noise = freqs_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCi_noises(j).freqs_pam_noise = freqs_pam_noise; % PAM lado derecho
        struct_lds_tec(index).struct_VSCd_noises(j).freqs_vscd_noise = freqs_vscd_noise; % VSCd
        struct_lds_tec(index).struct_VSCi_noises(j).freqs_vsci_noise = freqs_vsci_noise; % VSCi 
        %==========================================================================================================================================
    end
    
    % Aqui ya se han calculados todas las cwt de cada senal con ruido de
    % PAM, VSCd y VSCd. Ahora se procede a calcular los coeficientes(matrices complejas)
    % de las senales originales sin ruido.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PARA SENALES ORIGINALES DE PAM, VSCd Y VSCi del sujeto index:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % SUJETO SANO
    % [ORIGINAL PAM - CWT]
    [coefs_original_pam, freqs_original_pam, scalcfs_original_pam, psif_original_pam] = cwt(struct_lds_sano(index).signal_pam);
    % [ORIGINAL VSCd - CWT]
    [coefs_original_vscd, freqs_original_vscd, scalcfs_original_vscd, psif_original_vscd] = cwt(struct_lds_sano(index).signal_vscd);
    % [ORIGINAL VSCi- CWT]
    [coefs_original_vsci, freqs_original_vsci, scalcfs_original_vsci, psif_original_vsci] = cwt(struct_lds_sano(index).signal_vsci);
    
    % SUJETO SANO
    % Almacenando datos del sujeto index en la respectiva estructura
    % SUJETO SANO - ORIGINAL PAM
    struct_lds_sano(index).struct_original_PAM(1).matrix_complex_original_pam = coefs_original_pam;
    struct_lds_sano(index).struct_original_PAM(1).scalscfs_original_pam = scalcfs_original_pam;
    struct_lds_sano(index).struct_original_PAM(1).psif_original_pam = psif_original_pam;
    struct_lds_sano(index).struct_original_PAM(1).freqs_original_pam = freqs_original_pam;
    % SUJETO SANO - ORIGINAL VSCd
    struct_lds_sano(index).struct_original_VSCd(1).matrix_complex_original_vscd = coefs_original_vscd;
    struct_lds_sano(index).struct_original_VSCd(1).scalscfs_original_vscd = scalcfs_original_vscd;
    struct_lds_sano(index).struct_original_VSCd(1).psif_original_vscd = psif_original_vscd;
    struct_lds_sano(index).struct_original_VSCd(1).freqs_original_vscd = freqs_original_vscd;
    % SUJETO SANO - ORIGINAL VSCi
    struct_lds_sano(index).struct_original_VSCi(1).matrix_complex_original_vsci = coefs_original_vsci;
    struct_lds_sano(index).struct_original_VSCi(1).scalscfs_original_vsci = scalcfs_original_vsci;
    struct_lds_sano(index).struct_original_VSCi(1).psif_original_vsci = psif_original_vsci;
    struct_lds_sano(index).struct_original_VSCi(1).freqs_original_vsci = freqs_original_vsci;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % PACIENTE TEC
    % [ORIGINAL PAM - CWT]
    [coefs_original_pam, freqs_original_pam, scalcfs_original_pam, psif_original_pam] = cwt(struct_lds_tec(index).signal_pam);
    % [ORIGINAL VSCd - CWT]
    [coefs_original_vscd, freqs_original_vscd, scalcfs_original_vscd, psif_original_vscd] = cwt(struct_lds_tec(index).signal_vscd);
    % [ORIGINAL VSCi- CWT]
    [coefs_original_vsci, freqs_original_vsci, scalcfs_original_vsci, psif_original_vsci] = cwt(struct_lds_tec(index).signal_vsci);
    
    % PACIENTE TEC
    % Almacenando datos del sujeto index en la respectiva estructura
    % PACIENTE TEC - ORIGINAL PAM
    struct_lds_tec(index).struct_original_PAM(1).matrix_complex_original_pam = coefs_original_pam;
    struct_lds_tec(index).struct_original_PAM(1).scalscfs_original_pam = scalcfs_original_pam;
    struct_lds_tec(index).struct_original_PAM(1).psif_original_pam = psif_original_pam;
    struct_lds_tec(index).struct_original_PAM(1).freqs_original_pam = freqs_original_pam;
    % PACIENTE TEC - ORIGINAL VSCd
    struct_lds_tec(index).struct_original_VSCd(1).matrix_complex_original_vscd = coefs_original_vscd;
    struct_lds_tec(index).struct_original_VSCd(1).scalscfs_original_vscd = scalcfs_original_vscd;
    struct_lds_tec(index).struct_original_VSCd(1).psif_original_vscd = psif_original_vscd;
    struct_lds_tec(index).struct_original_VSCd(1).freqs_original_vscd = freqs_original_vscd;
    % PACIENTE TEC - ORIGINAL VSCi
    struct_lds_tec(index).struct_original_VSCi(1).matrix_complex_original_vsci = coefs_original_vsci;
    struct_lds_tec(index).struct_original_VSCi(1).scalscfs_original_vsci = scalcfs_original_vsci;
    struct_lds_tec(index).struct_original_VSCi(1).psif_original_vsci = psif_original_vsci;
    struct_lds_tec(index).struct_original_VSCi(1).freqs_original_vsci = freqs_original_vsci;
end


% Especifica el directorio donde deseas guardar la estructura
directory_structs = 'D:\TT\Memoria\MemoriaCodigoFuentev3\codigo_matlab\codigo_fuente\Estructuras_SANOS_TEC';

%============== Sanos ==================================================
% Especifica el nombre del archivo
filename_struct_sano = 'struct_lds_sano.mat';
% Crea la ruta completa del archivo
filepath_struct_sano = fullfile(directory_structs, filename_struct_sano);
% Guarda la estructura en el archivo .mat
save(filepath_struct_sano, 'struct_lds_sano', '-v7.3');
% Mensaje de confirmación
fprintf("(*) La estructura para sujetos sanos se ha guardado correctamente\n");

%============== TEC ==================================================
% Especifica el nombre del archivo
filename_struct_tec = 'struct_lds_tec.mat';
% Crea la ruta completa del archivo
filepath_struct_tec = fullfile(directory_structs, filename_struct_tec);
% Guarda la estructura en el archivo .mat
save(filepath_struct_tec, 'struct_lds_tec', '-v7.3');
% Mensaje de confirmación
fprintf("(*) La estructura para pacientes TEC se ha guardado correctamente\n");


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% GUARDAR MATRICES COMPLEJAS (COEFICIENTES) EN FORMATO ".mat" -> ENTRENAR RED U-NET %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Almacenar matrices complejas pam y vsc en carpetas especificas para 
% luego trabajar con la red profunda en python. Para ello se importan 
% las matrices en formato.mat y luego en python se utiliza un script
% para transformar dicho formato a npy.
direct_sanos = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/Signals_LDS/SANOS';
direct_tecs = 'D:/TT/Memoria/MemoriaCodigoFuentev3/codigo_matlab/codigo_fuente/Signals_LDS/TEC';

% Obtener los nombres de las carpetas dentro del directorio de SANOS
folders_sanos = dir(direct_sanos);
foldernames_sanos = {folders_sanos([folders_sanos.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
foldernames_sanos = setdiff(foldernames_sanos, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de sujetos sanos


% Obtener los nombres de las carpetas dentro del directorio de TEC
folders_tecs = dir(direct_tecs);
foldernames_tecs = {folders_tecs([folders_tecs.isdir]).name}; % se obtiene los nombres de las carpetas
% Eliminar los nombres '.' y '..' que representan el directorio actual y el directorio padre
foldernames_tecs = setdiff(foldernames_tecs, {'.', '..'}); % vector fila que almcena los nombres de todas las carpetas de pacientes tec


%################################################
% Guardar las matrices complejas en archivos .mat
%################################################
num_people = 27;
num_matrix_complexs = 50;
for i = 1:num_people
    % Se deben crear 3 carpetas para cada sujeto sano y paciente TEC. una
    % carpeta de matriz_compleja_pam, matriz_compleja_vscd y
    % matriz_compleja_vsci
    direct_folder_sano_i = fullfile(direct_sanos, foldernames_sanos{i}); % carpeta del sujeto sano posicion i
    direct_folder_tec_i = fullfile(direct_tecs, foldernames_tecs{i}); % carpeta del paciente tec posicion i

    % Creando las respectivas 3 carpetas, sino existen se crean.
    % Directorios para guardar los archivos.mat asociados a los inputs de coeficientes

    % Nombres d que tendran las respectivas carpetas:
    name_folder_pam = '/PAMnoises_matrixcomplex_mat';
    name_folder_vscd = '/VSCdnoises_matrixcomplex_mat';
    name_folder_vsci = '/VSCinoises_matrixcomplex_mat';
    % Para las matrices complejas de senales originales:
    name_folder_original_pam = '/PAMoriginal_matrixcomplex';
    name_folder_original_vscd = '/VSCdoriginal_matrixcomplex';
    name_folder_original_vsci = '/VSCioriginal_matrixcomplex';
    
    % Directorio donde se crearan dichas carpetas (3 total:
    % name_folder_pam, name_folder_vscd y name_folder_vsci) - SANO ***
    direct_final_cfs_pam_sano = fullfile(direct_folder_sano_i, name_folder_pam);
    direct_final_cfs_vscd_sano = fullfile(direct_folder_sano_i, name_folder_vscd);
    direct_final_cfs_vsci_sano = fullfile(direct_folder_sano_i, name_folder_vsci);
    % Directorios para senales originales
    direct_final_cfs_original_pam_sano = fullfile(direct_folder_sano_i, name_folder_original_pam);
    direct_final_cfs_original_vscd_sano = fullfile(direct_folder_sano_i, name_folder_original_vscd);
    direct_final_cfs_original_vsci_sano = fullfile(direct_folder_sano_i, name_folder_original_vsci);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Directorio donde se crearan dichas carpetas (3 total:
    % name_folder_pam, name_folder_vscd y name_folder_vsci) - TEC ***
    direct_final_cfs_pam_tec = fullfile(direct_folder_tec_i, name_folder_pam);
    direct_final_cfs_vscd_tec = fullfile(direct_folder_tec_i, name_folder_vscd);
    direct_final_cfs_vsci_tec = fullfile(direct_folder_tec_i, name_folder_vsci);
    % Directorios para senales originales
    direct_final_cfs_original_pam_tec = fullfile(direct_folder_tec_i, name_folder_original_pam);
    direct_final_cfs_original_vscd_tec = fullfile(direct_folder_tec_i, name_folder_original_vscd);
    direct_final_cfs_original_vsci_tec = fullfile(direct_folder_tec_i, name_folder_original_vsci);

    % Creando carpetas si no existen...
    %%%% SANO %%%%
    if ~exist(direct_final_cfs_pam_sano, 'dir')
        mkdir(direct_final_cfs_pam_sano);
    end
    if ~exist(direct_final_cfs_vscd_sano, 'dir')
        mkdir(direct_final_cfs_vscd_sano);
    end
    if ~exist(direct_final_cfs_vsci_sano, 'dir')
        mkdir(direct_final_cfs_vsci_sano);
    end
    % senales originales
    if ~exist(direct_final_cfs_original_pam_sano, 'dir')
        mkdir(direct_final_cfs_original_pam_sano);
    end
    if ~exist(direct_final_cfs_original_vscd_sano, 'dir')
        mkdir(direct_final_cfs_original_vscd_sano);
    end
    if ~exist(direct_final_cfs_original_vsci_sano, 'dir')
        mkdir(direct_final_cfs_original_vsci_sano);
    end



    %%%% TEC %%%%
    if ~exist(direct_final_cfs_pam_tec, 'dir')
        mkdir(direct_final_cfs_pam_tec);
    end
    if ~exist(direct_final_cfs_vscd_tec, 'dir')
        mkdir(direct_final_cfs_vscd_tec);
    end
    if ~exist(direct_final_cfs_vsci_tec, 'dir')
        mkdir(direct_final_cfs_vsci_tec);
    end
    % senales originales
    if ~exist(direct_final_cfs_original_pam_tec, 'dir')
        mkdir(direct_final_cfs_original_pam_tec);
    end
    if ~exist(direct_final_cfs_original_vscd_tec, 'dir')
        mkdir(direct_final_cfs_original_vscd_tec);
    end
    if ~exist(direct_final_cfs_original_vsci_tec, 'dir')
        mkdir(direct_final_cfs_original_vsci_tec);
    end

    % Se obtiene el sujeto sano i y el paciente tec i de sus respectivas
    % estructuras.
    person_sano = struct_lds_sano(i);
    person_tec = struct_lds_tec(i);
    fprintf("[%i] Rellenando carpetas: SANO: %s || TEC: %s ...\n", i, foldernames_sanos{i}, foldernames_tecs{i});
    fprintf("[%i] Guardando matrices complejas (PAM, VSCd y VSCi) de: SANO %s || TEC: %s\n",i, person_sano.name_file, person_tec.name_file);
    for j = 1:num_matrix_complexs
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SANO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % Guardar matriz_complex_pam
        matrix_complex_pam_sano = person_sano.struct_VSCd_noises(j).matrix_complex_pam; % la PAM es la misma ya sea en VSCd o VSCi
        save(fullfile(direct_final_cfs_pam_sano, sprintf('matrix_complex_pam_noise_%d.mat', j)), 'matrix_complex_pam_sano') 
        % Guardar matriz_complex_vscd
        matrix_complex_vscd_sano = person_sano.struct_VSCd_noises(j).matrix_complex_vscd;
        save(fullfile(direct_final_cfs_vscd_sano, sprintf('matrix_complex_vscd_noise_%d.mat', j)), 'matrix_complex_vscd_sano');
        % Guardar matriz_complex_vsci
        matrix_complex_vsci_sano = person_sano.struct_VSCi_noises(j).matrix_complex_vsci;
        save(fullfile(direct_final_cfs_vsci_sano, sprintf('matrix_complex_vsci_noise_%d.mat', j)), 'matrix_complex_vsci_sano');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % Guardar matriz_complex_pam
        matrix_complex_pam_tec = person_tec.struct_VSCd_noises(j).matrix_complex_pam;
        save(fullfile(direct_final_cfs_pam_tec, sprintf('matrix_complex_pam_noise_%d.mat', j)), 'matrix_complex_pam_tec');
        % Guardar matriz_complex_vscd
        matrix_complex_vscd_tec = person_tec.struct_VSCd_noises(j).matrix_complex_vscd;
        save(fullfile(direct_final_cfs_vscd_tec, sprintf('matrix_complex_vscd_noise_%d.mat', j)), 'matrix_complex_vscd_tec');
        % Guardar matriz_complex_vsci
        matrix_complex_vsci_tec = person_tec.struct_VSCi_noises(j).matrix_complex_vsci;
        save(fullfile(direct_final_cfs_vsci_tec, sprintf('matrix_complex_vsc_noise_%d.mat', j)), 'matrix_complex_vsci_tec');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    % PARA SUJETO SANO
    % Guardar matriz_complex_original_pam
    matrix_complex_original_pam_sano = person_sano.struct_original_PAM(1).matrix_complex_original_pam; % la PAM es la misma ya sea en VSCd o VSCi
    save(fullfile(direct_final_cfs_original_pam_sano, 'matrix_complex_original_pam.mat'), 'matrix_complex_original_pam_sano');
    % Guardar matriz_complex_original_vscd
    matrix_complex_original_vscd_sano = person_sano.struct_original_VSCd(1).matrix_complex_original_vscd;
    save(fullfile(direct_final_cfs_original_vscd_sano, 'matrix_complex_original_vscd.mat'), 'matrix_complex_original_vscd_sano');
    % Guardar matriz_complex_original_vsci
    matrix_complex_original_vsci_sano = person_sano.struct_original_VSCi(1).matrix_complex_original_vsci;
    save(fullfile(direct_final_cfs_original_vsci_sano, 'matrix_complex_original_vsci.mat'), 'matrix_complex_original_vsci_sano');

    % PARA PACIENTE TEC
    % Guardar matriz_complex_original_pam
    matrix_complex_original_pam_tec = person_tec.struct_original_PAM(1).matrix_complex_original_pam; % la PAM es la misma ya sea en VSCd o VSCi
    save(fullfile(direct_final_cfs_original_pam_tec, 'matrix_complex_original_pam.mat'), 'matrix_complex_original_pam_tec');
     % Guardar matriz_complex_original_vscd
    matrix_complex_original_vscd_tec = person_tec.struct_original_VSCd(1).matrix_complex_original_vscd;
    save(fullfile(direct_final_cfs_original_vscd_tec, 'matrix_complex_original_vscd.mat'), 'matrix_complex_original_vscd_tec');
    % Guardar matriz_complex_original_vsci
    matrix_complex_original_vsci_tec = person_tec.struct_original_VSCi(1).matrix_complex_original_vsci;
    save(fullfile(direct_final_cfs_original_vsci_tec, 'matrix_complex_original_vsci.mat'), 'matrix_complex_original_vsci_tec');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% GENERACION DE COEFICIENTES DE SENALES ORIGINALES PARA PREDECIR UNA SALIDA EN LA RED %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
