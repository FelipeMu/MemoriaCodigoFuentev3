function [coefs, freqs, scalcfs, psif] = cwt(signal)
    % [SIGNAL:pam OR vscd OR vsci - CWT]
    filters_bank = cwtfilterbank(SignalLength=length(signal), Boundary="periodic", Wavelet="amor", SamplingFrequency=5, VoicesPerOctave=5); % se obtiene una estructura (banco de filtros)
    psif = freqz(filters_bank, FrequencyRange="twosided",IncludeLowpass=true); % psif_amor: Como cada filtro responde a diferentes frecuencias. ayuda a comprender como se distribuyen las frecuencias a lo largo de mi señal
    [coefs , freqs, ~, scalcfs] = wt(filters_bank, signal); % se aplica la transformada continua a la senal
end