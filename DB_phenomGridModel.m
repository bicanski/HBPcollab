%% Directly compute rate and phase functions of grid cell firing patterns
%  Then decode location, heading direction, running speed, and more!
%  Daniel Bush, UCL (2017), following Chadwick et al. (2016) eLife


%% Provide some parameters
gridScales      = 30*1.4.^(0:4);                                % Grid scales (cm)
nGCs            = 40;                                           % Number of phase offsets per module
sampleRate      = 500;                                          % Simulation time step (Hz)
sigmas          = gridScales./10;                               % Firing field width parameter (cm)
meanRate        = 5;                                            % Mean firing rate (Hz)
phaseMod        = 1.5;                                         	% Extent of phase coding (au)
speedSlope      = 5/30;                                         % Slope of running speed v firing rate (Hz/cm/s)


%% Sort out the tracking data
v_range         = [2 30];                                       % Velocity range (cm/s)
t_log           = 1/sampleRate : 1/sampleRate : 30;             % Create a time base (s)
v_log           = cumsum(randn(length(t_log),1));               % Assign time varying velocity (cm/s)
v_log           = ((v_log-min(v_log))./range(v_log).*diff(v_range)+v_range(1)); clear v_range
x_log           = cumsum(v_log ./ sampleRate);                  % Compute x co-ordinates
y_log           = zeros(size(x_log));                           % Assign y co-ordinates
dt              = 1/sampleRate; clear sampleRate              	% Compute time step


%% Get some human LFP
LFP_range       = [2 20];                                       % LFP frequency range (Hz)
a_range         = [0 1];                                        % Abstract 'anxiety' range
lfpWin          = [318 348];                                    % Time window of human iEEG data
folder          = 'C:\Users\dbush\Documents\iEEG Project 2\Patient 4\';
edffile         = 'DM.edf';                                     % Human iEEG data file
channel         = 7;                                            % Human iEEG data channel
header          = read_edf_file('header',[folder edffile]);     % Load header info
header          = rmfield(header,'patient_id');                 % Anonymise
eeg             = read_edf_file('data',  [folder edffile], header, channel, 1, header.n_data_records); clear folder edffile
eeg(:,2)        = ((1/header.Fs) : (1/header.Fs) : (header.n_data_records.*header.nr_of_samples(channel) * (1/header.Fs)))'; clear channel
[b,a]           = butter(2,LFP_range/(header.Fs/2));            % Generate second order Butterworth filter
LFPFreq         = filtfilt(b,a,eeg(:,1)); clear b a             % Filter EEG data in frequency range of interest
LFPFreq         = LFPFreq(eeg(:,2)>=lfpWin(1)&eeg(:,2)<=lfpWin(2));
eeg             = eeg(eeg(:,2)>=lfpWin(1)&eeg(:,2)<=lfpWin(2),:);
LFPFreq         = angle(hilbert(LFPFreq));                      % Get the phase at each time point
LFPFreq         = angle(exp(1i.*diff(LFPFreq)));                % Get the phase difference and wrap around 
LFPFreq(LFPFreq<0)  = (LFP_range(1)*2*pi)/header.Fs;            % Ignore negative frequencies
LFPFreq         = fastsmooth(LFPFreq,header.Fs/20,3,1);         % Smooth the data
LFPFreq         = interp1(linspace(lfpWin(1),lfpWin(2),length(LFPFreq)),LFPFreq,lfpWin(1):dt:(lfpWin(2)-dt));
eeg             = interp1(linspace(lfpWin(1),lfpWin(2),length(eeg)),eeg(:,1),lfpWin(1):dt:(lfpWin(2)-dt));
LFPFreq         = (LFPFreq./(2*pi)).*header.Fs; clear header    % Convert to dynamic frequency
anxiety        	= ((LFPFreq-min(LFPFreq))./range(LFPFreq).*diff(a_range)+a_range(1)); clear a_range
LFP_Phase     	= cumsum([0 2.*pi.*dt.*LFPFreq(2:end)]);        % Compute LFP phase at each timepoint

% anxiety         = ones(1,15000);
% LFPFreq         = 8 .* anxiety;
% LFP_Phase     	= cumsum([0 2.*pi.*dt.*LFPFreq(2:end)]);        % Compute LFP phase at each timepoint
% eeg             = cos(LFP_Phase);


%% Compute firing rate and phase functions, generate output spiking
r_rate          = nan(length(gridScales)*nGCs,length(x_log));   % Assign memory for the rate code of each grid cell
r_phase         = nan(length(gridScales)*nGCs,length(x_log));   % Assign memory for the phase code of each grid cell
r_cell          = nan(length(gridScales)*nGCs,length(x_log));   % Assign memory for the combined firing output of each cell
for module      = 1 : length(gridScales)
    for cell    = 1 : nGCs
        ind             = (module-1)*nGCs + cell;
        x_centres       = ((cell-1)/nGCs)*gridScales(module)-gridScales(module) : gridScales(module) : max(x_log) + gridScales(module);
        offset          = min(abs(repmat(x_log,1,length(x_centres)) - repmat(x_centres,length(x_log),1)),[],2);
        r_rate(ind,:)   = exp(-(offset.^2) ./ (2.*sigmas(module)).^2); clear offset
        offset          = x_log - x_centres(1) - gridScales(module)/2; clear x_centres
        offset          = mod((-1/gridScales(module)).*offset,1)*2*pi;
        r_phase(ind,:) 	= exp(phaseMod*cos(offset-LFP_Phase')); clear offset
        r_phase(ind,:)  = r_phase(ind,:) ./ max(r_phase(ind,:));
        r_cell(ind,:)   = r_rate(ind,:) .* r_phase(ind,:); 
        r_cell(ind,:)   = r_cell(ind,:) .* LFPFreq .* speedSlope .* v_log';
        r_cell(ind,:)   = r_cell(ind,:) ./ sum(r_cell(ind,:)) .* meanRate .* range(t_log); clear ind
    end
end
firingRates     = poissrnd(r_cell);
clear module cell r_cell sigmas


%% Split the data into expected and actual firing rate per oscillatory 
%  cycle, and record mean position in each cycle
[~,peaks]       = findpeaks(cos(LFP_Phase));
expFire         = nan(length(gridScales),nGCs,length(peaks)-1);
rateMaps        = nan(length(gridScales),nGCs,length(peaks)-1);
actLoc          = nan(length(peaks)-1,1);
meanSpeed       = nan(length(peaks)-1,1);
meanRate        = nan(length(peaks)-1,1);
meanAnxiety     = nan(length(peaks)-1,1);
cycleLength    	= nan(length(peaks)-1,1);
for cycle       = 1 : length(peaks)-1
    spikes      = sum(firingRates(:,peaks(cycle):peaks(cycle+1)),2);
    rateMaps(:,:,cycle)     = reshape(spikes,[nGCs length(gridScales)])'; 
    meanRate(cycle,1)       = mean(spikes); clear spikes
    rate        = mean(r_rate(:,peaks(cycle):peaks(cycle+1)),2);
    expFire(:,:,cycle)      = reshape(rate,[nGCs length(gridScales)])'; clear rate
    actLoc(cycle,1)         = mean(x_log(peaks(cycle):peaks(cycle+1)));
    meanSpeed(cycle,1)      = mean(v_log(peaks(cycle):peaks(cycle+1)));       
    meanAnxiety(cycle,1)    = mean(anxiety(peaks(cycle):peaks(cycle+1)));       
    cycleLength(cycle,1)    = (peaks(cycle+1)-peaks(cycle)).*dt;
end
clear cycle


%% Decode location in each cycle using Mathis et al. (2012) MLE method
expExpFire      = exp(-expFire);
factRateMaps    = factorial(rateMaps);
decodedLoc  	= nan(size(actLoc));
for bin         = 1 : length(actLoc)
    currK           = repmat(rateMaps(:,:,bin), [1,1, length(actLoc)]);
    fact_currK      = repmat(factRateMaps(:,:,bin), [1,1, length(actLoc)]);
    pval_contrib    = ((expFire.^currK)./fact_currK) .* expExpFire;
    pval_contrib    = reshape(pval_contrib, [nGCs*length(gridScales), length(actLoc)]);    
    pval            = prod(pval_contrib);
    decode          = find(pval==max(pval));
    decodedLoc(bin,1)   = decode(randi(length(decode),1)); clear currK fact_currK pval_contrib pval decode
end
decodedLoc              = actLoc(decodedLoc);
clear bin expFire expExpFire rateMaps factRateMaps


%% Split the data again, by portions of a cycle (inefficient, integrate 
%  into the loops above)
phaseBins       = 5;
if phaseBins    == 7
    prcntiles 	= [0 0.32 0.4 0.47 0.53 0.6 0.68 1];
elseif phaseBins== 5
    prcntiles 	= [0 0.35 0.47 0.53 0.7 1];
end
expFireDyn      = nan(length(gridScales),nGCs,(length(peaks)-1)*phaseBins);
rateMapsDyn     = nan(length(gridScales),nGCs,(length(peaks)-1)*phaseBins);
actLocDyn       = nan((length(peaks)-1)*phaseBins,1);
spikesPerPBin   = zeros(1,7);
for cycle       = 1 : length(peaks)-1
    inds        = (cycle-1)*phaseBins+(1:phaseBins);
    bins        = peaks(cycle) + ceil(prcntiles.*(peaks(cycle+1)-peaks(cycle)));
    for bin     = 1 : length(bins)-1
        spikes      = sum(firingRates(:,bins(bin):bins(bin+1)-1),2);
        spikesPerPBin(bin)          = spikesPerPBin(bin) + sum(spikes);
        rateMapsDyn(:,:,inds(bin))	= reshape(spikes,[nGCs length(gridScales)])'; clear spikes
        rate        = mean(r_rate(:,bins(bin):bins(bin+1)-1),2);
        expFireDyn(:,:,inds(bin))   = reshape(rate,[nGCs length(gridScales)])'; clear rate
        actLocDyn(inds(bin),1)    	= mean(x_log(bins(bin):bins(bin+1)-1));
    end
    clear bin bins inds
end
clear cycle phaseBind


%% Decode location in each bin, again using Mathis et al. (2012) MLE method
expExpFireDyn 	= exp(-expFireDyn);
factRateMapsDyn = factorial(rateMapsDyn);
decodedLocDyn 	= nan(length(peaks)-1,phaseBins);
decodedB        = nan(length(peaks)-1,1);
for cycle       = 1 : length(peaks)-1
    for bin             = 1 : phaseBins
        ind             = (cycle-1)*phaseBins+bin;
        currK           = repmat(rateMapsDyn(:,:,ind), [1,1, size(rateMapsDyn,3)]);
        fact_currK      = repmat(factRateMapsDyn(:,:,ind), [1,1, size(rateMapsDyn,3)]);
        pval_contrib    = ((expFireDyn.^currK)./fact_currK) .* expExpFireDyn;
        pval_contrib    = reshape(pval_contrib, [nGCs*length(gridScales), size(rateMapsDyn,3)]);
        pval            = prod(pval_contrib);
        decode          = find(pval==max(pval));
        decodedLocDyn(cycle,bin)    = decode(randi(length(decode),1)); clear currK fact_currK pval_contrib pval decode ind                
    end
    b                   = regress(decodedLocDyn(cycle,:)',[1:phaseBins ; ones(1,phaseBins)]');
    decodedB(cycle,1)   = b(1); clear b
end
decodedLocDyn           = actLocDyn(decodedLocDyn);
clear cycle bin expFireDyn expExpFireDyn rateMapsDyn factRateMapsDyn peaks


%% Convert to a heat map
decodedLocDyn 	= decodedLocDyn - repmat(decodedLocDyn(:,round(phaseBins/2)),1,phaseBins);
distAx          = -20 : 0.5 : 20;
phaseAx         = (diff(prcntiles)/2+prcntiles(1:phaseBins))*2*pi; clear prcntiles
sweepMap        = nan(length(distAx),phaseBins);
for bin         = 1 : phaseBins
    sweepMap(:,bin)	= histc(decodedLocDyn(:,bin),distAx);
end
clear bin
sweepMap        = sweepMap./repmat(max(sweepMap),size(sweepMap,1),1);


%% Process the spiking data for a single grid cell
chosenCell      = 41;
[rmLimits,smoothRateMap,lags,rawAC,spkFreq,spkPower,lfpFreq,lfpPower,spikePos,spikeTime,spikePhase,allSpikePhase] = analyseFiring(firingRates,gridScales,nGCs,chosenCell,x_log,t_log,LFP_Phase,dt);


%% Plot some figures
axSizeS         = 14;
axSizeL         = 24;
xRange          = [0 500];
tRange          = t_log([find(x_log>=xRange(1),1,'first') find(x_log<=xRange(2),1,'last')]);

%  Grid cell and LFP dynamics figure
figure
subplot(3,4,1:4) % subplot(2,4,1:4)
plot(rmLimits,smoothRateMap,'k','LineWidth',2)
ylabel('Firing Rate (Hz)','FontSize',axSizeS)
xlim(xRange)
set(gca,'XTickLabel','')

subplot(3,4,5:8) % subplot(2,4,5:8)
plot(x_log,eeg,'k','LineWidth',2)
ylabel('LFP signal (au)','FontSize',axSizeS)
xlim(xRange)
set(gca,'XTickLabel','')

subplot(3,4,9:12)
scatter(spikePos,mod(spikePhase,2*pi),300,'r.')
ylabel('Spike phase (rad)','FontSize',axSizeS)
xlabel('Distance (cm)','FontSize',axSizeS)
xlim(xRange)
ylim([0 2*pi])

%  Phase coding figure
figure
subplot(1,3,1)
bar(lags(2:end),rawAC(2:end),'FaceColor','k')
xlabel('Temporal offset (ms)','FontSize',axSizeS)
ylabel('Relative frequency','FontSize',axSizeS)
xlim([0 1000])
axis square

subplot(1,3,2)
plot(spkFreq,spkPower,'k','LineWidth',2)
xlabel('Spike train frequency (Hz)','FontSize',axSizeS)
ylabel('Spike train power','FontSize',axSizeS)
axis square

subplot(1,3,3)
contourf(linspace(-0.5,0.5,size(allSpikePhase,1)),linspace(0,2*pi,size(allSpikePhase,2)),mean(allSpikePhase,3),50,'linestyle','none')
xlabel('Relative position','FontSize',axSizeS)
ylabel('Firing Phase (rad)','FontSize',axSizeS)
xlim([-0.5 0.5]); ylim([0 2*pi])
axis square
colormap jet

% subplot(2,2,4)
% plot(lfpFreq,lfpPower,'k','LineWidth',2)
% xlabel('LFP frequency (Hz)','FontSize',axSizeS)
% ylabel('Power','FontSize',axSizeS)
% xlim([0 50])
% axis square

%  Decoding figure
figure
subplot(1,3,1)
scatter(actLoc,decodedLoc,500,'k.')
axis square
set(gca,'FontSize',18)
xlabel('Actual location (cm)','FontSize',axSizeL)
ylabel('Decoded location (cm)','FontSize',axSizeL)
xlim([0 600]); ylim([0 600])

subplot(1,3,3)
contourf(phaseAx,distAx,sweepMap,50,'linestyle','none')
axis square
colormap jet
% c = colorbar;
set(gca,'FontSize',18)
xlabel('Firing phase (rad)','FontSize',axSizeL)
% ylabel({'Relative decoded';'location (cm)'},'FontSize',axSizeL)
% ylabel(c,'Normalised likelihood','FontSize',axSizeL); clear c distAx

subplot(1,3,2)
scatter(meanSpeed,meanRate,500,'k.')
axis square
set(gca,'FontSize',18)
xlabel('Running speed','FontSize',axSizeL)
ylabel('Mean spikes per cycle','FontSize',axSizeL)

figure
subplot(1,2,1)
scatter(meanAnxiety,1./cycleLength,500,'k.')
axis square
set(gca,'FontSize',18)
xlabel('Anxiety level (au)','FontSize',axSizeL)
ylabel('LFP frequency (Hz)','FontSize',axSizeL)

subplot(1,2,2)
slopeAx     = floor(min(decodedB)/10)*10 : 10 : ceil(max(decodedB)/10)*10;
bar(slopeAx,histc(decodedB,slopeAx)./length(decodedB),'FaceColor',[0.8 0.8 0.8],'EdgeColor','k','LineWidth',2)
axis square
set(gca,'FontSize',18)
xlabel('Slope of look-ahead activity','FontSize',axSizeL)
ylabel('Relative frequency','FontSize',axSizeL)
clear axSizeS axSizeL xRange slopeAx


%% Function to process the spiking data for a single simulated grid cell
function[rmLimits,smoothRateMap,lags,rawAC,spkFreq,spkPower,lfpFreq,lfpPower,spikePos,spikeTime,spikePhase,allSpikePhase] = analyseFiring(firingRates,gridScales,nGCs,chosenCell,x_log,t_log,LFP_Phase,dt)

% Provide some parameters
rmBinSize       = 2;    	% Rate map bin size (cm)
rmBoxCar        = 3;        % Rate map box car width (bins)
acBinSize       = 0.01;     % Temporal autocorrelogram bin size (s)
acWindow        = 1;        % Temporal autocorrelogram window (s)
fftLength       = 512;     	% Number of frequency points for FFTs
maxFreq         = 50;       % Maximum frequency for FFTs
lfpBoxCar       = 1;        % Power spectra box car width (Hz)
ppBinSize       = 30;       % Number of phase and position bins for precession map

% Get the smoothed rate map
spikePos        = x_log(firingRates(chosenCell,:)>0);
rmLimits        = 0:rmBinSize:ceil(max(x_log)/rmBinSize)*rmBinSize;         % Extent of the firing rate map (cm)
posBinned       = histc(spikePos,rmLimits);
timeBinned      = histc(x_log,rmLimits).*dt;
rateMap         = posBinned ./ timeBinned;                                  % Compute raw firing rate map
smoothRateMap   = fastsmooth(rateMap,rmBoxCar,3);                       	% Smooth firing rate map
clear posBinned rateMap

% Get the temporal auto-correlogram
spikeTime       = t_log(firingRates(chosenCell,:)>0); 
spikePhase      = LFP_Phase(firingRates(chosenCell,:)>0); clear chosenCell
nHistBins       = max(spikeTime) / acBinSize;                               % Compute number of bins for the histogram of spike times
binCentres      = (0:acBinSize:(acBinSize * (nHistBins-1)));                % Assign bin edges
spkTrHist       = histc(spikeTime, binCentres);                             % Calculate the spike time histogram
[rawAC,lags]    = xcorr(spkTrHist,'unbiased');                              % Generate the raw temporal auto-correlation
midPoint        = ceil(length(rawAC)/2);                                    % Find the mid-point (i.e. zero lag) of the auto-correlation
binsToKeep      = acWindow/acBinSize;                                       % Identify the auto-correlation bins to be kept
rawAC           = rawAC(midPoint:midPoint + binsToKeep);                    % Extract that part of the raw auto-correlation
lags            = lags(midPoint:midPoint + binsToKeep);                     % Extract that part of the lags
lags            = lags .* acBinSize .* 1000;                                % Change the lag values to ms
clear midPoint binsToKeep nHistBins binCentres spkTrHist

% Mean normalise, do the Fourier transform and extract the power spectra
meanNormAC      = [0 rawAC(2:end) - mean(rawAC(2:end))];                    % Mean normalise the auto-correlation
fftRes          = fft(meanNormAC, fftLength);                               % Compute the Fourier transform
spkPower     	= (fftRes.*conj(fftRes))/length(fftRes);                    % Convert from amplitude to power
spkFreq         = linspace(0,1,fftLength/2+1)./(2*acBinSize);             	% Compute frequency base
spkFreq         = spkFreq(spkFreq<=maxFreq);                               	% Truncate frequency base
spkPower     	= spkPower(1:length(spkFreq));                              % Truncate power values
spkPower        = fastsmooth(spkPower,sum(spkFreq<=lfpBoxCar),3);           % Smooth spike power spectrum
clear meanNormAC fftRes

% Compute the LFP power spectrum
lfpSignal       = cos(LFP_Phase);                                           % Convert LFP phase to simulated signal
lfpSignal       = lfpSignal - mean(lfpSignal);                              % Mean normalise
fftRes          = fft(lfpSignal, fftLength); clear lfpSignal              	% Compute the Fourier transform
lfpFreq         = linspace(0,1,fftLength/2+1)./(2*dt);                      % Compute frequency base
lfpFreq        	= lfpFreq(lfpFreq<=maxFreq);                                % Truncate frequency base
lfpPower        = (fftRes.*conj(fftRes))/length(fftRes); clear fftRes       % Convert from amplitude to power
lfpPower        = lfpPower(1:length(lfpFreq));                              % Truncate power values
lfpPower        = fastsmooth(lfpPower,sum(lfpFreq<=lfpBoxCar),3);          	% Smooth LFP power spectrum

% Extract the phase of firing for all cells
allSpikePhase 	= zeros(ppBinSize,ppBinSize);
for module      = 1 : length(gridScales)
    for cell  	= 1 : nGCs
        centres = ((cell-1)/nGCs)*gridScales(module)-gridScales(module) : gridScales(module) : max(x_log) + gridScales(module);
        pos     = x_log(firingRates((module-1)*nGCs+cell,:)>0);
        phase   = mod(LFP_Phase(firingRates((module-1)*nGCs+cell,:)>0),2*pi)';
        [~,ind] = min(abs(repmat(pos,1,length(centres)) - repmat(centres,length(pos),1)),[],2);
        pos     = (pos - centres(ind)')./gridScales(module); clear centres ind
        % spikePhase  = [spikePhase ; [pos phase]];
        % scatter(pos,phase), axis square, xlim([-0.5 0.5]), ylim([0 2*pi]), pause
        phase   = histnd([pos phase], [-0.5 0 ; 0.5 2*pi], [1/ppBinSize 2*pi/ppBinSize]); clear pos
        allSpikePhase(:,:,(module-1)*nGCs+cell) = phase; clear phase
    end
end
clear module cell
end
