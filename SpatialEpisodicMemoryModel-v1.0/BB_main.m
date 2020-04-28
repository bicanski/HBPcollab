

%% OVERVIEW

% The BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)





%% Abbreviations, variable prefixes, list of subroutines


% PCs:     place cells
% BVCs:    boundary vector cells
% OVCs:    object vector cells
% H/HPC:   hippocampus
% ID/PR:   identity neurons in perirhinal cortex, used for texture of walls.
% oID/oPR: object identiy neurons
% PW:      parietal window, contains egocentric analogs of BVCs, is "canvas" for imagery
% oPW:     parietal window, contains egocentric analogs of OVCs, is "canvas" for imagery
% MTL:     Medial Temporal Lobe (means PCs in HPC, BVCs and perirhinal ID neurons)
% RSC:     retrospenial cortex, contains tranformation circuit between PW and BVCs
% HD(Cs):  head direction (cells)
% TR:      transformation, usually refers to the RSC layers (see RSC)
% IP:      inhibitory feedback neuron
% GC:      grid cells for imagined movement



% list of subroutines (BB_subr ...)

% setup_Two_Room.m        (creates and discretizes environment variables)
% envdata_Two_Room.m      (room geometry)
% intersection.m          (figures out where two lines intersect and returns both alphas)
% MakeWeights.m           (uses data created by the above script and makes the weights for MTL
% MakeHDWeights.m         (creates weights among HD cells)
% MakeRotIntWeights.m     (creates weights for rotational integration of HD)
% MakeTransIntWeights.m   (creates a matrix of multipliers for translational weights - we don't use these here)
% MakeTRWeights.m         (generates all of the transformation layer weights - BVCs to PW and vice versa)
% Grid_Activity.m          (used to take a boundary in space, discretize it and calculate the corresponding neural activity for neurons arranged on a polar grid covering the space)





%% SIMULATION TYPE


% set 'simflag' choose the type of simulation, see paper as well

% simflag =  0;   % for longer simulations to create data with sufficient coverage 
                  % for decent rate maps (without plotting/video recording)
                  % not the same as video 1 in the paper, that is simflag60

% simflag = 10;     % sim. 1.0 in paper, approach and encode an object and perform object-cued recall later on
simflag = 11;   % sim. 1.1 in paper, Papez Ciruit lesion no grid cells
% simflag = 12;   % sim. 1.2 in paper, Papez Ciruit lesion no grid cells, but pre encoded object
% simflag = 241;  % sim. 1.3 in paper, Novelty detection without HPC lesion: object is moved after encoding
% simflag = 242;  % sim. 1.4 in paper, full imagery attempt with HPC lesion: only coincidental loation match

% simflag = 21;   % sim. 2.1 in paper, BVC traces (memory) of removed barrier
% simflag = 26;   % same as simflag 21 but longer simulation without plotting to get nice rate maps
% simflag = 27;   % same as simflag 21 even longer
% simflag = 22;   % sim. 2.2 in paper, OVC traces (memory) of removed object
% simflag = 28;   % same as simflag 22 but longer simulation without plotting to get nice rate maps

% simflag = 31;   % sim. 3.1 in paper, residual activity of unattended object in oPW, 
                  % can infer identity of that object.

% simflag = 40;   % sim. 4.1 in paper, mental navigation via GCs acting of PCs.  
                  % A sweep across grid cells activates a sequence of PCs, shifts POV
                  % OVCs for object 3 reactivated when passing through encoding location, despite 
                  % imagined direction being novel => a novel imagined experience

% simflag = 50;   % sim. 5.1 in paper, planning engenders preplay-like activity in PCs
                  % cognitive map extended by associating BVCs to newly recriuted PCs when shortcut 
                  % actually traversed (i.e. after planning/preplay stage)

% simflag = 60;   % sim 0/video 1 in paper, empty env just to illustrate TR circuit video





%% RUNTIME PARAMETERS 


% for noise simulations, new random seed each iteration.

% stream0 = RandStream('mt19937ar','Seed',3);
% RandStream.setGlobalStream(stream0);

WTSdir  = 'WTS_BVCcorr';    % naming implies BVC/OVC/PW/TR reps. use increasing receptive field size with radial distance
WTSname = 'BVCcorr';        % 
DATAdir = 'data';           % make this directory to save data

dt           = 0.001;   
step         = 1;      

plotting     =  1;          % decide if to plot
modstep      =  10;         % 10 gives decent framerate for videos, also means plots updated every modstep steps
savestep     =  250;        % save every savestep timesteps, interval increased below for long sims
makemovie    =  1;          % determines whether or not we make movie
save_flag    =  1;
plt_ttl_flag =  1;          % flag for plotting title in plot. 1=perception based navi, 2=imag, 3=reestablishing perception ...
noise_flag1  =  0;          % add firing rate noise
noise_flag2  =  0;          % add lesion noise, i.e. kill random cells.
LesNoisePerc = 20;          % percentage of dead cells in all populations ecept HD (no redundancy with only 100 neurons)
FRNoiseLevel = 20;          % percentage of FR noise in all populations

if simflag == 0 || simflag == 26 || simflag == 27 || simflag == 28
    FRmap_flag = 1;         % save data for FR maps
    savestep   = 100000;    % don't want to save too much data dor long sims
    modstep    = 10;
else
    FRmap_flag = 0;
end

if makemovie && simflag>0   
    vidcommand = ['writerObj = VideoWriter(''videos/BBmodel_sim' int2str(simflag) '.avi'');'];
    eval(vidcommand);
    writerObj.FrameRate = 30;
    open(writerObj);
end





%% Connection Properties (Phis), scaling previously calculated weights


PW2TRphi   = 50;        % from parietal window to TR
TR2PWphi   = 35;        % ...
TR2BVCphi  = 30;        % less than TR2PW because BVCs also receive input from H and PR
BVC2TRphi  = 45;        % ...

HDphi      = 15;   
HD2IPphi   = 10;
HD2TRphi   = 15;   
Rotphi     = 4;  

IP2TRphi   = 90;        % scale strength of inhibition to TR
IP2oTRphi  = 90;

Hphi       = 25;        % 2 PCs to PCs
H2BVCphi   = 2860;   
H2PRphi    = 6000;
BVC2Hphi   = 437; 
BVC2Hphi_b = 437;
BVC2PRphi  = 75;
BVC2OVCphi = 0;
BVC2oPRphi = 0;
PR2Hphi    = 25;
PR2BVCphi  = 3;

GC2PCphi_base = 3;
GC2PCphi      = GC2PCphi_base;

oPW2TRphi  = 60;        %
TR2oPWphi  = 30;        %
TR2OVCphi  = 60;        %
OVC2TRphi  = 30;        % 140

OcueScale     = 0.3;    % this is essentially like a phi for the perceptual drive
BcueScale     = 1.6;     
TDBU_Scale    = 0.05;   
TDBU_ScaleBVC = 0.05;   % because of upscaled weight for recall
TDBU_ScaleGC  = 0.1;
oPRCuescale   = 160;    
Pmod          = 0.05;
Imod          = 0.05;

H2OVCphi   = 2.1;
oPR2OVCphi = 7.2;

H2oPRphi   = 1;
% this should not be zero because in that case if the same OVC was active
% for two differnt objects (at different times), then inhibiting that object cell, the OVC->oPR
% connection will trigger an unwanted oPR neuron, which (if that neuron has
% been encoded) will change the HD before any mental nav can take place due to oPR2HD connections

OVC2Hphi   = 5;
OVC2BVCphi = 0;         % not used, though in principle a valid connection. 
                        % I.e. context could reactivete object location reps and vice verse
OVC2oPRphi = 5; 
OVC2OVCphi = 1;      
oPW2oPWphi = 0;         % considered this connection for inhibition but not necessary atm
oPR2Hphi   = 100;
oPR2PWphi  = 0;       
oPR2HDphi  = 0;
oPR2BVCphi = 0;
oPR2oPRphi = 115;       % together with inh necessary for proper function

% Inhibitory phis, phis scale weights. E.g. imagine a Gaussian conncetion
% profile and subtract a constant number, yield local excitation (peak of
% Gaussian) and long-rage (i.e. to remaining cells) inhibition. Actual
% weight matrices calculated elsewhere, see readme etc.

H_inhib_phi       = 0.4;    
BVC_inhib_phi     = 0.2;    
PR_inhib_phi      = 9;      
HD_inhib_phi      = 0.4;    
TR_inhib_phi      = 0.075;  
oTR_inhib_phi     = 0.1;
PW_inhib_phi      = 0.165;  
OVC_inhib_phi     = 0.7;    
oPW_inhib_phi     = 0.2;    
oPR_inhib_phi     = 1;      
OVC2oPR_inhib_phi = 0;      





%% Neuron parameters, load weights, initialize neurons


% Neuron Properties

base_tau = 0.02;
Htau     = base_tau;   Hbeta    = 0.1;   Halpha   = 5;
BVCtau   = base_tau;   BVCbeta  = 0.1;   BVCalpha = 5;
OVCtau   = base_tau;   OVCbeta  = 0.1;   OVCalpha = 5;
PRtau    = base_tau;   PRbeta   = 1;     PRalpha  = 5;
HDtau    = base_tau;   HDbeta   = 0.1;   HDalpha  = 5;
TRtau    = base_tau;   TRbeta   = 0.1;   TRalpha  = 5;
oTRtau   = base_tau;   oTRbeta  = 0.1;  oTRalpha  = 5;
PWtau    = base_tau;   PWbeta   = 0.1;   PWalpha  = 5;
oPWtau   = base_tau;   oPWbeta  = 0.1;  oPWalpha  = 5;
IPtau    = base_tau;   IPbeta   = 0.1;   IPalpha  = 50;
GCtau    = base_tau;   GCbeta   = 0.1;   GCalpha  = 5;     % GC ADDITION
ICtau    = base_tau;  % comp current 2*base_tau;




% load Weights

if simflag == 50
    eval(['load ' WTSdir '/Weights_VR_MTL_preplay_env_closed_EXT_' WTSname '']);
elseif simflag == 21 || simflag == 26
    eval(['load ' WTSdir '/Weights_VR_MTL_inserted_barrier_' WTSname '']);
else
    eval(['load ' WTSdir '/Weights_VR_MTL_square_room_' WTSname '']);
end
eval(['load ' WTSdir '/HDWeights']);
eval(['load ' WTSdir '/TRWeights_NEW2_18kSP_' WTSname '']);
eval(['load ' WTSdir '/RotIntWeights']);
eval(['load ' WTSdir '/GC2PCwts_BB']);                 
STD_GC2PCwts = GC2PCwts;
if simflag ~= 50
    GC2PCwts = GC2PCwts*2;  % not really necessary, could have been the same from 
                            % the start and in all sims. Stronger GC2PW weights 
                            % simply make PCs more stable.
end




% Set some number values

N_H    = size(H2Hwts,1);
N_BVC  = size(H2BVCwts,1);
N_PR   = size(H2PRwts,1);
N_HD   = size(HD2HDwts,1);
BVCone = ones(N_BVC,1);
N_GC   = size(GC2PCwts,2);       
N_oPW  = size(H2BVCwts,1);       % introduce Object vector cells (OVCs/OVCs) and the parietal analog oPW
N_OVC  = size(H2BVCwts,1);       % not yet clear if in CA1 (so in MTL) or in LEC/MEC




% TR neurons project 1-to-1 to BVCs and back

BVC2TRwts = eye(N_BVC);
TR2BVCwts = eye(N_BVC);




% Set initial activities to zero for all neurons

% no need to numerically integrate the IP neuron activity, corresponds to summed HD activity
% remains nearly constant over entire simulation, hence interneuron firing rate will also be a constant

Hact    = zeros(N_H,1);       % HPC / MTL
BVCact  = zeros(N_BVC,1);     % BVC / MTL
PRact   = zeros(N_PR,1);      % PR  / MTL
PWact   = zeros(N_BVC,1);     % PW
HDact   = zeros(N_HD,1);      % HD  / TRANS
OVCact  = zeros(N_OVC,1);
oPWact  = zeros(N_oPW,1);

% next loop is slow-ish in Matlab but saves writing out 40 
% lines for 2x20 layers, I opted for readbility/conciseness here

for i = 1:20                  
    eval(['TRact' int2str(i) ' = zeros(N_BVC,1);']);
    eval(['oTRact' int2str(i) ' = zeros(N_BVC,1);']);
end

eval(['load ' WTSdir '/GC_FRmaps_BB.mat']);              % GC rate are sampled from their firing rate maps. No rate dynamics
GCs = reshape(GC_FRmaps_SD,[44,44,N_GC]);





%% Training data, room parameters, RF grids


if simflag == 50   % room is different in sim 5.0 (planning/preplay)
    eval(['load ' WTSdir '/TrainingData_preplay_env_closed_EXT_' WTSname '']); 
else
    eval(['load ' WTSdir '/TrainingData_square_room_' WTSname '']);
    %eval(['load ' WTSdir '/TrainingData_inserted_barrier_' WTSname '']);
end
% if simflag == 26
%     eval(['load ' WTSdir '/TrainingData_inserted_barrier_' WTSname '']); 
% end


Hres         = 0.5;       % Place cells on grid with 2 cells per unit length, 22 units map onto 2 meters
maxR         = 16;        % BVCs to cover space up to 16 units from origin
maxX         = maxTrainX;
maxY         = maxTrainY;
minX         = 0;
minY         = 0;
polarDistRes = 1;         % One BVC per unit radial length
polarAngRes  = 2*pi/51;   % Angular separation of neurons on radial grid, 
                          % 51 radia because of poligonal edges in surface plots. 
                          % 50 with a scatterplot would have also 
                          % worked, but doesn't matter. Results are the same

% Set up HPC grid
NHx     = round((maxX-minX)/Hres);
NHy     = round((maxY-minY)/Hres);
Hx      = [minX+Hres/2:Hres:minX+(NHx-0.5)*Hres];
Hy      = [minY+Hres/2:Hres:minY+(NHy-0.5)*Hres];
[HX,HY] = meshgrid(Hx,Hy);

% values like maxX etc. come from the load above.
% should be selfexplanatory, e.g. maxX is the maximum X coordinate, Hres,
% the resolution of the (H)ippocampus place cell grid etc.

% Set up BVC grid
NBVCR        = round((maxR)/polarDistRes);  % number of BVC units radially, but note, separation changes below
NBVCTheta    = floor((2*pi-0.01)/polarAngRes)+1;
% polarDist    = [polarDistRes/2:polarDistRes:(NBVCR-0.5)*polarDistRes];   %

% The above line was used for constant radial separation between receptive fields,
% like in the BBB model, BB model uses more realistic (if modestly) increasing radial separation of RFs.

[polarDist]  = BB_subr_radscaling(maxR);   %%% SUBROUTINE CALL

polarAng     = [0:(NBVCTheta)*polarAngRes/(NBVCTheta-1):NBVCTheta*polarAngRes];
[pDist,pAng] = meshgrid(polarDist,polarAng);
[BVCX,BVCY]  = pol2cart(pAng,pDist);

% similar to HPC, but a polar grid. compare to the graphs in the paper.
% Plotting these grids gives you the PW and BVC activities.

% Set up HD ring
HDangles         = [0:2*pi/(N_HD-1):2*pi] + pi/2;
[HDDist,HDAngle] = meshgrid([1 1.5],HDangles);
[HDX,HDY]        = pol2cart(HDAngle,HDDist);

% Clean up
clear Hx Hy polarAng pDist pAng maxX maxY minX minY Hres ...
    HDangles HDDist HDAngle maxR %polarDistRes polarAngRes

if simflag == 50  
    load roomGridPrePlay_ext.mat
    
    RoomSize             = size(roomGridPrePlay);
    Xrange               = RoomSize(1,1);
    Yrange               = RoomSize(1,2);
    red_grid             = roomGridPrePlay(2:end-1,2:end-1);
    Xbarrier_top_min     = find(red_grid(Yrange/2,:)==1,1,'first')+1; % max is at wall
    Xbarrier_top_min_tmp = find(red_grid(Yrange/2,:)==1,2,'first')+1;  % in case barrier is removed
    Xbarrier_top_max     = Xbarrier_top_min_tmp(2);
    Ybarrier_top         = find(red_grid(:,end)==1,1,'last')+1;
    Xbarrier_bot_min     = Xbarrier_top_min;  % max is at wall
    Ybarrier_bot         = find(red_grid(:,end)==1,1,'first')+1;
end





%% Sim-no dependent trajectory variables, object locations etc.


oldCoordsTMP = [];   % dummy
CoordsTMPmem = [];

% I left simflag 1 parameters in here so one does not need to go to a
% separete file to understand what is happeing, but similar parameters for
% other simulations have been moved to separate script to make BB_main more
% readable. For consitent naming of files it is also named 'BB_subr...'
% even though technically it does not implement any computations and only
% sets parameters.

if simflag == 10                % encode and object-cued recall
    Xag              = 5; 
    Yag              = 10;
    HDag             = pi;      % initial HD
    Vlin             = 2.5;     % linear velocity start value
    percep_flag      = 1;       % as long as this is 1 we are in bottom-up mode
    imag_flag        = 0;       % when this is 1, the imagery setup is used, top-down mode
    navi_flag        = 0;       % when 1 the agent starts to move
    move2nextTarget  = 0;       % what does this do? no idea.
    locateNextTarget = 0;       % ...
    rotResolved      = 1;       % set to 1 after rotation is resolved, same at start
    target_no        = 0;       % index of next target, init with 0
    targetlist = [ 5,  6; ...   % where to go
                  10, 10; ...
                  16,  8; ...
                  16, 12];    
    SandT = 3;                  % "stop and think", engage top-down mode near trajectory point Xst,Yst
    targetlist_imag_nav = [];   % where to go in imagined movement
    target_no_imag_nav  = 0;    % ...
    CoordsTMP    = [ 12, 12];   % object coordinates when not using externally supplied file
    
    Nobj         = length(CoordsTMP(:,1));     % no. of objects set above
    ObjEncThresh = 6.0;                        % Encode object when closer than this distance
end

BB_subr_TrajectoryParas;   % Trajectory parameters etc. for other simulations

oldVlin = Vlin;            % in case we need to stop and resume





%% Agent model navigation and imagination/perception setup (cueing)


navi_flag_imag            = 0;    % start out without imagined movement, turn on later on demand
move2nextTarget_imag_nav  = 0;    % flag that indicates the agent should move to the next target in imagined movement
locateNextTarget_imag_nav = 0;
rotResolved_imag_nav      = 1;    % set to 1 after rotation in imagery is resolved, same at start
target_no_imag_nav        = 0;    % ...
DWELL      = 150;                 % dwelltime at target, including initial location, arbitrary
dwell      = DWELL;
n_targets  = length(targetlist(:,1));


if simflag == 26
    eval(['load ' WTSdir '/Roomdataforperceptiondrive_inserted_barrier_' WTSname '']);
    BndryPtX_insB = BndryPtX;
    BndryPtY_insB = BndryPtY;   % the rest will be overwritten by the next if
else
    BndryPtX_insB = [];
    BndryPtY_insB = [];
end
if simflag == 50
    eval(['load ' WTSdir '/Roomdataforperceptiondrive_preplay_env_closed_EXT_' WTSname '']);
else
    eval(['load ' WTSdir '/Roomdataforperceptiondrive_square_room_' WTSname '']);
    %eval(['load ' WTSdir '/Roomdataforperceptiondrive_inserted_barrier_' WTSname '']);
end


% determine cue for entire field of view, do this also during navigation.
% Meaning, calculate drive to PW
[egocues,~,~,~,~,~,~,~] = BB_subr_PdrivePW_withObj(r0,dir,line,BndryPtX,BndryPtY,Xag,Yag,HDag,0,0,0);   


% for imagined movement
DWELL_imag_nav            = 150;       % dwelltime in imagery
dwell_imag_nav            = DWELL;
move2nextTarget_imag_nav  = 0;   
locateNextTarget_imag_nav = 0;
rotResolved_imag_nav      = 1;        
n_targets_imag_nav  = length(targetlist(:,1));


% Arrays to memorize imagined trajectory for plotting
AllImagX = [];
AllImagY = [];


% initialize oPW
ObjPtX  = zeros(Nobj,5);
ObjPtY  = zeros(Nobj,5);
ObjCenX = zeros(Nobj,1);
ObjCenY = zeros(Nobj,1);
for i = 1:Nobj
    ObjPtX(i,:)  = CoordsTMP(i,1)-0.2:0.1:CoordsTMP(i,1)+0.2;   % add some extent so that the objects are not dim-less points
    ObjCenX(i,1) = CoordsTMP(i,1);
    ObjPtY(i,:)  = CoordsTMP(i,2)-0.2:0.1:CoordsTMP(i,2)+0.2;
    ObjCenY(i,1) = CoordsTMP(i,2);
end
% calculate drive to oPW, similar to above for boundaries
[OBJcues,VBX_rt,VBY_rt,L_r,BX,BY,TX,TY] = BB_subr_PdrivePW_withObj(r0,dir,line,ObjPtX,ObjPtY,Xag,Yag,HDag,0,0,1);


% more cueing, uses derived boundary outlines in egocues, see just above
% these scripts basically calculate the current to be injected into the
% various neuron populations (EgoCue for PW, HDcue for HDCs, PRcue for perirhinal)
EgoCue      = 40 * BB_subr_cue_PWact_multiseg(0.5,egocues);     
HDCue       =  0 * BB_subr_cue_HDact(HDag);
HDCue_init  = 40 * BB_subr_cue_HDact(HDag);
oHDCue      =  0 * HDCue;
PRCue       = zeros(N_PR,1);                                
old_HD      = nan;                   % will use this to remember perceptual HD during imagery

CueDur         = DWELL;              % for initial cueing at start of sim
PRcue_percep   = 0;                  % cues to PR neurons
oPRCue         = 0;                  % cues to oPR neurons

ObjCue_percep   = 0;                 % perceptual cue during sim, corresponds to sensory input
ObjCue_percepIM = 0;                 % cue during imagery, corresponds to cue to imagine something

ObjEncoded       = zeros(Nobj,1);    % keepigng track of which object has been encoded
ObjEncoded_prev  = ObjEncoded;
ObjRecalled      = zeros(Nobj,1);    % keepigng track of which object has been recalled
ObjRecalled_prev = ObjRecalled;

HDestim             = HDag;          % for later use, estimate of HD
HDestim_prev        = HDag;          % for later use, in previous timestep (see navigation of agent)

toturn              = 0;             % ..., how far does the agent still need to turn (ongoing, i.e. changes as total angle diminshes).
toturn_imag_nav     = 0;
CWturn              = 0;             % clockwise or
CCWturn             = 0;             % counterclockwise
ang2f_tot           = 0;             % total angle to turn before we started turing
dist2f_cur          = 0;             % total distance to target location
ang2f_tot_imag_nav  = 0;             % total angle to turn before we started turing
dist2f_cur_imag_nav = 0;





%% Allocate arrays for correlation analysis


% Arrays for memorizing firing rates during encoding (PE), recall (IM), and
% random times in bottom-up mode (PErand)

CCnt_rand  = 1;   % count random snapshots

PRcorr_IM  = zeros(N_PR,Nobj);     PRcorr_PE = zeros(N_PR,Nobj);     PRcorr_PErand = zeros(N_PR,Nobj);
oPRcorr_IM = zeros(Nobj,Nobj);    oPRcorr_PE = zeros(Nobj,Nobj);    oPRcorr_PErand = zeros(Nobj,Nobj);
OVCcorr_IM = zeros(N_BVC,Nobj);   OVCcorr_PE = zeros(N_BVC,Nobj);   OVCcorr_PErand = zeros(N_BVC,Nobj);
BVCcorr_IM = zeros(N_BVC,Nobj);   BVCcorr_PE = zeros(N_BVC,Nobj);   BVCcorr_PErand = zeros(N_BVC,Nobj);
PCcorr_IM  = zeros(N_H,Nobj);      PCcorr_PE = zeros(N_H,Nobj);      PCcorr_PErand = zeros(N_H,Nobj);
PWcorr_IM  = zeros(N_BVC,Nobj);    PWcorr_PE = zeros(N_BVC,Nobj);    PWcorr_PErand = zeros(N_BVC,Nobj);
oPWcorr_IM = zeros(N_BVC,Nobj);   oPWcorr_PE = zeros(N_BVC,Nobj);   oPWcorr_PErand = zeros(N_BVC,Nobj);

% In retrospect should have also done correslations between
% no-noise and noise conditions, to assess how well an old memory can 
% be recalled after neuron loss. But comparing noisy encoding to noisy 
% recall and noisy random snapshots to noisry recall is still interesting
% since it tells us that the memorized represenations are still separable 
% from random times.





%% Set up attention and encoding (new, empty weight matrices)


ObjAttThresh = 10;                % Attend to all object within this radius (in front of you, 180 deg FOV)
encoded      = 0;                 % tells us if the next object is already encoded ...
encoding     = 0;                 % ... or still encoding
bup          = 0;
N_oPR        = Nobj;              % number of object identity cells, we will only use 3 for now, but this scales easily
oPRact       = zeros(N_oPR,1);    % activation array of object identiy cells
ATTcounter   = 1;                 % for crude attentional modulation
ATTcycle     = 600;                
ATTperObj    = ATTcycle/Nobj;     % devide attention equally among all objects, but sequentially
pseudoadap   = 1;                 % adaptation 
oPR_drive    = zeros(Nobj,1);     % this will be the sensory drive to oPR neurons

H2oPRwts   = zeros(N_oPR,N_H);    % potential new weights to and from oID cells
H2OVCwts   = zeros(816,N_H);
OVC2Hwts   = zeros(N_H,816);
OVC2BVCwts = zeros(816,816);
BVC2OVCwts = zeros(816,816);
OVC2OVCwts = zeros(816,816);
OVC2oPRwts = zeros(N_oPR,816);
oPR2OVCwts = zeros(816,N_oPR);
oPW2oPWwts = zeros(816,816);
oPR2Hwts   = zeros(N_H,N_oPR);
oPR2PWwts  = zeros(816,N_oPR);    % will only be instantiated once an object is encoded
oPR2HDwts  = zeros(100,N_oPR);    % will only be instantiated once an object is encoded
BVC2oPRwts = zeros(N_oPR,816);    % for now only via PCs, so we don't use the next 3 yet. Could in future
oPR2BVCwts = zeros(816,N_oPR);
oPR2oPRwts = zeros(N_oPR,N_oPR);





%% load and rescale weights, apply inh phis


% RESCALE WEIGHTS WITH INHIBITION
% Only do this subtraction once
% interpretation: this will turn some weights negative, thus preventing activation from spreading throughout
% the PC or HD netwrork where weights were all positive, calculated based on Hebbian correlations
% basically imagine a Gaussian (in the case of HD weights) where the tails
% fall below zero.

H2Hwts     = H2Hwts     - H_inhib_phi;
BVC2BVCwts = BVC2BVCwts - BVC_inhib_phi;
PR2PRwts   = PR2PRwts   - PR_inhib_phi;
OVC2OVCwts = OVC2OVCwts - OVC_inhib_phi;
oPR2oPRwts = eye(Nobj)*(oPR_inhib_phi) - oPR_inhib_phi;        
OVC2oPRwts = OVC2oPRwts - OVC2oPR_inhib_phi;               % to help newly driven RF to overcome previously active oPR neuron
HD2HDwts   = HD2HDwts   - HD_inhib_phi;

TR_bath    = TR_inhib_phi*BVCone;                          % BVC cone is just an array of ones, for gobal inhibition to be applied later
oTR_bath   = oTR_inhib_phi*BVCone;                         % ported this convention from the BBB model, could be improved
PW_bath    = PW_inhib_phi*BVCone;
oPW_bath   = oPW_inhib_phi*BVCone;





%% decide when and how to recall

recallobj = 0;                        % flag will tell us to recall, here a dummy, is reset in BB_subr_BUvsTU.m
% if simflag==21
%     imag_cuedur = 5000;
% else
    imag_cuedur = 2400;               % how long do we cue with the object identity
%end
if simflag==50 || simflag == 40
    imag_cuedur_nav = 10000;          % dummy because we enforce a break when an imagined location is reached
else
    imag_cuedur_nav = 0;              % only do imagined movement in sims 4 and 5
end
switchdur       = 600;                % how much time do we give the netweork for switiching between modes (recall vs percep)
                                      % number value comes from old BBB,
                                      % where it was equated to a theta cycle
imag_start      = 140000;             % start to imagine at this time pt, init with unaatainable value
imag_start_nav  = 140000;             % same for mental navigation ...
imag_loc        = 1;                  % Starting imagination at a given location, not time, this is a flag, not actual location




%% INITIALIZE ACTIVITIES AND RATES

% Calculate initial derivatives (ks), meaning just the decay term, i.e.
% minus the initial value devided by the time constant.
% Calculate activity, then send it through a sigmoid nonlinearity

k_H   = -Hact;                           
k_BVC = -BVCact;
k_PR  = -PRact + (CueDur>step)*PRCue;    % use initial values  we calculated above
k_oPR = -oPRact;                         
k_HD  = -HDact;                          % use initial values we calculated above
k_PW  = -PWact + (CueDur>step)*EgoCue;
k_OVC = -OVCact;
k_oPW = -oPWact;
for i = 1:20                             % again, lots of 'eval' are slower, but more readable
    eval(['k_TR' int2str(i) ' = -TRact' int2str(i) ';']);
    eval(['k_oTR' int2str(i) ' = -oTRact' int2str(i) ';']);
end

% Do first activity update
Hact   = Hact   + dt/Htau*k_H;
BVCact = BVCact + dt/BVCtau*k_BVC;
PRact  = PRact  + dt/PRtau*k_PR;
oPRact = oPRact + dt/PRtau*k_oPR;                
HDact  = HDact  + dt/HDtau*k_HD;
PWact  = PWact  + dt/PWtau*k_PW;
OVCact = OVCact  + dt/OVCtau*k_OVC;
oPWact = oPWact  + dt/oPWtau*k_oPW;
for i = 1:20
    eval(['TRact' int2str(i) ' = TRact' int2str(i) ' + dt/TRtau*k_TR' int2str(i) ';']);
    eval(['oTRact' int2str(i) ' = oTRact' int2str(i) ' + dt/TRtau*k_oTR' int2str(i) ';']);
end

% Calculate rates based on activities
H_rate   = 1./(1+exp(-2*Hbeta*(Hact-Halpha)));
BVC_rate = 1./(1+exp(-2*BVCbeta*(BVCact-BVCalpha)));
PR_rate  = 1./(1+exp(-2*PRbeta*(PRact-PRalpha)));
oPR_rate = 1./(1+exp(-2*PRbeta*(oPRact-PRalpha)));    % object identity cells, use same beta and alpha as other PR neurons
HD_rate  = 1./(1+exp(-2*HDbeta*(HDact-HDalpha)));
PW_rate  = 1./(1+exp(-2*PWbeta*(PWact-PWalpha)));
OVC_rate = 1./(1+exp(-2*OVCbeta*(OVCact-OVCalpha)));
oPW_rate = 1./(1+exp(-2*oPWbeta*(oPWact-oPWalpha)));
for i = 1:20
    eval(['TR_rate' int2str(i) '  = 1./(1+exp(-2*TRbeta*(TRact' int2str(i) '-TRalpha)));']);
    eval(['oTR_rate' int2str(i) '  = 1./(1+exp(-2*TRbeta*(oTRact' int2str(i) '-TRalpha)));']);
end

% same for grid cells from the look-up table
GC2PCinputs_prev = zeros(N_GC,1);
GC_rate_prev     = zeros(N_GC,1);
GC_rate          = 0 * squeeze(GCs(1,1,:));




%% Extra stuff for preplay sim

if simflag == 22 || simflag == 23 || simflag == 12 || simflag == 27 || simflag == 28
    eval(['load ' DATAdir '/BB_vid_sim10_T13953_newBVCs H2OVCwts OVC2Hwts oPR2OVCwts OVC2oPRwts oPR2HDwts oPR2Hwts H2oPRwts']);
end
if simflag == 24
    eval(['load ' DATAdir '/BB_vid_sim10_T13953_newBVCs H2OVCwts OVC2Hwts oPR2OVCwts OVC2oPRwts oPR2HDwts oPR2Hwts H2oPRwts']);
    oPR2OVCwts = [oPR2OVCwts, zeros(length(oPR2OVCwts),1)];   OVC2oPRwts = oPR2OVCwts';
    oPR2HDwts  = [oPR2HDwts, zeros(length(oPR2HDwts),1)];
    oPR2Hwts   = [oPR2Hwts, zeros(length(oPR2Hwts),1)];   H2oPRwts = oPR2Hwts';
end

% In the preplay/planning simulation (sim_flag 50)
% changes for sealed-off part of arena introduce all this messy
% overhead, because the reservoir place cells, random connections
% from grid cell, variables for tracking activity and for plotting
% etc. need to be defined

BB_subr_PreplayOverhead;   

% similarly to above these parameters have been moved to separate 
% script to make BB_main more readable. For consitent naming of 
% files it is also named 'BB_subr...' 





%% Imagined Movement and multiple static imagination episodes 

Xag_imag_nav     = -1;
imag_end_flag    = 0;
Yag_imag_nav     = 0;
imag1            = 1;
imag2            = 0;                      % reset later in preplay sim
first_imag_HD    = nan;
Xi_prev          = round(Xag_imag_nav);
Yi_prev          = round(Xag_imag_nav);
statechanges     = 0;                      % track how often we switch from top-down to bottom up
percep_flag_prev = percep_flag;
nobarrier        = 0;

oPRHDchange      = 1;                      % once we have created a point of view we want to 
                                           % release the HD ring from influence from oPR neurons

  
                                                  
                                                  
%% Theta modulation, PC FB, Correlation arrays,                                                   
                                                  
I_comp           = 0;       % compensatory current to keep PC activty in check
                            % was added to have siilar amount of active PCs during 
                            % preplay as during normal sim.
trackI_c         = [];      % Can be used to track and plot I_comp
                   
% For sims 2.X, with smooth, periodic top-down/bottom-up switch
bleed            = 1;       % for allowing memory to bleed into perceptual rep
perbl            = 1;
Bcount           = 1;


OVCs_o1_imag   = [];        % for the correlation analysis for object novelty (simflags 241/242)
OVCs_o2_imag   = [];
OVCs_o1_percep = [];
OVCs_o2_percep = [];
oPW_o1_imag    = [];
oPW_o2_imag    = [];
oPW_o1_percep  = [];
oPW_o2_percep  = [];





%% SET UP ARRAYS TO ACCUMULATE DATA FOR FR MAPS

X_for_FRM        = [];
Y_for_FRM        = [];
HD_for_FRM       = [];
time_at_XY       = zeros(44,44);
time_at_HD       = zeros(60,1);
CumPCrate_at_XY  = zeros(44,44,N_H);
CumBVCrate_at_XY = zeros(44,44,N_BVC);
CumOVCrate_at_XY = zeros(44,44,N_BVC);
CumoPRrate_at_XY = zeros(44,44,Nobj);
for i = 1:20
    eval(['CumTRrate' int2str(i) '_at_XY = zeros(44,44,N_BVC);']);
end
if FRmap_flag
    Vlin = 2.5;   % override slow Vlin from above if in sim 0
end





%% Add noise

if noise_flag1 % firing rate noise
    
    H_FRnoise   = -rand(N_H,1)/100*FRNoiseLevel;
    BVC_FRnoise = -rand(N_BVC,1)/100*FRNoiseLevel;
    OVC_FRnoise = -rand(N_BVC,1)/100*FRNoiseLevel;
    PW_FRnoise  = -rand(N_BVC,1)/100*FRNoiseLevel;
    oPW_FRnoise = -rand(N_BVC,1)/100*FRNoiseLevel;
    HD_FRnoise  = -rand(N_HD,1)/100*FRNoiseLevel;
    
    for i = 1:20
        eval(['TR_FRnoise' int2str(i) ' = -rand(N_BVC,1)/100*FRNoiseLevel;']);
        eval(['oTR_FRnoise' int2str(i) ' = -rand(N_BVC,1)/100*FRNoiseLevel;']);
    end 
    
end

if noise_flag2 % dead cells
    
    H_indArr  = ones(N_H,1);             % indArr is an array of indices
    Nles_H    = round(N_H/100*LesNoisePerc);
    H_allinds = randperm(N_H)';
    H_lesinds = H_allinds(1:Nles_H,1);   % these will be lesioned
    H_indArr(H_lesinds) = 0;             % mask
    
    BVC_indArr  = ones(N_BVC,1);
    Nles_BVC    = round(N_BVC/100*LesNoisePerc);
    BVC_allinds = randperm(N_BVC)';
    BVC_lesinds = BVC_allinds(1:Nles_BVC,1);   % these will be lesioned
    BVC_indArr(BVC_lesinds) = 0;               % mask
    
    OVC_indArr  = ones(N_BVC,1);
    Nles_OVC    = round(N_BVC/100*LesNoisePerc);
    OVC_allinds = randperm(N_BVC)';
    OVC_lesinds = OVC_allinds(1:Nles_OVC,1);   % these will be lesioned
    OVC_indArr(OVC_lesinds) = 0;
    
    PW_indArr  = ones(N_BVC,1);
    Nles_PW    = round(N_BVC/100*LesNoisePerc);
    PW_allinds = randperm(N_BVC)';
    PW_lesinds = PW_allinds(1:Nles_PW,1);   % these will be lesioned
    PW_indArr(PW_lesinds) = 0;              % mask
    
    oPW_indArr  = ones(N_BVC,1);
    Nles_oPW    = round(N_BVC/100*LesNoisePerc);
    oPW_allinds = randperm(N_BVC)';
    oPW_lesinds = oPW_allinds(1:Nles_oPW,1);   % these will be lesioned
    oPW_indArr(oPW_lesinds) = 0;              % mask
    
    for i = 1:20
        eval(['TR_indArr' int2str(i) ' = ones(N_BVC,1);']);
        Nles_TR = round(N_BVC/100*LesNoisePerc);
        TR_allinds = randperm(N_BVC)';
        TR_lesinds = TR_allinds(1:Nles_TR,1);
        eval(['TR_indArr' int2str(i) '(TR_lesinds) = 0;']);
    end
    for i = 1:20
        eval(['oTR_indArr' int2str(i) ' = ones(N_BVC,1);']);
        Nles_oTR = round(N_BVC/100*LesNoisePerc);
        oTR_allinds = randperm(N_BVC)';
        oTR_lesinds = oTR_allinds(1:Nles_oTR,1);
        eval(['oTR_indArr' int2str(i) '(oTR_lesinds) = 0;']);
    end
    
    HD_indArr  = ones(N_HD,1);
    Nles_HD    = round(N_HD/100*LesNoisePerc);
    HD_allinds = randperm(N_HD)';
    HD_lesinds = HD_allinds(1:Nles_HD,1);   % these will be lesioned
    HD_indArr(HD_lesinds) = 0;
    
    Nles_GC    = round(N_GC/100*LesNoisePerc);
    GC_allinds = randperm(N_GC)';
    GC_lesinds = GC_allinds(1:Nles_GC,1);   % these will be lesioned
    GC2PCwts(:,GC_lesinds) = 0;
    
end





%% integration loop


% These next few lines can be commented in to avoid the encoding 
% phase of sim. 241 when looking at sim. 242, meaning we can simply
% load the learned weights and skip to recall phases of both sims  
%
% if simflag == 241 || simflag == 242
%     load HPClesionTMP_BVCcorr
%     if simflag == 241
%         target_no = 5;
%     end
%     if simflag == 242
%         target_no = 1;
%         H2oPRwts = H2oPRwts*0;
%         H2OVCwts = H2OVCwts*0;
%         H2BVCwts = H2BVCwts*0;
%         H2PRwts  = H2PRwts*0;
%         H2oPRwts = H2oPRwts*0;
%         oPR2OVCwts = oPR2OVCwts*3.5;  % to get similar levels of firing, the brain would compensate for lost inputs
%     end
%     ObjEncoded = [1;1];
%     ObjEncoded_prev = ObjEncoded;
% end



while target_no<=n_targets  % run sim until all target pts have been visited by the agent
    

    
    if step == 100
        navi_flag = 1;
    end
    
    
    
    % saving data fro FR maps
    if FRmap_flag
        Xind                          = round(Xag*2);
        Yind                          = round(Yag*2);
        if HDag == 0
            HDag = 2*pi;
        end
        HDind                         = ceil(HDag/(2*pi)*360/6);   % 6 degree bins
        HD_for_FRM                    = [HD_for_FRM ; round(HDag/(2*pi)*360/6)*6;];
        X_for_FRM                     = [X_for_FRM ; round(Xag*2)/2];
        Y_for_FRM                     = [Y_for_FRM ; round(Yag*2)/2];
        time_at_XY(Xind,Yind)         = time_at_XY(Xind,Yind) + 1;
        time_at_HD(HDind)             = time_at_HD(HDind) + 1;
        H_rateFR   = H_rate.*(H_rate>0.05);
        BVC_rateFR = BVC_rate.*(BVC_rate>0.05);   % remove small amplitude background,
        OVC_rateFR = OVC_rate.*(OVC_rate>0.05);   % could be removed by slightly different transfer function parameters
        oPR_rateFR = oPR_rate.*(oPR_rate>0.05);   % just never got around to it
        CumPCrate_at_XY(Xind,Yind,:)  = squeeze(CumPCrate_at_XY(Xind,Yind,:)) + H_rateFR;
        CumBVCrate_at_XY(Xind,Yind,:) = squeeze(CumBVCrate_at_XY(Xind,Yind,:)) + BVC_rateFR;
        CumOVCrate_at_XY(Xind,Yind,:) = squeeze(CumOVCrate_at_XY(Xind,Yind,:)) + OVC_rateFR;
        CumoPRrate_at_XY(Xind,Yind,:) = squeeze(CumoPRrate_at_XY(Xind,Yind,:)) + oPR_rateFR;
        for i = 1:20
            eval(['TR_rateFR' int2str(i) ' = TR_rate' int2str(i) '.*(TR_rate' int2str(i) '>0.05);']);
            eval(['CumTRrate' int2str(i) '_at_XY(Xind,Yind,:) = squeeze(CumTRrate' int2str(i) '_at_XY(Xind,Yind,:)) + TR_rateFR' int2str(i) ';']);
        end
    end
    
    
    
    % mental navigation specific stuff
    if simflag == 40
        if target_no==SandTnav && imag_start_nav>99999
            imag_start_nav  = step+1;
        end
        if Xag_imag_nav>=7  % once we embark on an imagined trajectory, prevent further drive from oPR to HD
            oPRHDchange = 0;
        end
        if target_no_imag_nav==1 && Xag_imag_nav>=(targetlist_imag_nav(1,1)-0.5) && imag_end_flag==0  % reached imagined target
            % using Xag_imag_nav>= as criterion because horizontal movement
            % from left to right, not very general as a criterion
            imag_cuedur_nav  = step - imag_start_nav; % this should stop imagined movement
            locateNextTarget = 1;
            rotResolved      = 1;
            imag_end_flag    = 1;
        end
    end
    
    
    
    % These next two if-statements look messy and they are, but work fine. 
    % Should implement cleaner top-down/botom-up switching code to make it 
    % easier to just specify times or locations for imagery episodes, 
    % maybe even dependent environmental/runtime variables, e.g. an object
    % encounter
    
    if simflag == 241   % no lesion
        if target_no == 6   % move object when target 6 is reached (corresponds to an experimental animal taken out of the env.)
            ObjPtX(1,:)  = 7-0.2:0.1:7+0.2;
            ObjCenX(1,1) = 7;
            ObjPtY(1,:)  = 14-0.2:0.1:14+0.2;
            ObjCenY(1,1) = 14;
        end
        if novelty_sig < 0                   % calculated below as correlation of population vectors              
            targetlist(7,:) = [7.5; 13.5];   % go to object that has moved
        end
        if target_no<6   % this is necessary because the agent passes through the same location twice
            imag_loc=0;  % imag_loc tells the code that there is specific location near which imagery should be enganged
        else
            imag_loc=1;  
        end
        if target_no == 2
            Vlin = 0;               % stop to encode objects
            ObjEncThresh = 7;
            if sum(ObjEncoded)==2   % once both objects encoded, resume movement
                Vlin = oldVlin;
            end
        end
        if imag_flag && step>imag_start+switchdur
            if step>=imag_start && step>=imag_start+imag_cuedur/4 && step<=imag_start+imag_cuedur/2 && isempty(OVCs_o1_imag)
                if max(OVC_rate)>0.6 && round(oPR_rate(1,1))==1   % we enter only once here becasue of the 'isempty' above
                    TMP_rate = OVC_rate;                          % recall 
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;     % theshold
                    OVCs_o1_imag = TMP_rate;                      % cue with only memorized objects. Could be any.
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o1_imag = TMP_rate;                       % for object 1
                end
            end
            if step>=imag_start+imag_cuedur*3/4 && step<=imag_start+imag_cuedur && isempty(OVCs_o2_imag)
                if max(OVC_rate)>0.6 && round(oPR_rate(2,1))==1   % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(PR_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o2_imag = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o2_imag = TMP_rate;                       % same as above for object 2
                end
            end
        end
        if any(OVCs_o1_imag) && any(OVCs_o2_imag) && Vlin == 0 && percep_flag
            Vlin = oldVlin;   % resume motion
        end
        if any(OVCs_o1_imag) && any(OVCs_o2_imag) && any(OVCs_o1_percep) && any(OVCs_o2_percep) && percep_flag % compute novelty signal
            corrtmp = corrcoef(OVCs_o1_percep,OVCs_o1_imag)-corrcoef(OVCs_o2_percep,OVCs_o2_imag);
            novelty_sig = corrtmp(1,2);
            corr1OVC = corrcoef(OVCs_o1_percep,OVCs_o1_imag);
            corr2OVC = corrcoef(OVCs_o2_percep,OVCs_o2_imag);
            OVCs_o1_imag   = [];
            OVCs_o2_imag   = [];
            OVCs_o1_percep = [];
            OVCs_o2_percep = [];
        end
    end
    
    if simflag == 242   % with lesion
        if imag_flag && step>imag_start+switchdur
            if step>=imag_start && step>=imag_start+imag_cuedur/4 && step<=imag_start+imag_cuedur/2 && isempty(OVCs_o1_imag)
                if max(OVC_rate)>0.6 && round(oPR_rate(1,1))==1
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o1_imag = TMP_rate;                       % cue with only memorized objects. Could be any.
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o1_imag = TMP_rate;
                end
            end
            if step>=imag_start+imag_cuedur*3/4 && step<=imag_start+imag_cuedur && isempty(OVCs_o2_imag)
                if max(OVC_rate)>0.6 && round(oPR_rate(2,1))==1   % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o2_imag = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o2_imag = TMP_rate;
                end
            end
        end
        if sum(ObjRecalled)==2 && Vlin == 0 && percep_flag
            ObjRecalled = ObjRecalled*0;
            Vlin = oldVlin;   % resume motion
        end
        if any(OVCs_o1_imag) && any(OVCs_o2_imag) && any(OVCs_o1_percep) && any(OVCs_o2_percep)      % compute novelty signal
            corrtmp = corrcoef(OVCs_o1_percep,OVCs_o1_imag)-corrcoef(OVCs_o2_percep,OVCs_o2_imag);   % do this at the two locations
            novelty_sig = corrtmp(1,2);
            if SandT == 1                                            % 1st loc.: incidental object-location match for one object
                corr1OVC = corrcoef(OVCs_o1_percep,OVCs_o1_imag);    % please see the article for what an object-location match means
                corr2OVC = corrcoef(OVCs_o2_percep,OVCs_o2_imag);
                corr1oPW = corrcoef(oPW_o1_percep,oPW_o1_imag);
                corr2oPW = corrcoef(oPW_o2_percep,oPW_o2_imag);
                OVCs_o1_imag   = [];
                OVCs_o2_imag   = [];
                OVCs_o1_percep = [];
                OVCs_o2_percep = [];
                oPW_o1_imag    = [];
                oPW_o2_imag    = [];
                oPW_o1_percep  = [];
                oPW_o2_percep  = [];
            end
            if SandT == 3                                            % 2nd loc.: incidental object-location match for other object
                corr3OVC = corrcoef(OVCs_o1_percep,OVCs_o1_imag);
                corr4OVC = corrcoef(OVCs_o2_percep,OVCs_o2_imag);
                corr3oPW = corrcoef(oPW_o1_percep,oPW_o1_imag);
                corr4oPW = corrcoef(oPW_o2_percep,oPW_o2_imag);
                OVCs_o1_imag   = [];
                OVCs_o2_imag   = [];
                OVCs_o1_percep = [];
                OVCs_o2_percep = [];
                oPW_o1_imag    = [];
                oPW_o2_imag    = [];
                oPW_o1_percep  = [];
                oPW_o2_percep  = [];
            end
        end
        if target_no==3 && SandT~=3
            imag_start = 140000; % needs resetting to default values, to trigger 2nd imagination phase
            imag_loc = 1;
            SandT    = 3;
        end
    end
        
    
    
    if imag_flag && sum(ObjRecalled)<Nobj   % track if we have recalled an object, main use for simflags 241/242
        for q = 1:Nobj
            if step>=imag_start+((q-1)*Nobj+1)*imag_cuedur/(2*Nobj) && step<=imag_start+q*imag_cuedur/Nobj
                if max(OVC_rate)>0.5 && round(oPR_rate(q,1))==1
                    ObjRecalled(q,1) = 1;
                end
            end
            
        end
    end
    
    
    
    % preplay specific stuff
    if simflag == 50   % do imagined movement through new part of enviroment once at targetloc 2    
        
        if target_no==2 && nobarrier==0
            imag_cuedur_nav = 10000;            % will be cut sort once we reach object 2
            imag_start_nav  = step+1;
            pseudoadap      = 1;
        end
        
        if target_no==5 && imag2 == 0           % second imagination phase
            imag_cuedur_nav           = 10000;  % will be cut sort once we reach object 2
            imag_start_nav            = step+1;
            pseudoadap                = 1;
            target_no_imag_nav        = 0;
            DWELL_imag_nav            = 150;
            dwell_imag_nav            = DWELL;
            move2nextTarget_imag_nav  = 0;
            locateNextTarget_imag_nav = 0;
            rotResolved_imag_nav      = 1;
            imag2                     = 1;
            Xag_imag_nav              = 16;
            Yag_imag_nav              = 6;
            imag_end_flag             = 0;
            dwell                     = 150;
            HDag                      = first_imag_HD;
        end
        
        if target_no_imag_nav==1 && Yag_imag_nav>17 && imag_end_flag==0  % reached imagined target
            imag_cuedur_nav = step - imag_start_nav;                     % this should stop imagined movement
            imag_end_flag   = 1;
        end
        
    end

    
    
    % deternine whether we are in bottom-up (perception driven) or top-down
    % (imagery) mode according to sim parameters. this includes mental navigation
    % cueing with a specific object is also set in this subroutine
            [oPRCue,HDCue,oHDCue,HDag,old_HD,percep_flag,navi_flag,navi_flag_imag,imag_flag,recallobj,plt_ttl_flag,imag2] = ...
    BB_subr_BUvsTD(step,imag_start,imag_start_nav,imag_cuedur,imag_cuedur_nav,switchdur,HDag,old_HD,first_imag_HD, ...
            percep_flag,navi_flag,navi_flag_imag,imag_flag,recallobj,plt_ttl_flag,imag2,oPRCue,HDCue,oHDCue,simflag,0,oPRCuescale);
    
    
    
    if imag1 && imag_loc==1 && SandT>0   % must come after BUvTD
        
        Xst = targetlist(SandT,1);
        Yst = targetlist(SandT,2);
        
        if simflag == 241 % this is independent of the other recordings for correlation analysis, 
                          % because we must compare the moved object rep, not the one at encoding
                          % meaning above e.g. oPW_o1_imag is recorded. Here we record oPW_o1_percep
            if sqrt((Xst-Xag).^2 + (Yst-Yag).^2)<0.02 && sum(ObjRecalled)<Nobj
                Vlin = 0;
                if max(OVC_rate)>0.6 && round(oPR_rate(1,1))==1 && mod(step,ATTcycle)>ATTcycle/(2*Nobj) && percep_flag && isempty(OVCs_o1_percep) % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o1_percep = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o1_percep = TMP_rate;
                end
                if max(OVC_rate)>0.6 && round(oPR_rate(2,1))==1 && mod(step,ATTcycle)>ATTcycle/(2*Nobj) && percep_flag && isempty(OVCs_o2_percep)   % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o2_percep = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o2_percep = TMP_rate;
                end
                if imag_flag == 0 && imag_start == 140000
                    imag_start = step+1.5*ATTcycle;             % override imag_start as a temporal definition if we want.
                    imag_loc = 0;                               % specific object to recall set in BB_subr_BUvsTD
                end
            end
        end
        
        if simflag == 242
            if sqrt((Xst-Xag).^2 + (Yst-Yag).^2)<0.02 && sum(ObjRecalled)<Nobj
                Vlin = 0;
                if max(OVC_rate)>0.6 && round(oPR_rate(1,1))==1 && mod(step,ATTcycle)>ATTcycle/(2*Nobj) && percep_flag && isempty(OVCs_o1_percep) % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o1_percep = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o1_percep = TMP_rate;
                end
                if max(OVC_rate)>0.6 && round(oPR_rate(2,1))==1 && mod(step,ATTcycle)>ATTcycle/(2*Nobj) && percep_flag && isempty(OVCs_o2_percep)   % reference for novelty detection
                    TMP_rate = OVC_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    OVCs_o2_percep = TMP_rate;
                    TMP_rate = oPW_rate;
                    TMP_rate(TMP_rate<max(TMP_rate)*0.1) = 0;
                    oPW_o2_percep = TMP_rate;
                end
                if imag_flag == 0 && imag_start == 140000
                    imag_start = step+1.5*ATTcycle;              % override imag_start as a temporal definition if we want.
                end
                if ~isempty(OVCs_o1_percep) && ~isempty(OVCs_o2_percep)
                    imag_loc = 0;
                end
            end
        end
        
        if simflag ~= 241 && simflag ~= 242
            if sqrt((Xst-Xag).^2 + (Yst-Yag).^2)<0.01
                if imag_flag == 0
                    imag_start = step+1;                        % override imag_start as a temporal definition if we want.
                    imag_loc = 0;                               % specific object to recall set in BB_subr_BUvsTD
                end
            end
        end
        
    end
    
    
    
    if step == imag_start
        ObjCue_percep = ObjCue_percep*0;  % reset perceptual cues to zero once we start imagining, similar to below
    end
    
    
    
    if percep_flag == 1    % perceptual drive, we are in bottom-up mode, 
                           % determine ongoing ego cues for k_PW, k_PWo
        if imag2~=1        % quick and dirty fix for video
                    [egocues,VBX_rt,VBY_rt,L_r,BX,BY,TX,TY] = ...
            BB_subr_PdrivePW_withObj(r0,dir,line,BndryPtX,BndryPtY, ...
                    Xag,Yag,-HDag,step,modstep,0);   % cue with entire field of view, egocue gets transformed one line below and ... 
        end
        
        EgoCue_percep = 30 * BB_subr_cue_PWact_multiseg(0.5,egocues);  % ... becomes EgoCue_percep, an input to PW neurons, see below
        
        if step > 300 && bleed<=1   % attention and encoding, see subroutine for details, updates weights and ObjCueing current
                    [ObjCue_percep,oPR_drive,ObjEncoded,H2OVCwts,OVC2Hwts,OVC2BVCwts,BVC2OVCwts, ...
                    oPR2OVCwts,OVC2oPRwts,oPR2HDwts,oPR2Hwts,H2oPRwts,Xag_encode,Yag_encode,CurrObjCenX,CurrObjCenY] = ...
            BB_subr_attention_enc( ...
                    ObjCenX,Xag,ObjCenY,Yag,ObjAttThresh,ATTcycle,r0,dir,line,HDag,step,ObjCue_percep, ...
                    ObjEncoded,ObjEncThresh,HD_rate,OVC_rate,H_rate,NHx,NHy,BVC_rate,oPR_rate,H2OVCwts, ...
                    OVC2Hwts,OVC2BVCwts,BVC2OVCwts,oPR2OVCwts,OVC2oPRwts,oPR2Hwts,H2oPRwts,oPR2HDwts, ...
                    N_oPR,N_BVC,OVC2oPR_inhib_phi,simflag);
        end
        
        % with engaged perception buondary cueing (i.e. 'recognizing boundaries you look at') 
        % also goes on. I.e. we determine the identity of the visible boundaries, this is
        % not modelled at a mechanistic level. Reconition, supposedly mediated by the ventral 
        % visual stream (with perirhinal cortex near the apex), is implemented heuristically
        if simflag == 50    % separate because boundary config is different
            [PRcue_percep] = BB_subr_PR_BoundaryCueing_with_Barrier_sim50(VBX_rt,VBY_rt,HDag,Xag,Yag,Ybarrier_bot,Ybarrier_top,Xbarrier_top_min,Xbarrier_top_max,nobarrier);
        else
            [PRcue_percep] = BB_subr_PR_BoundaryCueing(VBX_rt,VBY_rt,HDag,Xag,Yag);
            if simflag == 21 || simflag == 26
                PRcue_percep = [PRcue_percep; 0; 0];  % Expecting 6dim input but the barrier from training is removed
            end
        end
        
    else
        EgoCue_percep = 0;   % if we are not perceiving ...
        PRcue_percep  = 0;
    end
    
    
    
    if step>=imag_start+imag_cuedur && step<imag_start+imag_cuedur+switchdur
        oPR_rate    = oPR_rate*0;   
    end
    
    
    
    % The agent model, i.e. this script basically calculates where the
    % little black arrow head in the allocentric agent plot should be. Xag,
    % Yag, are the agents position. X/Yag_imag_nav the same for imagined
    % movement. The othter variable names are hopefully self-seplanatory.
    % E.g. "toturn" refers to the amount left to turn etc. See inside
    % subrouting for details.
    
           [Xag            ,Yag                ,ang2f_tot                 ,toturn                   ,dist2f_cur           ,dwell          , ...
            Xag_imag_nav   ,Yag_imag_nav       ,ang2f_tot_imag_nav        ,toturn_imag_nav          ,dist2f_cur_imag_nav  ,dwell_imag_nav , ...
            navi_flag      ,target_no          ,locateNextTarget          ,move2nextTarget          ,rotResolved          , ...
            navi_flag_imag ,target_no_imag_nav ,locateNextTarget_imag_nav ,move2nextTarget_imag_nav ,rotResolved_imag_nav ,CWturn ,CCWturn ] = ...
    BB_subr_agent_model(...
            step, Vlin ,HDag ,HDestim_prev ,HDestim ,imag_cuedur_nav ,percep_flag ,imag_flag ,dt ,CWturn ,CCWturn , H_rate, NHx, NHy, ...
            Xag            ,Yag                       ,ang2f_tot                ,toturn               ,dist2f_cur          ,targetlist          ,target_no         , dwell      , ...
            Xag_imag_nav   ,Yag_imag_nav              ,ang2f_tot_imag_nav       ,toturn_imag_nav      ,dist2f_cur_imag_nav ,targetlist_imag_nav ,target_no_imag_nav, dwell_imag_nav, ...
            DWELL          ,locateNextTarget          ,move2nextTarget          ,rotResolved          ,navi_flag , ...
            DWELL_imag_nav ,locateNextTarget_imag_nav ,move2nextTarget_imag_nav ,rotResolved_imag_nav ,navi_flag_imag ,encoding, imag_start_nav, switchdur);
    

    
    if imag_flag == 1 && simflag==50 && isnan(first_imag_HD)   % becausse we have 2 mental nav episodes during sim. 50
        first_imag_HD = HDag;
    end
    
    
    
    if percep_flag_prev ~= percep_flag   % track statechanges 
        statechanges = statechanges + 1;
    end
    
    
    
    if simflag==50 && statechanges==2   % update environment to remove barriers of shortcut
        if nobarrier==0
            eval(['load ' WTSdir '/Roomdataforperceptiondrive_preplay_env_open_EXT_' WTSname '']);
            nobarrier = 1;
        end
    end
    
    
    
    % Now Calculate all k terms
    % E.G.:   if    dX/dt = k   then with     dX/dt = (X_new - X_old)/dt = k     
    % it follows that     X_new = X_old + dt*k
    % here X_new is the variable at time t+1 and X_old at time t
    % once all k terms are calculated we can use them to update the
    % activation of the neurons which is then passed through the transfer
    % function
    
    % k-terms usually consist of a decay term given by the previous
    % activity plus inputs, which take the following form: phi (scalar
    % scaling factor) * Weights * Rate of afferent neurons
    
    
    
    % HEAD DIRECTION CELLS
    k_HD = -HDact + HDphi*HD2HDwts * HD_rate + (step<CueDur) * HDCue_init + percep_flag * HDCue + imag_flag * oHDCue ...   
        + Rotphi * CCWturn * Rotwts  * HD_rate ...                                                                               
        + Rotphi * CWturn  * Rotwts' * HD_rate ...
        + imag_flag * 60 * oPR2HDwts * oPR_rate * oPRHDchange;   
    
    
    
    % GRID CELLS
    if imag_flag == 1 && percep_flag == 0           % GCs have to distinguish mental nav from preplay
        if Xag_imag_nav>=0                          % covers mental navigation and preplay
            GC_rate  = squeeze(GCs(max(1,round(Yag_imag_nav*2)),max(1,round(Xag_imag_nav*2)),:));
            AllImagX = [AllImagX, Xag_imag_nav];    % keep track of imagined trajectory for plotting
            AllImagY = [AllImagY, Yag_imag_nav];
        else                                        % this else may be unnecessary
            GC_rate = TDBU_ScaleGC * squeeze(GCs(max(1,round(Yag*2)),max(1,round(Xag*2)),:));
        end
    end
    if imag_flag == 0 && percep_flag == 1   % during perception
        GC_rate = squeeze(GCs(max(1,round(Yag*2)),max(1,round(Xag*2)),:));
    end

    
    
    if simflag==50 % record H activity for preplay plots
        
        resH = sum(H_rate(Rlininds));                            % firing of reservoir PCs
        envH = sum(H_rate(ENVlininds));                          % firing of PCs across known environment
        
        % phase 1: 'preplay'-like activity
        if statechanges==1 && Xag_imag_nav>=12.5 && Yag_imag_nav>minYb/2 && Yag_imag_nav<maxYb/2
            if resH > envH
                Xi = round(Xag_imag_nav*2);                      % these function as indices for the Hmem array
                Yi = round(Yag_imag_nav*2);
                if (Xi~=Xi_prev || Yi~=Yi_prev)
                    Xi_prev  = Xi;
                    Yi_prev  = Yi;
                    Hmem1    = [Hmem1 H_rate(Rlininds)];
                    Hm1Count = Hm1Count + 1;
                    Hm1Cmean = [Hm1Cmean 1];
                else
                    Hmem1(:,Hm1Count-1) = Hmem1(:,Hm1Count-1) + H_rate(Rlininds);
                    Hm1Cmean(end) = Hm1Cmean(end) + 1;
                end
            end
        end
        if statechanges==1 && Xag_imag_nav>=12.5 && Yag_imag_nav>minYb/2 && Yag_imag_nav>=maxYb/2 && Cflag==0
            H1count = repmat(Hm1Cmean,length(Hmem1(:,1)),1);
            Hmem1 = Hmem1./H1count;
            Cflag = 1;
        end
        
        % phase 2: running across the shortcut
        if statechanges==2 && Xag>=12.5 && Yag>minYb/2 && Yag<maxYb/2
            if resH > envH
                Xi = round(Xag*2);                                % these function as indices for the Hmem array
                Yi = round(Yag*2);
                if (Xi~=Xi_prev || Yi~=Yi_prev)
                    Xi_prev              = Xi;
                    Yi_prev              = Yi;
                    Hmem2    = [Hmem2 H_rate(Rlininds)];
                    Hm2Count = Hm2Count + 1;
                    Hm2Cmean = [Hm2Cmean 1];
                else
                    Hmem2(:,Hm2Count-1) = Hmem2(:,Hm2Count-1) + H_rate(Rlininds);
                    Hm2Cmean(end) = Hm2Cmean(end) + 1;
                end
                
            end
        end
        if statechanges==1 && Xag>=12.5 && Yag>minYb/2 && Yag>=maxYb/2 && Cflag==1
            H2count = repmat(Hm2Cmean,length(Hmem2(:,1)),1);
            Hmem2 = Hmem2./H2count;
            Cflag = 2;
        end
        
        % phase 3: mental navigation across shortcut with updated weights
        % BVCs and reservoir PCs have been associated
        if statechanges==3 && Xag_imag_nav>=12.5 && Yag_imag_nav>minYb/2 && Yag_imag_nav<maxYb/2
            if resH > envH
                Xi = round(Xag_imag_nav*2);                       % these function as indices for the Hmem array
                Yi = round(Yag_imag_nav*2);
                if (Xi~=Xi_prev || Yi~=Yi_prev)
                    Xi_prev  = Xi;
                    Yi_prev  = Yi;
                    Hmem3    = [Hmem3 H_rate(Rlininds)];
                    Hm3Count = Hm3Count + 1;
                    Hm3Cmean = [Hm3Cmean 1];
                else
                    Hmem3(:,Hm3Count-1) = Hmem3(:,Hm3Count-1) + H_rate(Rlininds);
                    Hm3Cmean(end) = Hm3Cmean(end) + 1;
                end
            end
        end
        if statechanges==3 && Xag_imag_nav>=12.5 && Yag_imag_nav>minYb/2 && Yag_imag_nav>=maxYb/2 && Cflag==2
            H3count = repmat(Hm3Cmean,length(Hmem3(:,1)),1);
            Hmem3 = Hmem3./H3count;
            Cflag = 3;
        end
        
    end
    
    
    
    % very simple Hebbian learning between BVCs and PCs (and GCs and PCs) during shortcut traveral
    % for a more universal setup probably need to include some normalization etc. 
    if simflag==50 && statechanges==2 && Xag>=12.5 && Yag>minYb/2 && Yag<maxYb/2
        if resH > envH 
            
            dW1      = lrate_GC2H * H_rate * GC_rate';     dW1(dW1<lrate_GC2H/100) = 0;
            GC2PCwts = GC2PCwts + dW1;   
           
            dW3      = lrate_H2BVC * BVC_rate * H_rate';   dW3(dW3<lrate_H2BVC/100) = 0;
            H2BVCwts = H2BVCwts + dW3;  
            
            dW3b     = lrate_BVC2H * H_rate * BVC_rate';   dW3b(dW3b<lrate_BVC2H/100) = 0;
            BVC2Hwts = BVC2Hwts + dW3b; 
            
        end
    end
    
    
    
    % make bottom-up/top-down changes more smooth
    if percep_flag == 1 && imag_flag == 0
        Pmod     = Pmod+0.01;        % increase Pmod until it reaches 1 over 100 ms
        Pmod     = min(Pmod,1);      % up-regulate during perception
        Imod     = Imod-0.01;
        Imod     = max(Imod,TDBU_Scale);
        GC2PCphi = TDBU_ScaleGC*GC2PCphi_base;   
        % note, down-regulting GCs during perception is not really necessary (they would 
        % simply make PC firing more stable), but doing so shows that BVC inputs to PCs 
        % (derived from sensory input in the PW) can be sufficient to self-localize
        if simflag==50 && statechanges==2 && percep_flag == 1
            GC2PCphi = GC2PCphi_base;
            BVC2Hphi = 0;    % avoid interference in novel part of the environment
            PR2Hphi  = 0;
        else
            BVC2Hphi = BVC2Hphi_b;
            PR2Hphi  = 25;   % same as initially
        end
    end
    if imag_flag == 1 && percep_flag == 0
        Pmod     = Pmod-0.01;
        Pmod     = max(Pmod,TDBU_Scale);   % down-regulate during imagery
        Imod     = Imod+0.01;
        Imod     = min(Imod,1);
        GC2PCphi = GC2PCphi_base;  
    end
    
    
    
    % PLACE CELLS
    k_H = -Hact + Hphi * H2Hwts * H_rate + ...
        Pmod * BVC2Hphi * BVC2Hwts * BVC_rate + ...        
        PR2Hphi * PR2Hwts * PR_rate + ...                  
        OVC2Hphi * OVC2Hwts * OVC_rate +  ...
        Imod * oPR2Hphi *  oPR2Hwts * oPR_rate + ...          
        GC2PCphi * GC2PCwts * GC_rate + I_comp;    
    if simflag == 242
        k_H = 0;   % lesion
    end
    
    
    
    % Theta modulation for rodent-like theta, see paper for rational, bleed parameter in Appendix
    if simflag == 26 || simflag == 27 || simflag == 28
        if Pmod == 1      % only do this in full perceptual mode because the transient 
            bcap = 1.0;   % rise and fall of Pmod and Imod messes things up otherwise
            bdur = 150;
            if mod(step,bdur) < (bdur/3)
                bleed = 1;  
                TDBU_ScaleGC = 0.1;
            end
            if mod(step,bdur) >= (bdur/3)
                bleed = bleed + 0.2; 
                TDBU_ScaleGC = min(bleed * Imod,1);
            end
        end
    else
        if Pmod == 1   
            bcap = 1.0; 
            bdur = 1000;
            if (simflag>20 && simflag<30) && target_no==4 && bleed>=1 && bleed<bcap/Imod/2 && percep_flag==1 && rotResolved==1
                bleed = bleed + 0.01;
                Vlin  = max(0,Vlin-(0.01*Vlin)); % slow down
                perbl = 1.0;
            end
            if (simflag>20 && simflag<30) && bleed>=bcap/Imod/2 && percep_flag==1 && Bcount<=bdur
                Vlin   = 0;   % stop
                Bcount = Bcount+1;
                perbl  = 1.0;
            end
            if (simflag>20 && simflag<30) && Bcount>bdur
                Vlin  = oldVlin;  % resume
                bleed = 0.9999;
                perbl = 1;
            end
        end
    end
    
    
    
    % Boundary Vector Cells
    k_BVC = -BVCact + BVC2BVCwts * BVC_rate + ...           % BVC2BVCwts are currently zero, same for OVC to BVC and vice verse
        bleed * Imod * H2BVCphi * H2BVCwts * H_rate + ...   
        OVC2BVCphi * OVC2BVCwts * OVC_rate + ...
        PR2BVCphi * PR2BVCwts * PR_rate + ...
        1/bleed * Pmod * TR2BVCphi * TR2BVCwts * ( TR_rate1 + ...
        TR_rate2 + TR_rate3 + TR_rate4 + TR_rate5 + TR_rate6 + TR_rate7 + ...
        TR_rate8 + TR_rate9 + TR_rate10 + TR_rate11 + TR_rate12 + TR_rate13 + ...
        TR_rate14 + TR_rate15 + TR_rate16 + TR_rate17 + TR_rate18 + TR_rate19 + TR_rate20 ); 
    
    
    
    % Perirhinal Identity Neurons (boundary specific)
    k_PR = -PRact + PR2PRwts * PR_rate + ...
        Imod * H2PRphi * H2PRwts * H_rate + ...
        BVC2PRphi * BVC2PRwts * BVC_rate + (CueDur>step) * PRCue + PRcue_percep;   % PRCue will always be zero except at start
    
    
    
    
    if simflag == 31 && step>imag_start+imag_cuedur/2 && step<=imag_start+imag_cuedur
        Tdecay = step-(imag_start+imag_cuedur/2);  % add some adaptation to allow the other PR neuron to win
        pseudoadap = [1/Tdecay; 1];
    else
        pseudoadap = 1;
    end
    
    

    % Perirhinal Identity Neurons (object specific)
    k_oPR = -oPRact + (oPR2oPRphi * oPR2oPRwts * oPR_rate + ...          
        H2oPRphi * H2oPRwts * H_rate + ...
        OVC2oPRphi * OVC2oPRwts * OVC_rate + ...                    
        recallobj * oPRCue ).* pseudoadap + percep_flag*200*oPR_drive;   % added actual PR drive, drive 0 during imagery
    
    
    
    % Object Vector Cells
    k_OVC = -OVCact + OVC2OVCphi * OVC2OVCwts * OVC_rate + ...            % OVC to OVC is purely inhibitory
        bleed * Imod * H2OVCphi * H2OVCwts * H_rate + ...
        BVC2OVCphi * BVC2OVCwts * BVC_rate + ...
        bleed * Imod * oPR2OVCphi * oPR2OVCwts * oPR_rate + ...           
        1/bleed * Pmod * TR2OVCphi * TR2BVCwts * ( oTR_rate1 + ...
        oTR_rate2 + oTR_rate3 + oTR_rate4 + oTR_rate5 + oTR_rate6 + oTR_rate7 + ...
        oTR_rate8 + oTR_rate9 + oTR_rate10 + oTR_rate11 + oTR_rate12 + oTR_rate13 + ...
        oTR_rate14 + oTR_rate15 + oTR_rate16 + oTR_rate17 + oTR_rate18 + oTR_rate19 + oTR_rate20 );
    
    
    
    % Allocate attention to residual oPW peak in imagery, infering identity
    % of second object which is also part of the imagined scence.implemented heuristically
    if simflag == 31 && step>imag_start+imag_cuedur/2 && step<=imag_start+imag_cuedur    
        % after half the time, cue a different receptive field. 
        % for this must have objects encoded at same location
        if ObjCue_allocATT==0
                    [ObjCue_allocATT] = ...
            BB_subr_allocATTinPW(r0,dir,line,step,imag_start,imag_cuedur,H_rate,NHx,NHy,oPW_rate, ...
                    NBVCR,NBVCTheta,polarDistRes,polarAngRes,HDag,CurrObjCenX,CurrObjCenY,polarDist);
            oPRCue   = [0; 0];
        end
        plt_ttl_flag = 77;
    else
        ObjCue_allocATT = 0;
    end
    
    
    
    % Parietal Window Cells (object specific)
    k_oPW = -oPWact - sum(oPW_rate) * oPW_bath + ...
        bleed * Imod * TR2oPWphi * (TR2PWwts1 * oTR_rate1 + ...                       
        TR2PWwts2*oTR_rate2 + TR2PWwts3*oTR_rate3 + TR2PWwts4*oTR_rate4 + TR2PWwts5*oTR_rate5 + ...
        TR2PWwts6*oTR_rate6 + TR2PWwts7*oTR_rate7 + TR2PWwts8*oTR_rate8 + TR2PWwts9*oTR_rate9 + ...
        TR2PWwts10*oTR_rate10 + TR2PWwts11*oTR_rate11 + TR2PWwts12*oTR_rate12 + TR2PWwts13*oTR_rate13 + ...
        TR2PWwts14*oTR_rate14 + TR2PWwts15*oTR_rate15 + TR2PWwts16*oTR_rate16 + TR2PWwts17*oTR_rate17 + ...
        TR2PWwts18*oTR_rate18 + TR2PWwts19*oTR_rate19 + TR2PWwts20*oTR_rate20) + 1/(min(bleed,2)) ... 
        * OcueScale * ObjCue_percep + 1 * ObjCue_allocATT;
    
    

    % Parietal Window Cells (boundary specific)
    k_PW = -PWact - 100*PW_bath + ...
        bleed * Imod * TR2PWphi * (TR2PWwts1 * TR_rate1 + ...     
        TR2PWwts2*TR_rate2 + TR2PWwts3*TR_rate3 + TR2PWwts4*TR_rate4 + TR2PWwts5*TR_rate5 + ...
        TR2PWwts6*TR_rate6 + TR2PWwts7*TR_rate7 + TR2PWwts8*TR_rate8 + TR2PWwts9*TR_rate9 + ...
        TR2PWwts10*TR_rate10 + TR2PWwts11*TR_rate11 + TR2PWwts12*TR_rate12 + TR2PWwts13*TR_rate13 + ...
        TR2PWwts14*TR_rate14 + TR2PWwts15*TR_rate15 + TR2PWwts16*TR_rate16 + TR2PWwts17*TR_rate17 + ...
        TR2PWwts18*TR_rate18 + TR2PWwts19*TR_rate19 + TR2PWwts20*TR_rate20) + 1/(min(bleed,2)) ...
        * BcueScale * EgoCue_percep;
    
    
    
    % Transformation Circuit(s)
    BVC2TR = BVC2TRwts * BVC_rate * bleed;                            % Only want to do this once
    IPrate = 1./(1+exp(-2*IPbeta*(HD2IPphi*sum(HD_rate)-IPalpha)));   % inhibition
    if simflag==11 || simflag == 12 || simflag == 111
        HD_rate = HD_rate*0;  % HD/Papez' Circuit lesion
    end
    for i = 1:20
        eval(['k_TR' int2str(i) '  = -TRact' int2str(i) '  - sum(TR_rate' int2str(i) ')  * TR_bath + HD2TRphi*HD2TRwts(:,:,'...
            int2str(i) ')*HD_rate  - IP2TRphi*IPrate + Imod*BVC2TRphi*BVC2TR + 1/bleed * Pmod*PW2TRphi*PW2TRwts' int2str(i) '*PW_rate;']);
    end
    OVC2TR  = BVC2TRwts * OVC_rate * bleed;                            % Only want to do this once
    for i = 1:20
        eval(['k_oTR' int2str(i) '  = -oTRact' int2str(i) '  - sum(oTR_rate' int2str(i) ')  * oTR_bath + HD2TRphi*HD2TRwts(:,:,' ...
            int2str(i) ')*HD_rate  - IP2oTRphi*IPrate + Imod*BVC2TRphi*OVC2TR + 1/bleed * Pmod*oPW2TRphi*PW2TRwts' int2str(i) '*oPW_rate;']);
    end
    % eval is slow but compared to getframe (used below for video) it does
    % not matter, also makes the code more readable (no repetition of the
    % same line 40 times).
    
    
    
    % plotting and recording video
    if mod(step,modstep) == 0 && plotting   % plot and getframe every modestep steps  
        if simflag > 0
            if simflag == 60   % special plotting for illustration of transformation circuit
                [frame] = ....
                    BB_subr_PLT_TR(step, modstep, plt_ttl_flag, percep_flag, simflag, ...
                    H_rate, NHx, NHy, HX, HY, GC2PCwts, GC_rate*GC2PCphi, ...
                    BVCX, BVCY, BVC_rate, NBVCR, NBVCTheta, HDX, HDY, HD_rate, ...
                    PW_rate, OVC_rate, oPW_rate, PR_rate, nobarrier, oPR_rate, ...
                    ObjEncoded, TX, TY, BX, BY, ObjCenX, ObjCenY, old_HD, HDag, ...
                    Xag, Yag, imag_flag, Xag_imag_nav, Yag_imag_nav, oldCoordsTMP, ...
                    AllImagX, AllImagY, VBX_rt, VBY_rt, L_r, ...
                    0, trackI_c, ...
                    Hmem1, Hmem2, Hmem3, oPRCue, ...
                    TR_rate1, TR_rate2, TR_rate3, TR_rate4, TR_rate5, TR_rate6, TR_rate7, ...
                    TR_rate8, TR_rate9, TR_rate10, TR_rate11, TR_rate12, TR_rate13, TR_rate14, ...
                    TR_rate15, TR_rate16, TR_rate17, TR_rate18, TR_rate19, TR_rate20, ...
                    oTR_rate1, oTR_rate2, oTR_rate3, oTR_rate4, oTR_rate5, oTR_rate6, oTR_rate7, ...
                    oTR_rate8, oTR_rate9, oTR_rate10, oTR_rate11, oTR_rate12, oTR_rate13, ...
                    oTR_rate14, oTR_rate15, oTR_rate16, oTR_rate17, oTR_rate18, oTR_rate19, oTR_rate20);
            else
                [frame] = ....
                    BB_subr_PLT_fancy(step, modstep, plt_ttl_flag, percep_flag, simflag, ...
                    H_rate, NHx, NHy, HX, HY, GC2PCwts, GC_rate*GC2PCphi, ...
                    BVCX, BVCY, BVC_rate, NBVCR, NBVCTheta, HDX, HDY, HD_rate, ...
                    PW_rate, OVC_rate, oPW_rate, PR_rate, nobarrier, oPR_rate, ...
                    ObjEncoded, TX, TY, BX, BY, ObjCenX, ObjCenY, old_HD, HDag, ...
                    Xag, Yag, imag_flag, navi_flag_imag, Xag_imag_nav, Yag_imag_nav, oldCoordsTMP, ...
                    AllImagX, AllImagY, VBX_rt, VBY_rt, L_r, ...
                    0, trackI_c, ...
                    Hmem1, Hmem2, Hmem3, oPRCue, BndryPtX_insB, BndryPtY_insB,CoordsTMPmem);
            end
        end
        if makemovie && simflag>0 && step>100
            writeVideo(writerObj,frame);
        end
    end
    
    
    
    % finally update neuron activities and pass through non-linearity to get rates
    Hact     = Hact     + dt/Htau*k_H;
    BVCact   = BVCact   + dt/BVCtau*k_BVC;
    PRact    = PRact    + dt/PRtau*k_PR;
    oPRact   = oPRact   + dt/PRtau*k_oPR;
    HDact    = HDact    + dt/HDtau*k_HD;
    PWact    = PWact    + dt/PWtau*k_PW;
    OVCact   = OVCact   + dt/OVCtau*k_OVC;
    oPWact   = oPWact   + dt/oPWtau*k_oPW;
    for i = 1:20
        eval(['TRact' int2str(i) ' = TRact' int2str(i) ' + dt/TRtau*k_TR' int2str(i) ';']);
        eval(['oTRact' int2str(i) ' = oTRact' int2str(i) ' + dt/TRtau*k_oTR' int2str(i) ';']);
    end
    
    I_comp   = I_comp + dt/ICtau*(15-sum(H_rate));  % Diff. eq. for compensatory H current (keeping the amounf to active cells approx. constant)
    
    % apply sigmoid transfer fct to get rates
    H_rate     = 1./(1+exp(-2*Hbeta*(Hact-Halpha)));   if simflag==242 H_rate=H_rate*0;  end
    BVC_rate   = 1./(1+exp(-2*BVCbeta*(BVCact-BVCalpha)));
    PR_rate    = 1./(1+exp(-2*PRbeta*(PRact-PRalpha)));
    oPR_rate   = 1./(1+exp(-2*PRbeta*(oPRact-PRalpha)));
    HD_rate    = 1./(1+exp(-2*HDbeta*(HDact-HDalpha)));
    PW_rate    = 1./(1+exp(-2*PWbeta*(PWact-PWalpha)));
    OVC_rate   = 1./(1+exp(-2*OVCbeta*(OVCact-OVCalpha)));
    oPW_rate   = 1./(1+exp(-2*oPWbeta*(oPWact-oPWalpha)));
    for i = 1:20
        eval(['TR_rate' int2str(i) ' = 1./(1+exp(-2*TRbeta*(TRact' int2str(i) '-TRalpha)));']);
        eval(['oTR_rate' int2str(i) '  = 1./(1+exp(-2*TRbeta*(oTRact' int2str(i) '-TRalpha)));']);
    end
    
    
    
    % implement FR noise
    if noise_flag1
        H_pm        = [ones(N_H/2,1); -ones(N_H/2,1)];
        H_pm        = H_pm(randperm(N_H));
        H_FRnoise   = ones(N_H,1)+(0.05*randn(N_H,1)+FRNoiseLevel/100.*H_pm);
        BVC_pm      = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];
        BVC_pm      = BVC_pm(randperm(N_BVC));
        BVC_FRnoise = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*BVC_pm);
        OVC_pm      = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];
        OVC_pm      = OVC_pm(randperm(N_BVC));
        OVC_FRnoise = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*OVC_pm);
        PW_pm       = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];
        PW_pm       = PW_pm(randperm(N_BVC));
        PW_FRnoise  = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*PW_pm);
        oPW_pm      = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];
        oPW_pm      = oPW_pm(randperm(N_BVC));
        oPW_FRnoise = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*oPW_pm);    
        for i = 1:20
            eval(['TR_pm' int2str(i) ' = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];']);
            eval(['TR_pm' int2str(i) ' = TR_pm' int2str(i) '(randperm(N_BVC));']);
            eval(['TR_FRnoise' int2str(i) ' = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*TR_pm' int2str(i) ');']);
            eval(['oTR_pm' int2str(i) ' = [ones(N_BVC/2,1); -ones(N_BVC/2,1)];']);
            eval(['oTR_pm' int2str(i) ' = oTR_pm' int2str(i) '(randperm(N_BVC));']);
            eval(['oTR_FRnoise' int2str(i) ' = ones(N_BVC,1)+(0.05*randn(N_BVC,1)+FRNoiseLevel/100.*oTR_pm' int2str(i) ');']);
        end
        H_rate   = max(0,min(H_rate.*H_FRnoise,1));
        %HD_rate  = max(0,min(HD_rate.*HD_FRnoise,1));   % HD ring too brittle at only 100 neurons
        BVC_rate = max(0,min(BVC_rate.*BVC_FRnoise,1));
        OVC_rate = max(0,min(OVC_rate.*OVC_FRnoise,1));
        PW_rate  = max(0,min(PW_rate.*PW_FRnoise,1));
        oPW_rate = max(0,min(oPW_rate.*oPW_FRnoise,1));
        PW_rate  = max(0,min(PW_rate.*PW_FRnoise,1));
        for i = 1:20
            eval(['TR_rate' int2str(i) ' = max(0,min(TR_rate' int2str(i) '.*TR_FRnoise' int2str(i) ',1));']);
            eval(['oTR_rate' int2str(i) ' = max(0,min(oTR_rate' int2str(i) '.*oTR_FRnoise' int2str(i) ',1));']);
        end

    end
    % implement neuron loss
    if noise_flag2
        H_rate   = H_rate.*H_indArr;
        %HD_rate  = HD_rate.*HD_indArr;
        BVC_rate = BVC_rate.*BVC_indArr;
        OVC_rate = OVC_rate.*OVC_indArr;
        oPW_rate = oPW_rate.*oPW_indArr;
        PW_rate  = PW_rate.*PW_indArr;
        for i = 1:20
            eval(['TR_rate' int2str(i) ' = TR_rate' int2str(i) '.*TR_indArr' int2str(i) ';']);
            eval(['oTR_rate' int2str(i) ' = oTR_rate' int2str(i) '.*oTR_indArr' int2str(i) ';']);
        end
    end
    
    
    
    % record rates for correlation based quantification, recall v perception at encoding
           [PRcorr_IM, oPRcorr_IM, OVCcorr_IM, BVCcorr_IM, PCcorr_IM, PWcorr_IM, oPWcorr_IM, ...
            PRcorr_PE, oPRcorr_PE, OVCcorr_PE, BVCcorr_PE, PCcorr_PE, PWcorr_PE, oPWcorr_PE, ...
            PRcorr_PErand, oPRcorr_PErand, OVCcorr_PErand, BVCcorr_PErand, PCcorr_PErand, PWcorr_PErand, oPWcorr_PErand, ...
            CCnt_rand] = ...
    BB_subr_QuantCorr(step, ObjEncoded, ObjEncoded_prev, ObjRecalled, ObjRecalled_prev, CCnt_rand, ...
            PR_rate, oPR_rate, OVC_rate, BVC_rate, H_rate, PW_rate, oPW_rate, ...
            PRcorr_IM, oPRcorr_IM, OVCcorr_IM, BVCcorr_IM, PCcorr_IM, PWcorr_IM, oPWcorr_IM, ...
            PRcorr_PE, oPRcorr_PE, OVCcorr_PE, BVCcorr_PE, PCcorr_PE, PWcorr_PE, oPWcorr_PE, ...
            PRcorr_PErand, oPRcorr_PErand, OVCcorr_PErand, BVCcorr_PErand, PCcorr_PErand, PWcorr_PErand, oPWcorr_PErand);
    
    
    
    % estimate HD from population
    percep_flag_prev = percep_flag;
    imag_flag_prev   = imag_flag;
    HDestim_prev     = HDestim;
    ObjEncoded_prev  = ObjEncoded;
    ObjRecalled_prev = ObjRecalled;
    poptmp           = find(HD_rate==max(HD_rate));
    popmax           = poptmp(1); % in case there are two maxima
    HDestim          = mod(popmax*2*pi/100,2*pi);
    if isempty(HDestim)
        HDag = old_HD;
    else
        HDag = HDestim;
    end
    
    
    
    % track time
    step = step + 1;
    if mod(step,modstep) == 0
        disp(['sim ' int2str(simflag) ' at step ' int2str(step) '']);   % check progress of sim
        if simflag == 0
            disp(['' int2str(round(100*target_no/length(tl(:,1)))) ' percent complete, target no ' int2str(target_no) ' ']);
        end
    end
    
    
    
    % intermittent saving
    if mod(step,savestep) == 0 && save_flag
        savename = ['save ' DATAdir '/BB_vid_sim' int2str(simflag) '_RED_T' int2str(step) '_newBVCs.mat HD_rate GC_rate H_rate BVC_rate PW_rate ' ...
            'PR_rate OVC_rate oPW_rate oPR_rate TR_rate1 TR_rate2 TR_rate3 TR_rate4 TR_rate5 TR_rate6 ' ...
            'TR_rate7 TR_rate8 TR_rate9 TR_rate10 TR_rate11 TR_rate12 TR_rate13 TR_rate14 TR_rate15 TR_rate16 ' ...
            'TR_rate17 TR_rate18 TR_rate19 TR_rate20 oTR_rate1 oTR_rate2 oTR_rate3 oTR_rate4 oTR_rate5 oTR_rate6 ' ...
            'oTR_rate7 oTR_rate8 oTR_rate9 oTR_rate10 oTR_rate11 oTR_rate12 oTR_rate13 oTR_rate14 oTR_rate15 oTR_rate16 ' ...
            'oTR_rate17 oTR_rate18 oTR_rate19 oTR_rate20 Hmem1 Hmem2 Hmem3 step modstep plt_ttl_flag percep_flag ' ...
            'simflag NHx NHy HX HY GC2PCwts BVCX BVCY NBVCR NBVCTheta HDX HDY nobarrier ObjEncoded TX TY BX BY ' ...
            'ObjCenX ObjCenY old_HD HDag Xag Yag imag_flag Xag_imag_nav Yag_imag_nav oldCoordsTMP AllImagX ' ...
            'AllImagY VBX_rt VBY_rt L_r oPRCue X_for_FRM Y_for_FRM HD_for_FRM time_at_HD time_at_XY CumPCrate_at_XY CumBVCrate_at_XY CumoPRrate_at_XY ' ...
            'CumOVCrate_at_XY CumTRrate1_at_XY CumTRrate2_at_XY CumTRrate3_at_XY CumTRrate4_at_XY CumTRrate5_at_XY ' ...
            'CumTRrate6_at_XY CumTRrate7_at_XY CumTRrate8_at_XY CumTRrate9_at_XY CumTRrate10_at_XY CumTRrate11_at_XY ' ...
            'CumTRrate12_at_XY CumTRrate13_at_XY CumTRrate14_at_XY CumTRrate15_at_XY CumTRrate16_at_XY CumTRrate17_at_XY ' ...
            'CumTRrate18_at_XY CumTRrate19_at_XY CumTRrate20_at_XY PW2TRphi TR2PWphi TR2BVCphi BVC2TRphi HDphi HD2IPphi ' ...
            'HD2TRphi Rotphi IP2TRphi IP2oTRphi Hphi H2BVCphi H2PRphi BVC2Hphi BVC2Hphi_b BVC2PRphi BVC2OVCphi BVC2oPRphi ' ...
            'PR2Hphi PR2BVCphi GC2PCphi_base oPW2TRphi TR2oPWphi TR2OVCphi OVC2TRphi OcueScale BcueScale TDBU_Scale ' ...
            'TDBU_ScaleBVC TDBU_ScaleGC oPRCuescale Pmod Imod H2OVCphi oPR2OVCphi H2oPRphi OVC2Hphi OVC2BVCphi ' ...
            'OVC2oPRphi OVC2OVCphi oPW2oPWphi oPR2Hphi oPR2PWphi oPR2HDphi oPR2BVCphi oPR2oPRphi H_inhib_phi ' ...
            'BVC_inhib_phi PR_inhib_phi HD_inhib_phi TR_inhib_phi oTR_inhib_phi PW_inhib_phi OVC_inhib_phi ' ...
            'oPW_inhib_phi oPR_inhib_phi OVC2oPR_inhib_phi X_for_FRM Y_for_FRM'];
        eval(savename);
    end
    
    
    
end





%% make video and save data

if makemovie && simflag>0
    close(writerObj);
end





%% saving at the end

if save_flag
    savename = ['save ' DATAdir '/BB_vid_sim' int2str(simflag) '_RED_T' int2str(step) '_newBVCs.mat HD_rate GC_rate H_rate BVC_rate PW_rate ' ...
        'PR_rate OVC_rate oPW_rate oPR_rate TR_rate1 TR_rate2 TR_rate3 TR_rate4 TR_rate5 TR_rate6 ' ...
        'TR_rate7 TR_rate8 TR_rate9 TR_rate10 TR_rate11 TR_rate12 TR_rate13 TR_rate14 TR_rate15 TR_rate16 ' ...
        'TR_rate17 TR_rate18 TR_rate19 TR_rate20 oTR_rate1 oTR_rate2 oTR_rate3 oTR_rate4 oTR_rate5 oTR_rate6 ' ...
        'oTR_rate7 oTR_rate8 oTR_rate9 oTR_rate10 oTR_rate11 oTR_rate12 oTR_rate13 oTR_rate14 oTR_rate15 oTR_rate16 ' ...
        'oTR_rate17 oTR_rate18 oTR_rate19 oTR_rate20 Hmem1 Hmem2 Hmem3 step modstep plt_ttl_flag percep_flag ' ...
        'simflag NHx NHy HX HY GC2PCwts BVCX BVCY NBVCR NBVCTheta HDX HDY nobarrier ObjEncoded TX TY BX BY ' ...
        'ObjCenX ObjCenY old_HD HDag Xag Yag imag_flag Xag_imag_nav Yag_imag_nav oldCoordsTMP AllImagX ' ...
        'AllImagY VBX_rt VBY_rt L_r oPRCue X_for_FRM Y_for_FRM HD_for_FRM time_at_HD time_at_XY CumPCrate_at_XY CumBVCrate_at_XY CumoPRrate_at_XY ' ...
        'CumOVCrate_at_XY CumTRrate1_at_XY CumTRrate2_at_XY CumTRrate3_at_XY CumTRrate4_at_XY CumTRrate5_at_XY ' ...
        'CumTRrate6_at_XY CumTRrate7_at_XY CumTRrate8_at_XY CumTRrate9_at_XY CumTRrate10_at_XY CumTRrate11_at_XY ' ...
        'CumTRrate12_at_XY CumTRrate13_at_XY CumTRrate14_at_XY CumTRrate15_at_XY CumTRrate16_at_XY CumTRrate17_at_XY ' ...
        'CumTRrate18_at_XY CumTRrate19_at_XY CumTRrate20_at_XY PW2TRphi TR2PWphi TR2BVCphi BVC2TRphi HDphi HD2IPphi ' ...
        'HD2TRphi Rotphi IP2TRphi IP2oTRphi Hphi H2BVCphi H2PRphi BVC2Hphi BVC2Hphi_b BVC2PRphi BVC2OVCphi BVC2oPRphi ' ...
        'PR2Hphi PR2BVCphi GC2PCphi_base oPW2TRphi TR2oPWphi TR2OVCphi OVC2TRphi OcueScale BcueScale TDBU_Scale ' ...
        'TDBU_ScaleBVC TDBU_ScaleGC oPRCuescale Pmod Imod H2OVCphi oPR2OVCphi H2oPRphi OVC2Hphi OVC2BVCphi ' ...
        'OVC2oPRphi OVC2OVCphi oPW2oPWphi oPR2Hphi oPR2PWphi oPR2HDphi oPR2BVCphi oPR2oPRphi H_inhib_phi ' ...
        'BVC_inhib_phi PR_inhib_phi HD_inhib_phi TR_inhib_phi oTR_inhib_phi PW_inhib_phi OVC_inhib_phi ' ...
        'oPW_inhib_phi oPR_inhib_phi OVC2oPR_inhib_phi X_for_FRM Y_for_FRM ' ...
        'H2OVCwts OVC2Hwts oPR2OVCwts OVC2oPRwts oPR2HDwts oPR2Hwts H2oPRwts ' ...
        'PRcorr_IM oPRcorr_IM OVCcorr_IM BVCcorr_IM PCcorr_IM PWcorr_IM oPWcorr_IM ' ...
        'PRcorr_PE oPRcorr_PE OVCcorr_PE BVCcorr_PE PCcorr_PE PWcorr_PE oPWcorr_PE ' ...
        'PRcorr_PErand oPRcorr_PErand OVCcorr_PErand BVCcorr_PErand PCcorr_PErand PWcorr_PErand oPWcorr_PErand '];
    eval(savename);
end

if simflag == 242
    eval(['save ' DATAdir '/HPClesionObj_corrdata corr1OVC corr2OVC corr3OVC corr4OVC corr1oPW corr2oPW corr3oPW corr4oPW']);
end
if simflag == 241
    eval(['save ' DATAdir '/HPCnolesionObj_corrdata corr1OVC corr2OVC']);
end





