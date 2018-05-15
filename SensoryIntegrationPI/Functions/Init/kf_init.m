%% Initialise the system parameters
function [s,par] = kf_init(par)

% Grid cell parameters
par.grid.NcellSheet = 1 + sum((1:par.grid.Ncell)*6) - par.grid.Ncell*3 - double(par.grid.Ncell>0);

% Correct the grid scale (it was a factor of 2/sqrt(3) out anf this it's
% easier to correct it here than to go through all the code)
GS0 = 0.2; % This is the grid scale to which all other parameters are calibrated
noiseScaleFactor = (par.grid.scale/GS0)^2;

% Noise parameters
s.X = par.x0.*[par.X;par.Y];           % The true location

s.Q = par.Q0*noiseScaleFactor*[1,0;0,1]; % * (par.grid.scale/GS0)^2 *[1,0;0,1];         % Process noise, 2 S.D (variance = S.D^2). The noise associated with the imperfect internal update.

s.xd = s.X;             % Intermediate internal (a priori) estimate. Initialise to the true state.
s.Pd = [];              % Intermediate internal (a priori) update covariance. 

s.xPI = s.X;            % Keep a track of what a pure PI estimate would look like

%s.R = par.R0 * noiseScaleFactor * repmat([1,0;0,1],[1,1,par.pc.npc]);  ! Note ! I se this later now, because it is always equal to the PC tuning width !         % Measurement noise vector
s.H = [1,0;0,1];        % Measurement transformation matrix is a 1:1 mapping

s.x = s.X;              % Initial a posteriori state estimate. Position in X and Y. Intialise to the true state

s.P = par.R0*noiseScaleFactor.*[1,0;0,1];       % Covariance of initial a posteriori state estimate

s.z = s.x;              % Measurement

%% The environment grid
par.plot.xg = 0:par.plot.grid_res:par.X; % The x vector
par.plot.yg = 0:par.plot.grid_res:par.Y; % The y vector
[ym,xm] = meshgrid(par.plot.yg,par.plot.xg);
par.plot.grid = [xm(:),ym(:)]; % Coordinates of the environmental grid
par.plot.gridsize = size(xm);
par.plot.wrappedX_nReps = 10;
[~,par.plot.wrappedX,par.plot.wrappedX_nReps]          = hex_circ_gauss(par.grid.offset,par.plot.grid,par.grid.scale,par.grid.offset,s.P,par.grid.phi);

% Generate environment mask
par = kf_generate_env_mask(par);

% The base cell grid coordinates (one voronoi region)
par.plot.sc = 1/sqrt(3)*par.grid.scale/par.grid.Ncell; % The discretization grid
[par.plot.grid_baseSheet,par.plot.cart_baseSheet,par.plot.cart_mask,par.plot.cart_mask_ind] = kf_generate_X_base_sheet_from_square(par);
[~,par.plot.wrappedX_baseSheet,par.plot.wrappedX_nReps] = hex_circ_gauss([0,0],par.plot.grid_baseSheet,par.grid.scale,par.grid.offset,s.P,par.grid.phi);

%% Place cells parameters
par.pc.C0 = par.R0 * noiseScaleFactor;          % Measurement noise vector

par = kf_generate_pc_population(par);

s.R = par.pc.C; % Set the noise associated with the sensory estimate as equal to the tuning width

s.pc.Th_v = zeros(par.pc.npc,1);
s.pc.F_mean = ones(par.pc.npc,1);

% PC->GC weights
if ~isempty(par.pc.prelearned_weights)
    s.pc.w = par.pc.prelearned_weights;
else
    s.pc.w = 1e2*rand(par.pc.npc,par.grid.NcellSheet) / par.pc.npc;
end

% Recurrent weights
if par.opts.pc_remapping.option
    s.pc.w_p2p = kf_initialise_recurrent_weights(par);
else
    s.pc.w_p2p = zeros(par.pc.npc,par.pc.npc);
end

% Define the connected pairs of cells according to some predefined connected fraction
%tmp = rand(size(s.pc.w_p2p)); 
%s.pc.c_p2p = tmp<par.opts.replay.p2p_conn_frac;
s.pc.c_p2p = par.pc.Dpw0_euc<par.opts.replay.p2p_max_dist &...
             par.pc.Dpw0_euc>par.opts.replay.p2p_min_dist;

% GC->PC weights
s.pc.Th_v2 = zeros(par.pc.npc,1);

if par.opts.pc_remapping.option
    par.learn.pc.rate2 = par.learn.pc.rate;
    s.pc.w2 = 2*par.opts.pc_remapping.pc.w20*rand(par.pc.npc,par.grid.NcellSheet); %ones(par.pc.npc,par.grid.NcellSheet)/par.grid.NcellSheet;
else
    par.learn.pc.rate2 = 0; % No learning
    s.pc.w2 = zeros(par.pc.npc,par.grid.NcellSheet);
end
   
% Initialise firing
s.pc.F = ones(par.pc.npc,1);
s.pc.V = zeros(size(s.pc.F));
s.pc.measurement_weight = par.learn.measurement_weight;
s = kf_get_measurement_from_sensory(par,s);

if ~isempty(par.pc.fmax_func)
    par.pc.fmax = par.pc.fmax_func(par.pc.mu(:,1),par.pc.mu(:,2));
else
    par.pc.fmax = 1;
end

%% Replay settings
par.opts.replay.noise_conv_factor = [sqrt(s.R(1)),sqrt(s.Q(1))];                          % Distance dependent noise between place cells is the same as path integration noise

%% Bayesian updating (periodic)
[par,s] = kf_initialise_bayesian_periodic(par,s);
    
%% Initialise settings
par.plot.init = 0;                                                         % This is set to 1 once the plots have been initially drawn
par.opts.save_gap = round(par.T/100);

par.opts.grid_readout_reset = par.opts.RefreshRate;

par.opts.ResetGap = par.opts.SummaryGap;                                   % Reset the firing rates every now and again

par.plot.mon_cell = round(par.grid.NcellSheet/2);

%% Misc, shortcuts etc.
par.misc.names = {'pure_pi','apriori','measurement','aposteriori'};

%% Convert to GPU if required
if par.opts.gpu_option
    gpuVar = {'s.pc.Th_v',...
        's.pc.w',...
        'par.plot.wrappedX_baseSheet',...
        's.bys.p.P_pos',...
        's.bys.p.P_pos_msk',...
        's.bys.p.P_sys',...
        's.bys.p.P_sys_msk',...
        's.bys.p.P_est',...
        's.bys.p.P_est_msk',...
        'par.bys.p.est.MSK_ind',...
        };
    
    for g=1:length(gpuVar); eval([gpuVar{g},'=gpuArray(',gpuVar{g},');']); end
end

end