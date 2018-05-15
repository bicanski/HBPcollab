
%% Simulation options
%
% Nothing in this file can depend on a value already stored in par (for
% example setting:
%
% par.opts.RefreshRate = max([100,round(par.T/100)]);            
%
% ...since it is possible that the par values will be later overwritten,
% leading to errors. Instead, place anything depending on par in the
% kf_init() file

function par = kf_sim_opts(par)

%% Simulation types
    par.opts.sim.visual_input = 'pc'; % {'pc','measurement','bvc'}
    par.opts.perfect_tracking = 0; % Remove the noise from the tracking input for debugging purposes
    
%% Computing
    par.opts.gpu_option = 0;
    par.opts.debug = 0;
    
    % Choose which computations to perform
    par.opts.update.CANsheet = 1;
    par.opts.update.pc_firing = 1;
    
    par.opts.independent_sensory_measurements = 0; % Whether or not each PC receives an independently noised estimate of current location

%% Plotting

    % Options
    par.opts.RefreshRate =  max([100,round(par.T/10)]);             % Update plots every n iterations
    par.opts.plot = 1;                                              % Plot yes or no
    
    % Choose which figures to display
    par.opts.drawfigs.CANsheet = 1;
    par.opts.drawfigs.stats = 1;
    par.opts.drawfigs.pos_estimates = 0;
    par.opts.drawfigs.readout = 1;
    par.opts.drawfigs.pc_firing = 1;
    par.opts.drawfigs.weights = 0;
    par.opts.drawfigs.changeweights = 0;
    par.opts.drawfigs.threshold = 0;
    par.opts.save_plots = 1;
    par.opts.save_final_weights = 1;

    % Appearance
    par.opts.layout = [];
    
    par.opts.rateMapSmoothing = 5; % Boxcar filter of width p pixels
    
    par.opts.numMons = 1; % Number of monitors

% Statistics
par.opts.save_stats = 1;        % Save stats or not

par.opts.StatsGap = round(par.T/10);        % Record lighter stuff semi frequently
par.opts.SummaryGap = round(par.T/5);       % Record big stuff infrequently

%% Experiment specific

    %% Stretchy box
    par.opts.stretchy_box.t_compress = par.opts.SummaryGap;
    par.opts.stretchy_box.t_reexpand = par.opts.SummaryGap*3; % One section of normal size, 2 sections of compression and 2 sections of noral size again
    par.opts.stretchy_box.scale_factor = 0.75;

    %% Novelty expansion
    par.opts.novelty_expansion.t_contract = nan;
    par.opts.novelty_expansion.t_modify_weights = nan;
    par.opts.novelty_expansion.scale_factor = [];
    par.opts.novelty_expansion.force_recompute_base_sheet = 0;

    par.opts.novelty_expansion.weight_scale_func = []; 
    par.opts.novelty_expansion.weight_scale_func_params = [];
    par.opts.novelty_expansion.section_length = [];
    par.opts.novelty_expansion.block_length = [];

    %% Two compartments
    par.opts.env_type = 'empty';

    %% Grid cell driven PC remapping
    par.opts.pc_remapping.option = 0;

    par.opts.pc_remapping.bvc.connProb = 0.2;       % Connection probability from BVC->PC
    par.opts.pc_remapping.bvc.connMean = 1;         % Mean BVC->PC connection strength
    par.opts.pc_remapping.bvc.connVar = 0.1;        % BVC->PC connection variance
    par.opts.pc_remapping.bvc.connDist = 'logn';    % Distribution describing BVC->PC connections
    par.opts.pc_remapping.bvc.N = 100;              % Number of BVCs
    par.opts.pc_remapping.bvc.sigA = 0.05;          % Angular tuning of BVC receptive fields
    par.opts.pc_remapping.bvc.sig0 = 122e-3;        % Radial tuning of BVC receptive fields
    par.opts.pc_remapping.pc.w20 = 5;               % Default mean strength of GC->PC weights

    %% Replay annealing
    par.opts.replay.option = 0;                     % Run with replay
    par.opts.replay.pw_dist_rec_gap = 10;           % How often to record the inferred pairwise distances between place cells
    
    par.opts.replay.p2p_min_dist = 0;               % The fraction of cells to which each pc is connected
    par.opts.replay.p2p_max_dist = Inf;             % The fraction of cells to which each pc is connected

    par.opts.replay.stop_thresh = 0; %1e-3;         % Threshold percentage change in total entropy below which we stop the simulation
    par.opts.replay.schedule_type = 'max_update';   % How to schedule the nodes for updating : {'sequential','entropy'}
    par.opts.replay.max_iter = 10;                  % Maximum number of iterations
    par.opts.replay.bias_param = 5;                 % Bias parameter for random walk scheduling
    par.opts.replay.stats_gap = 1;                  % Interval for computing more expensive stats
    par.opts.replay.sync_async = 'sync';            % Synchronous or asynchronous updating method
    par.opts.replay.pred_err_thresh = 1;            % Threshold for initiating replay events
    par.opts.replay.plot_option = 0;                % Plot replay event or not
    par.opts.replay.gpu_option = 1;                 % Use GPU
    par.opts.replay.debug_option = 0;               % Debug option
    par.opts.replay.spwr_time = [];                 % Time at which we choose replay events to happen
    par.opts.replay.record_video = 0;               % Record video of the replay process
    
    par.opts.replay.update_thresh = 0;              % Threshold for the difference between new and old beliefs above which a node updates its messages
    
end