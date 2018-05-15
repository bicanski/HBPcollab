%% Simulation parameters
function par = kf_default_sim_params(par)

par.T = 1e5; % Number of iterations
par.X = 1; % Enviroment size X
par.Y = 1; % Environment size Y

par.plot.grid_res = 1/100; % Grid resolution of plot

par.Q0 = 0.5e-1; % Default movement noise variance (sig^2)
par.R0 = 1e-3; % Default sensory  noise variance (sig^2)

% Trajectory
par.traj.ang_var = 0.5; % Angular variance (how quickly the agent deviate from its current heading direction)
par.traj.vel = 0.05; % Movement speed
par.traj.type = 'random_exploration'; % 'edge'
par.traj.alternate_time = 1e3;
par.traj.real_ind = 1; % Index of real rat trajectory, 1-30

if any(strcmpi(par.traj.type,{'edge','alternate_egde_plus_middle'}))
    par.x0 = [0,0]; % Bottom corner
else
    par.x0 = [0.5;0.5]; % The agent's starting location
end

%% Grid parameters
par.grid.scale = 0.2;
par.grid.offset = [0,0];
par.grid.phi = 0;
par.grid.firing_rate = 1e5;

par.grid.Ncell = 5; % The characteristic dimension of the base grid cell sheet

%% Place cell parameters 
par.pc.npc = 500;             % Number of place cells
%par.pc.C0_base = 1e-3;       % Tuning width
par.pc.distribution = 'grid'; % {'uniform','nonuniform'}
par.pc.tuning = 'constant';
par.pc.only_within_boundaries = 1;
par.pc.tc = 1; % Membrane time-constant (only used in remapping simulations)
par.pc.Vrest = 0; % Resting membrane potential in mV

par.pc.distribution_func.f = []; % To be used with the distribution:'nonuniform' setting
par.pc.distribution_func.params = [];

par.pc.noise_func.f = []; % Non-uniform noise parameters 
par.pc.noise_func.params = [];

par.pc.fmax_func = []; % Function to control the max firing rate of PCs as a function of space
par.pc.mu = []; % Location soof place field centres

%% Learning parameters
par.learn.rule = 'bcm';
par.learn.dt = 1;
par.learn.learning_order = 'inbetween'; % {'inbetween','after'}

% PCs
par.learn.pc.rate = 1e-4; % Learning rate
par.learn.pc.rate_p2p = par.learn.pc.rate; % Learning rate of recurrent connections between PCs
par.learn.wact = @(x) x.*(x>0); % Weight thresholding function
par.learn.measurement_weight = 0.5;

par.pc.prelearned_weights = [];

%% Misc
par.session_name = '';
par.session_date = seshdate();
par.session_tag = [];

par.multi.param_vec   = []; % Multiple trial structure
par.multi.param_names = []; % Multiple trial structure

end
