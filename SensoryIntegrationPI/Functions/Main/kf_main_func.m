%% Run the Kalman Filter Simulation
function OUT = kf_main_func(simargs)

% Define the simulation parameters
par = kf_default_sim_params();

% Simulation options
par = kf_sim_opts(par);

% Parse additional arguments from an external calling script, if they exist
if nargin==1
    par = updateStruct(simargs,par);
end

% Initialise
[s,par] = kf_init(par);

% Run the main simulation
[stats,S_record,par] = kf_run_main(par,s);

% Save if desired
OUT = kf_output(par,stats,S_record);

end