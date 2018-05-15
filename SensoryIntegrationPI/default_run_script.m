%% Default run script
clearvars; close all; clc

% Simulation specific arguments can be supplied by defining a simargs
% structure which is fed to kf_main_func
%
% E.g.:
%
% simargs.Q0 = 1;
% simargs.R0 = 1;
%
% kf_main_func(simargs)
%
% Will overwrite the defaults in the kf_default_sim_args file

%% Session name
simargs.session_name = 'test/'; % Must be followed by a '/'

%% Define simargs

% * defined simargs here * %
%simargs.T = 1e5;

simargs.grid.scale = 0.3;

simargs.T = 1e5;

simargs.opts.RefreshRate = 5e3;

%% Run the function
if exist('simargs','var')==1
    OUT = kf_main_func(simargs);
else
    OUT = kf_main_func();
end