%% Test the model from Barry and Burgess (2007)
%
% Generate a bunch of BVCs and use them to learn associations to PCs
% Associations change according to the BCM rule
close all; clearvars; clc;

%% Initialisation
% Load some parameters from the main KF settings file
par = kf_default_sim_params([]);

% And some default sim options
par = kf_sim_opts(par);

% Define the parameter file
par = kf_bvc_param_file(par);

% Define the environment
par = kf_generate_bvc_environment(par);

%% Generate example firing maps
[ym,xm] = meshgrid(0:0.01:1);
F = zeros([par.bvc.nBVC,size(xm)]);
for i = 1:size(xm,1)
    for j = 1:size(xm,2)
        F(:,i,j) = kf_bvc_func([xm(i,j),ym(i,j)],par)';
    end
end

%% Run main loop
x = zeros(3,par.T); x(:,1) = [par.var.x0(:);par.var.hd0];
for t = 2:par.T
    
    % Generate new coordinate
    x(:,t) = kf_trajectory_func(x(:,t-1),par);
    
    % Update the BVC firing
    F = kf_bvc_func(x,par);
    
    % Generate PC firing
    
end
    