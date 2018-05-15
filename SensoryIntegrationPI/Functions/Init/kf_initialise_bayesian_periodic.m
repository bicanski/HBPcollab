function [par,s] = kf_initialise_bayesian_periodic(par,s)

%% PI distribution. We need both the hex and cart sheets, the latter is
% larger than the normal sheet because we use it in the convolution
[~,par.bys.p.sys.XC,par.bys.p.sys.MSK,par.bys.p.sys.MSK_ind] = ... % Generate base coordinates
    kf_generate_X_base_sheet_from_square(par);
            
if par.opts.gpu_option
    par.bys.p.sys.XC = gpuArray(par.bys.p.sys.XC);
    par.bys.p.sys.MSK_ind = gpuArray(par.bys.p.sys.MSK_ind);
    par.bys.p.sys.MSK = gpuArray(par.bys.p.sys.MSK);
end

s.bys.p.P_sys = hex_circ_gauss(s.x,par.plot.wrappedX_baseSheet,par.grid.scale,par.grid.offset,... % Generate the movement distribution
    s.Q,par.grid.phi,par,par.plot.wrappedX_nReps);

s.bys.p.P_sys = s.bys.p.P_sys / sum(s.bys.p.P_sys(:)); % Normalise to 1

[s.bys.p.P_sys_msk,~,par.bys.p.sys.cart_inds] = kf_tile_movement_cart_distro(...                        % Tile the distribution so we can convolve it
    par.bys.p.sys.MSK,par.bys.p.sys.MSK_ind,par.bys.p.sys.XC,s.bys.p.P_sys,par);

%% Coordinate grid for the measurement estimate distribution (same as
% the sys distribution)
s.bys.p.P_est = ...
    hex_circ_gauss(s.z,par.plot.wrappedX_baseSheet,par.grid.scale,par.grid.offset,... % Generate the estimate distribution
    s.R(:,:,1),par.grid.phi,par,par.plot.wrappedX_nReps);

s.bys.p.P_est = s.bys.p.P_est / sum(s.bys.p.P_est(:)); % Normalise

par.bys.p.est.MSK     = par.bys.p.sys.MSK; % Copy over these parameters from the sys distribution
par.bys.p.est.MSK_ind = par.bys.p.sys.MSK_ind;

s.bys.p.P_est_msk = par.bys.p.est.MSK; % Initialise the estimate distribution mask
s.bys.p.P_est_msk(par.bys.p.est.MSK_ind) = s.bys.p.P_est;

%% Initalise the posterior distribution
s.bys.p.P_pos = hex_circ_gauss(s.x,par.plot.wrappedX_baseSheet,par.grid.scale,par.grid.offset,... % Generate the movement distribution
    1e-4*[1,0;0,1],par.grid.phi,par,par.plot.wrappedX_nReps);

s.bys.p.P_pos = s.bys.p.P_pos / sum(s.bys.p.P_pos);

par.bys.p.pos.XC  = par.bys.p.sys.XC;
par.bys.p.pos.MSK = par.bys.p.sys.MSK;
par.bys.p.pos.MSK_ind = par.bys.p.sys.MSK_ind;
par.bys.p.pos.MSK_ind2 = setdiff(1:numel(par.bys.p.pos.MSK),par.bys.p.pos.MSK_ind); % Indices of the elements of the mask that we dont care about

s.bys.p.P_pos_msk = par.bys.p.pos.MSK;
s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind) = s.bys.p.P_pos;

end