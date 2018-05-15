%% Update the states for the ap case using recursive bysian filtering
%
% See http://people.csail.mit.edu/mrub/talks/filtering.pdf for details
%
% Prediction step (PI):
%
% p(x_k | z_{1:k-1}) =
%       Int p(x_k | x_{k-1}) * p(x_{k-1} | z_{1:k-1}) dx_{k-1} _________(1)
%
% The first term is the system model and the second is the previous
% posterior distribution.
%
% Update step (integrating the measurement):
%
% p(x_k | z_{1:k}) =
%       p(z_k | x_k) * p(x_k | z_{1:k-1}) / p(z_k | z_{1:k-1}) _________(2)
%
% where the normalizing constant:
%
% p(z_k | z_{1:k-1}) = Int p(z_k | x_k) * p(x_k | z_{1:k-1}) dx_k ______(3)
%
% The first term on the top is the measurement model. The second term on
% the top is the current prior (obtained from equation (1))
%
%%
function [s,par] = kf_update_state_bayesian_periodic(par,s)
%
% P_pos : Previous / new posterior distribution
% P_sys : System update distribution (PI)
% P_est : Measurement estimate distribution (sensory estimate)

%% 1. Generate the new distributions

NK = par.plot.sc^2*sqrt(3)/2; % Normalisation constant

% Noise modulated by velocity (Q = sig^2, noise model is sig*v, i.e.
% standard deviation proprtional to velocity)
Q = (sqrt(s.Q) * s.absVel).^2;                                            

% Generate the movement distribution
s.bys.p.P_sys = NK*hex_circ_gauss(...                                       
    s.u,par.plot.wrappedX_baseSheet,...
    par.grid.scale,par.grid.offset,... 
    Q,par.grid.phi,par,par.plot.wrappedX_nReps);

s.bys.p.P_sys_msk(par.bys.p.sys.cart_inds) = repmat(s.bys.p.P_sys,7,1);

% Measurement
switch lower(par.opts.sim.visual_input)
    case 'measurement'
        s.bys.p.P_est = NK*hex_circ_gauss(s.z,par.plot.wrappedX_baseSheet,par.grid.scale,par.grid.offset,... % Generate the movement distribution
            s.R,par.grid.phi,par,par.plot.wrappedX_nReps);
        
        s.bys.p.P_est_msk(par.bys.p.est.MSK_ind) = s.bys.p.P_est;
    case 'pc'
        s.bys.p.P_est = s.pc.gc_u;
        s.bys.p.P_est_msk(par.bys.p.est.MSK_ind) = s.bys.p.P_est;
    case 'bvc'
        % Insert code here...
end

%% 2. Generate the new posterior distribution
M = 2; N = 5; p = 1;
mon_cell = round(par.grid.NcellSheet/2);
X_mon_cell = par.plot.grid_baseSheet(mon_cell,:);

if par.opts.debug; subplot(M,N,1); kfh_plot('pos'); title('POS_{t-1}'); end
if par.opts.debug; subplot(M,N,7); kfh_plot('sys'); title('SYS_t'); end

%% 2a. Movement update

% Convolve the movement distribution with the previous posterior
s.bys.p.P_pos_msk                         = real(conv_fft2(s.bys.p.P_pos_msk,s.bys.p.P_sys_msk,'same')); % Note that the order of the two masks matters apparently
s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind2) = 0;                                                           % Set the element outside the grid to zero since we dont care about them and we dont want them to affect later calculations

% Convert the mask to the actual distribution (the mask is just a
% convenient structure for doing the convolution)
s.bys.p.P_pos = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind);

% Get the apriori estimate by taking the grid cell with max. firing
[~,indmax]        = max(s.bys.p.P_pos(:)); 
s.xd              = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.apriori = gather(s.bys.p.P_pos(mon_cell)); % Get the firing

% Update the pure PI estimate of position. This is the PI movement in
% this time added to the last pure PI estimate
s.xPI = wrap_xy_in_hex(s.xPI + s.u,par.grid.phi,par.grid.scale,par.grid.offset); s.xPI = s.xPI(:);
s.bys.p.f.pure_pi = hex_circ_gauss(s.xPI',X_mon_cell,par.grid.scale,par.grid.offset,s.Q,par.grid.phi,par);

if par.opts.debug; subplot(M,N,3); p=p+1; s.bys.p.P_pos = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind); kfh_plot('pos'); title('POS''_t'); end

%% 2b. Measurement update

% Measurement update
s.bys.p.P_pos_msk = real(s.bys.p.P_pos_msk.* ...%.^(1-par.learn.measurement_weight).*...
                         s.bys.p.P_est_msk);    %.^(  par.learn.measurement_weight));

s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind2) = 0; % Set the element outside the grid to zero since we dont care about them and we dont want them to affect later calculations

[~,indmax] = max(s.bys.p.P_est(:)); 
s.z_wrap = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.measurement = gather(s.bys.p.P_est(mon_cell)); % Get the firing

% Normalise
s.bys.p.P_pos_msk = s.bys.p.P_pos_msk / sum(s.bys.p.P_pos_msk(:));
s.bys.p.P_pos = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind);

if par.opts.debug; subplot(M,N,9); p=p+1; kfh_plot('est'); title('EST_t'); end
if par.opts.debug; subplot(M,N,5); p=p+1; s.bys.p.P_pos = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind); kfh_plot('pos'); title('POS_t'); set(gcf,'color','w'); end

% Get estimate of position
%
% Note that we store the firing of all the grid cells here because we want
% to analyse their relative properties
[~,indmax] = max(s.bys.p.P_pos(:)); s.x = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.aposteriori = gather(s.bys.p.P_pos(:)); % Get the firing

end