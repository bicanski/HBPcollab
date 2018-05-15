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
function [s,par] = kf_do_movement_update(par,s)
%
% P_pos : Previous / new posterior distribution
% P_sys : System update distribution (PI)
% P_est : Measurement estimate distribution (sensory estimate)

%par.opts.debug=1; s.u = [0.25,0]*par.grid.scale;

%% Generate movement distribution

NK = par.plot.sc^2*sqrt(3)/2; % Normalisation constant

% Noise is modulated by animal speed
Q = s.Q * s.absVel.^2; 

% Generate the movement distribution
s.bys.p.P_sys = NK*hex_circ_gauss(...
    s.u,par.plot.wrappedX_baseSheet,...
    par.grid.scale,par.grid.offset,...
    Q,par.grid.phi,par,par.plot.wrappedX_nReps);

s.bys.p.P_sys_msk(par.bys.p.sys.cart_inds) = repmat(s.bys.p.P_sys,7,1);

%% Misc. initialisation
mon_cell = round(par.grid.NcellSheet/2);
X_mon_cell = par.plot.grid_baseSheet(mon_cell,:);

%% Do movement update

% Convolve the movement distribution with the previous posterior
s.bys.p.P_pos_msk                         = real(conv_fft2(s.bys.p.P_pos_msk,s.bys.p.P_sys_msk,'same')); % Note that the order of the two masks matters apparently
s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind2) = 0;                                                           % Set the element outside the grid to zero since we dont care about them and we dont want them to affect later calculations

% Convert the mask to the actual distribution (the mask is just a
% convenient structure for doing the convolution)
s.bys.p.P_pos = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind);
s.bys.p.P_apriori = s.bys.p.P_pos; % Save this intermediate quantity for later

% Get the apriori estimate by taking the grid cell with max. firing
[~,indmax]        = max(s.bys.p.P_pos(:)); 
s.xd              = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.apriori = gather(s.bys.p.P_pos(mon_cell)); % Get the firing

% Update the pure PI estimate of position. This is the PI movement in
% this time added to the last pure PI estimate
s.xPI = wrap_xy_in_hex(s.xPI + s.u,par.grid.phi,par.grid.scale,par.grid.offset); s.xPI = s.xPI(:);
s.bys.p.f.pure_pi = hex_circ_gauss(s.xPI',X_mon_cell,par.grid.scale,par.grid.offset,s.Q,par.grid.phi,par);


%% Make sure no values go to zero or negative
FN = fieldnames(s.bys.p.f);
for fn=1:length(FN)
    s.bys.p.f.(FN{fn})(s.bys.p.f.(FN{fn})<eps) = eps;
end

end