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
function [s,par] = kf_do_measurement_update(par,s)
%
% P_pos : Previous / new posterior distribution
% P_sys : System update distribution (PI)
% P_est : Measurement estimate distribution (sensory estimate)

%par.opts.debug=1; s.u = [0.25,0]*par.grid.scale;

%% Generate movement distribution

NK = par.plot.sc^2*sqrt(3)/2; % Normalisation constant

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

%% Misc. initialisation
mon_cell = round(par.grid.NcellSheet/2);
X_mon_cell = par.plot.grid_baseSheet(mon_cell,:);

%% Do measurement update

% Measurement update
s.bys.p.P_pos_msk = real(s.bys.p.P_pos_msk.* ...%.^(1-par.learn.measurement_weight).*...
                         s.bys.p.P_est_msk);    %.^(  par.learn.measurement_weight));

s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind2) = 0; % Set the element outside the grid to zero since we dont care about them and we dont want them to affect later calculations

[~,indmax] = max(s.bys.p.P_est(:)); 
s.z_wrap = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.measurement = gather(s.bys.p.P_est(mon_cell)); % Get the firing

% Normalise
s.bys.p.P_pos_msk = s.bys.p.P_pos_msk / sum(s.bys.p.P_pos_msk(:));
s.bys.p.P_pos     = s.bys.p.P_pos_msk(par.bys.p.pos.MSK_ind);

% Get estimate of position
%
% Note that we store the firing of all the grid cells here because we want
% to analyse their relative properties
[~,indmax] = max(s.bys.p.P_pos(:)); s.x = [par.plot.grid_baseSheet(indmax,1),par.plot.grid_baseSheet(indmax,2)]';
s.bys.p.f.aposteriori = gather(s.bys.p.P_pos(:)); % Get the firing

%% Make sure no values go to zero or negative
FN = fieldnames(s.bys.p.f);
for fn=1:length(FN)
    s.bys.p.f.(FN{fn})(s.bys.p.f.(FN{fn})<eps) = eps;
end


end