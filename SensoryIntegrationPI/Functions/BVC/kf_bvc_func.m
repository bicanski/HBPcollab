%% BVC firing function

% sigR(d) = ((d/beta) + 1)*sig0
%
% In Barry and Burgess (2007), values are sigA = 0.2 radians, beta = 1830 mm, and sig0 = 122 mm.
%
% The preferred firing direction (phi) for each BVC set was selected randomly from the continuous range 0�2pi.
%
% The preferred firing distance (d) for each BVC set was selected randomly from the following values:
% 81.0, 169.0, 265.0, 369.0, 482.5, 606.5, and 741.0 mm.
% It can be seen that BVCs with shorter preferred ?ring distance, and hence narrower tuning curves, are more densely represented.
%
% sigR - radial firing variance
% r - allocentric radial distance
% d - tuning distance
% the - current angle to boundary
% phi - tuning angle

% Simulate a multi-layered neural network that has head direction inputs at
% the bottom in addition to head direction sensitive BVC inputs. Hopefully,
% grid cells come out at the bottom that are directionally sensitive
% (conjunctive GCs), whereas they lose direction sensitivity as we go up
% the layers.
%
% In a biological sense, the directional sensitivity is inherent in the BVC
% representations since they're likely to be visually driven, and so it's
% necessary to include it in the model. However, it's not clear whether
% there is a functional advantage to directional snesitity in GCs if
% they're only meant as a coding mechanism. So, if they get more stable as
% we go from presubiculum (PrS) to parasubiculum (PaS) to mEC, they might
% lose their directional sensitivity.
%
% Environmental parameters to supply:
%
% env_mask : A square binary matrix containing 1s where a boundary exists, and 0s
% elsewhere
%
% Cell parameters to supply:
%
% d : The tuning distance
% sig0 : The basal tuning width (this is used to calculate sigR, which
% varies with distance)
% phi : The allocentric tuning direction

function F = kf_bvc_func(x,par)

% Defaults (chnaged by varargin parser)
gpu_option = 0;

%% Constant parameters (set according to literature)
sigA = 0.2;
sig0 = 122e-3;
beta = 1830e-3;

%% Do main function
sigR = (par.bvc.tuning_distance(:)/beta + 1)*sig0;

% Distance between current point and all the boundary elements
DX = x(1)-par.env.x_bound;
DY = x(2)-par.env.y_bound;
R = sqrt(DX.^2+DY.^2);

% Angle that the element subtends is given by:
% dTheta = 2*atan(dEl./(2*R)) where dx is the pixel width
F = 2*atan(par.env.dEl./(2*R)).*...
    exp( -(  R-par.bvc.tuning_distance(:)'  ).^2 ./ (2*sigR(:)'.^2) ) .*... / sqrt(2*pi*sigR^2).*...
    exp( -(  -pi+mod(atan2(DY,DX)-par.bvc.phi(:)'+pi,2*pi) ).^2 ./ (2*sigA(:)'.^2) ); % / sqrt(2*pi*sigA^2);

F = sum(F,1); % Sum the contributions of all cells

F(isnan(F))=0;

F = gather(F);

end