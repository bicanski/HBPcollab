%% Get sample rat trajectory from caswell's data
%
% Optionally takes ind as an argument, which will load one of several
% precomputed trajectories
%
function X = get_sample_rat_trajectory(par,varargin)

% Parse arguments
if nargin==1
    ind = 1;
else
    ind = varargin{1};
end

% Load from data
here = fileparts(which('get_sample_rat_trajectory'));
d = dir([here,'/../../../..\Github\MatLab_Tools\Rat_Trajectories\Trajectories\*']);
d = d(3:end);
load([here,'/../../../..\Github\MatLab_Tools\Rat_Trajectories\Trajectories\',d(ind).name])

% Make sure the trajectory fits within the required simulation
X = [TRAJ.x(:),TRAJ.y(:)];
X = X / max(X(:)) * par.X;

X(:,1) = (X(:,1) - par.X/2)*0.999 + par.X/2;
X(:,2) = (X(:,2) - par.X/2)*0.999 + par.X/2;

% Calculate the approximate time scaling factor
x = X(:,1); y = X(:,2);
dx = x(1:(end-1))-x(2:end); dy = y(1:(end-1)) - y(2:end);
DX = sqrt(dx.^2 + dy.^2);
v = mean(DX); % Average speed
dist = sum(DX); % Total distance travelled

% Scale based on the total distance the rat needs to travel based on the
% parameters
dist_needed = par.traj.vel*par.T;
if dist_needed/dist > 1
    X = repmat(X,ceil(dist_needed/dist),1);
end

%{
% Interpolate or skip points depending on required velocity
tvec = 1:(par.traj.vel*par.learn.dt / v):length(X);
x = interp1(1:length(X),X(:,1),tvec);
y = interp1(1:length(X),X(:,2),tvec);

X = [x(:),y(:)];
%}

end