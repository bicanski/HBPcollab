% Generate circular gaussian on hexagonal lattice
%
% F = hex_circ_gauss(X,L,M,C,P)
%
% Inputs:
% 
% X : [x,y] two-column matrix of query coordinates
% L : [1,2] two element vector denoting the scale of the grid in the x and
%           y coordinates. L=0.2 indicates 5 fields per m
% M : [1,2] \in {0,1} two element vector denoting the grid offset in the i 
%           and j (0 and 60 deg.) directions
% C : [2,2] covariance matrix
% P : [1,1] scalar denoting the orientation of the grid
%
% Outputs:
%
% F : [size(X)] Vector denoting the firing at each of the [x,y] points in X
%
% Example:
%
% [xm,ym] = meshgrid(0:0.01:1); F = hex_circ_gauss([xm(:),ym(:)],[0.5,0.5],[0,0],[0.01,0;0,0.01],0); imagesc(reshape(F,101,101)); axis square

function [F,X,nReps] = hex_circ_gauss(M_Gauss,X,L,M_Base,C,P,par,varargin)

% Defaults
nReps = 0;

% Optional arguments
if ~isempty(varargin); nReps=varargin{1}; end % The function can accept coordinates that have already been wrapped so speed up simulation

% Correct the input size
if size(X,2)~=2; error('Coordinates must be provided as a two column [x(:),y(:)] matrix'); end
M_Gauss = M_Gauss(:)';
M_Base = M_Base(:)';

% Map the gaussian's mean (the positions of the animal) to the base region
M_Gauss = wrap_xy_in_hex(M_Gauss,P,L,M_Base);

if nReps==0

% Map the coordinates to the base Voronoi region
X = wrap_xy_in_hex(X,P,L,M_Base);

% Repeat the coordinates
% Calculate enough repeats such that that the gaussian has decayed to some
% fixed percent of its max value. This means that we can adjust the numerb
% of repeats according to the covariance and make the computation cheaper
% when possible while preserving the accuracy of the approximation. Smaller
% values of scale (L) mean that the gaussian decay relatively slower, such
% that the repeats will contribute more to the overall distribution. For
% large scales, repeats will be further out such that the gaussian will
% have decayed to a smaller value, so we can afford to add fewer repeats
% and still get the same accuracy
minDecay = 0.01; 
nReps = 5; %ceil(sqrt(-2*max(C(:))^2*log(minDecay))/max(L)); % from: alpha = mvnpdf(x)/mvnpdf(0), the distance at which the gaussian decays to some fixed percentage (minDecay)
[X,nReps] = repeat_in_hex(X,L,nReps,P);

end

% Get the value of the gaussian at these points
F = bvnpdf(X,M_Gauss,C);

% Add up the values of the repeating points
F = reshape(F,[size(F,1)/nReps,nReps]);
F = sum(F,2);

% For future reference: If we wanted to normalise we would have to multiply F by the area of one
% unit of the discretisation grid. In the case of the base grid sheet this
% would be par.plot.sc^2*sqrt(3)/2. For a normal square grid it would
% simply be par.plot.sc^2.

end