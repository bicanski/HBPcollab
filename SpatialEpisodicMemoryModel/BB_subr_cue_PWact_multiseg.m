

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% REF
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


function [Grid_Act] = BB_subr_cue_PWact_multiseg(res,env)


% Note that this function actually calculates something proportional
% to BVC FIRING RATE when an bndry is present. I use it to generate
% cuing current as well.

% Get grid parameters

maxR         = 16;        % BVCs to cover space up to 16 units from origin
polarDistRes = 1;         % old naming convention, its not a resolution anymore 
                          % becasue radial separation increase between receptive 
                          % fields along a radius, but the math here doesn't change 
polarAngRes  = 2*pi/51;   % Angular separation of neurons on radial grid

% Assign polar coordinates to each Grid (BVC/PW or whatever) neuron. Numbering moves outward along radius, starting at 0 radians
NGridR       = round((maxR)/polarDistRes);   % 16 grid points along a radius, just not uniformly spaced 
NGridTheta   = floor((2*pi-0.01)/polarAngRes)+1;
NGrid        = NGridR*NGridTheta;                                           % Num. grid neurons
%polarDist    = [polarDistRes/2:polarDistRes:(NGridR-0.5)*polarDistRes];    % linear scaling used in the BBB model
[polarDist]  = BB_subr_radscaling(maxR);

polarAng     = [0:polarAngRes:(NGridTheta-1)*polarAngRes];
[pDist,pAng] = meshgrid(polarDist,polarAng);
GridDist     = reshape(pDist',prod(size(pDist)),1);                         % Vector of distances for each neuron from origin
GridAng      = reshape(pAng',prod(size(pAng)),1);                           % Vector of corresponding angles
GridAng      = GridAng - 2*pi*(GridAng > pi);                               % Fix range

% Create cartesian gridpoints in window containing region covered by Grid neurons.
minX  = - maxR;
maxX  = maxR;
minY  = - maxR;
maxY  = maxR;
Nx    = round((maxX-minX)/res);
Ny    = round((maxY-minY)/res);
x     = [minX+res/2:res:minX+(Nx-0.5)*res];
y     = [minY+res/2:res:minY+(Ny-0.5)*res];
[X,Y] = meshgrid(x,y);

Grid_Act = zeros(NGrid,1);

% Env will contain vector of start and endpoints for visible boundary segments. Must find cartesian coords of each bndrypoint.
% env is actually egocues, which we calculated for the perceptual drive

for bndryNum = 1:size(env,1)     % env can contain just one line segment or many
 
    xi  = env(bndryNum,1);
	xf  = env(bndryNum,3);
	yi  = env(bndryNum,2);
	yf  = env(bndryNum,4);
	den = sqrt((xf-xi)^2+(yf-yi)^2);
	nx  = (xf-xi)/den;
	ny  = (yf-yi)/den;
	
    % Equation of line is given by x(t) = xi + t(xf-xi); y = yi + t(yf-yi). t is taken from [0,1]. 
    % used alpha in some other routines here for the same purpose
	% First find perp(endicular) disp(lacement) from grid points to the entire line.
	
    PerpDispFromGrdPtsX = -(X-xi)*(1-nx^2) + (Y-yi)*ny*nx;
	PerpDispFromGrdPtsY = -(Y-yi)*(1-ny^2) + (X-xi)*nx*ny;
	
    % Calculate t-value of the line point which is perp from each grid point
	if xf ~= xi
		t = (X+PerpDispFromGrdPtsX-xi)/(xf-xi);
	else
		t = (Y+PerpDispFromGrdPtsY-yi)/(yf-yi);
    end
	
    % Eliminate all grid points without a perp point, and farther away than res/2.
	BndryPts = (t>=0)&(t<=1)&...
        (PerpDispFromGrdPtsX>=-res/2)&(PerpDispFromGrdPtsX<res/2)&...
        (PerpDispFromGrdPtsY>=-res/2)&(PerpDispFromGrdPtsY<res/2);
    (PerpDispFromGrdPtsX.^2 + PerpDispFromGrdPtsY.^2 <= (res/2)^2);
    
    XBndryPts = X(find(BndryPts==1));
    YBndryPts = Y(find(BndryPts==1));
    
    % Covert bndrypts to polar coordinates
    [ThetaBndryPts,RBndryPts] = cart2pol(XBndryPts,YBndryPts);

    for bndryNum = 1:size(ThetaBndryPts,1)                                    % Add contribution from each bndry point
        AngDiff1 = abs(GridAng-ThetaBndryPts(bndryNum));
        AngDiff2 = 2*pi-abs(-GridAng+ThetaBndryPts(bndryNum));
        AngDiff  = (AngDiff1<pi).*AngDiff1 + (AngDiff1>pi).*AngDiff2;
        sigmaTH  = sqrt(0.05);
        sigmaR0  = 0.08;
        sigmaR   = (RBndryPts(bndryNum)+8)*sigmaR0; % has to increase with radius for increasing receptive field size with distance
        Grid_Act = Grid_Act + 1/RBndryPts(bndryNum) * ( exp(-((AngDiff)/sigmaTH).^2 )  .*  exp( - ( (GridDist-RBndryPts(bndryNum))./sigmaR ) .^2 )  );
    end
    
end

maximum = max(Grid_Act);

if maximum > 0.0
	Grid_Act = Grid_Act/maximum;    % Normalize
end


