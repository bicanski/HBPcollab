% Generate hexagonal grids from a square grid

function [XH,XS,M,HI] = kf_generate_X_base_sheet_from_square(par)

% Generate square lattice
N = par.grid.Ncell;

LX = 2*N;
LY = 2*N;

xv = 0:(LX-1);
yv = 0:(LY-1);
[xm,ym] = meshgrid(xv,yv);
X = [xm(:),ym(:)];

X = X - N;
X = X  / sqrt(3) * par.grid.scale / abs(min(X(:,1)));

L = xm+ym;

ld = (2*N-1)-N; % Upper and lower diagonals
ud = (2*N-1)+N;
ind = find(L<=ud & L>ld);

XS = X(ind,:);

% Transform
T = [cosd(0 ),sind(0 );...
     cosd(60),sind(60)];
 
XH = XS*T;

% Rotate
ph = par.grid.phi -30;
R = [cosd(ph),-sind(ph);sind(ph),cosd(ph)];

XH = XH*R;

% Output square mask and the indices corresponding to the hexagonal grid
HI = ind;
M = zeros(size(L));

end