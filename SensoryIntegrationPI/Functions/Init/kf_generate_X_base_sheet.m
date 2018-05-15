function X = kf_generate_X_base_sheet(par)

if par.grid.Ncell>1

% The spacing between cells needs to be such that:
%
% GR*NCELL = SCALE
%par.plot.sc = 0.5*2/sqrt(3)*par.grid.scale/par.grid.Ncell; % Note, this is
%defined in kf_init instead

% Generate a grid by tiling the base coordinate
X = repeat_in_hex([0,0],par.plot.sc,par.grid.Ncell,par.grid.phi+30);

else
    
    X = [0,0];
    
end

% Get rid of three of the edges so that the points aren't repeating
% themselves
X = X+[1e-6,1e-6]; % Shift each coordinate by a small amount to make sure the edge points are wrapped on one side
X2 = wrap_xy_in_hex(X,par.grid.phi,par.grid.scale,par.grid.offset); % Wrap the coordinates
indsame = find(sqrt(sum((X2-X).^2,2))<par.grid.scale/2); % See which coordinates were changed in the wrapping process
X = X(indsame,:); % Only keep the original coordinates that were not wrapped, otherwise we will get repeats
X = X-[1e-6,1e-6]; % Shift the coordinates back

end