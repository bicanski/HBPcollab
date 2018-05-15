function rUVW = cube_round(UVW)

% Rount to nearest integer of each coordinate
rUVW = round(UVW);

% Get the distace from the original points to the nearest rounded centre
du = abs(rUVW(:,1)-UVW(:,1));
dv = abs(rUVW(:,2)-UVW(:,2));
dw = abs(rUVW(:,3)-UVW(:,3));

% Correct for the regions where we actually rounded to a rhombus where we
% should have rounded to a hexagon
ind1 = (du>dv) & (du>dw);
rUVW(ind1,1) = -rUVW(ind1,2) -rUVW(ind1,3);

ind2 = (dv>dw);
rUVW(ind2,2) = -rUVW(ind2,1) -rUVW(ind2,3);

ind3 = ~(ind1 | ind2);
rUVW(ind3,3)  = -rUVW(ind3,1)  -rUVW(ind3,2);

end     