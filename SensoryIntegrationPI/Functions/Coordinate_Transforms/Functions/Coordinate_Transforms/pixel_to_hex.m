function IJ = pixel_to_hex(X,Phi)

% Generate the firing rate of each of the cells for each of the positions
% we're trying to decode, contained in x. The mod function accounts for the
% periodic nature of the grid cells.
% Define the angles of the two new coordinates axes [i,j]
T_i = 0;  % Relative orientation of the
T_j = 120; % two new coordinate axes

T_i = T_i + Phi;
T_j = T_j + Phi;

% Convert into IJ space
IJ = [cosd(T_i)*X(:,1) + sind(T_i)*X(:,2),... % At this point x and y are scalars
      cosd(T_j)*X(:,1) + sind(T_j)*X(:,2)];

end