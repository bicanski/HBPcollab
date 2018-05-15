function [X,nreps] = repeat_in_hex(X,L,N,P)

% Note: This generates a lattice of repeated points which looks like a
% rhombus, not a hexagon. But, we then exclude the points where the product
% of the coordinates is greater than N, the number of repeats. This cuts
% off the corners of the rhombus and produces a hexagon.

% All integer combinations of the two coordinates within N
[im,jm] = meshgrid(-N:N);

% Cut off the corners of the rhombus to produce a hexagon
zm = abs(im(:)+jm(:));
im = im(zm<=N);
jm = jm(zm<=N);

% Number of repeats of each point (including the centre point)
nreps = length(im(:));
 
% Add the coordinate shifts to the original points
X = permute(repmat(X,[1,1,nreps]),[1,3,2]); % [Num. coordinates, repeats for each coordinate,x and y] 
  
T1 = P;
T2 = T1 + 60;

X = X + L(1).*...
        permute(repmat(...
        [im(:).*cosd(T1) + jm(:).*cosd(T2),...
         im(:).*sind(T1) + jm(:).*sind(T2)],...
        [1,1,size(X,1)]),[3,1,2]);
     
% Reshape to a two column vector of xs and ys
X = reshape(X,nreps*size(X,1),2);

end
