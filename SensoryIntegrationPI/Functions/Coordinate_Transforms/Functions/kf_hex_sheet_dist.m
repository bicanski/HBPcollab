% Calculate distance between two points in hexagonal space
%
% Tile one of the points to the adjacent hexagon cells. Compute the distance
% between each pair and take the minimum
function D = kf_hex_sheet_dist(x1,x2,par)

x1 = repmat(x1(:)',7,1) + par.grid.scale*[0,0;cosd(0:60:300)',sind(0:60:300)'];

D = min(sqrt(sum((bsxfun(@plus,x1,-x2(:)')).^2,2)));

end