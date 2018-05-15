% Make the movement distribution mask, which we get from tiling the
% cartesian coordinate version of the hexagonal coordinates

function [M,XS,IND] = kf_tile_movement_cart_distro(MSK,HI,XS,F,par)

SZ = size(MSK);

[xi,yi] = ind2sub(SZ,HI);
X = [xi(:),yi(:)];

DX = [1, 1;2,-1;1,-2;-1,-1;-2,1;-1,2]; % List of translations for each repeat
L = SZ(1)/2;
DX = DX * L;

M = zeros(6*L); % Create a blank mask
X = bsxfun(@plus,X,L*[2,2]); % Shift the base coordinates to the centre
ind = sub2ind(size(M),X(:,1),X(:,2));
c=1;
M(ind) = F;
%imagesc(M); axis equal; set(gca,'YDir','normal')

xs = XS;
IND = ind(:);
for i = 1:length(DX)
    XS = vertcat(XS,bsxfun(@plus,xs,DX(i,:) /L * par.grid.scale/2*2/sqrt(3)));
    x = bsxfun(@plus,X,DX(i,:));
    ind = sub2ind(size(M),x(:,1),x(:,2));
    IND = [IND;ind(:)];
    M(ind) = F;
    %imagesc(M); axis equal; set(gca,'YDir','normal')
end


end