% Wrap x-y coordinates within a tiled hexagonal grid
%
% Talfan Evans (2017)
%
% Test:
%{
close all
M = 0*[0,0];
L = (1/3)*[1,1];
P = 0;
gr = 0.01;
xvec = 0:gr:(1-gr);
[xm,ym] = meshgrid(xvec);
X = [xm(:),ym(:)];

X = wrap_xy_in_hex(X,0,L,M);

F = mvnpdf(X,M,C);

figure
imagesc(xvec,xvec,reshape(F,size(xm))); shg; axis square
set(gcf,'color','w')
%}
%
% Inputs:
%
% X1    : [N,2] two-column matrix of x and y coordinates
% Phi   : Orientation in degrees
% Scale : [1,2] vector with the scales of both axes
function X2 = wrap_xy_in_hex(X1,Phi,Scale,M)

M = M(:)';

if size(X1,2)~=2
    if size(X1,1)==2
        X1 = X1';
    else
        error('Coordinates must be provided as a two column [x,y] matrix\n')
    end
end

Phi = Phi + 30;

X1 = bsxfun(@plus,X1,-M);      % The following operations assume that the base region is centered arond [0,0]
X1 = X1*2/sqrt(3)./Scale;      % Scale in the x and y dimensions

IJ1 = pixel_to_hex(X1,Phi);    % Convert cartesian pixels into hexagonal coordinates
UVW1 = hex_to_cube(IJ1);       % Convert hexagonal into cubic coordinates

rUVW = cube_round(UVW1);       % Get the centres of the nearest hexaons to each point
UVW2 = UVW1 - rUVW;            % Subtract the centre of the nearest hexagon (Voronoi region)
IJ2 = cube_to_hex(UVW2);       % Convert back to hexagonal coordinates
X2 = hex_to_pixel(IJ2,Phi);    % Convert back to cartesian pixels

X2 = X2*sqrt(3)/2*Scale;       % Reverse the scaling operation

end

% Plotting (place inside the function)
%{

subplot(2,3,1)
scatter(X1(:,1),X1(:,2),'.')
axis equal
%title('Pixels')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])

subplot(2,3,2)
scatter(IJ1(:,1),IJ1(:,2),'.')
axis equal
%title('Hex Coords')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])

subplot(2,3,3)
scatter3(UVW1(:,1),UVW1(:,2),UVW1(:,3),'.')
axis equal
set(gca,'CameraPosition',[5.4744    5.4720    7.7444])
%title('Cubic Coords')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

subplot(2,3,4)
scatter3(UVW2(:,1),UVW2(:,2),UVW2(:,3),'.')
axis equal
view(135,45)
%title('Wrap in Cubic Coords')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

subplot(2,3,5)
scatter(IJ2(:,1),IJ2(:,2),'.')
axis equal
%title('Wrapped Hex Coords')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])
set(gca,'ZTickLabel',[])

subplot(2,3,6)
scatter(X2(:,1),X2(:,2),'.')
axis equal
%title('Wrapped Pixels')
set(gca,'XTickLabel',[])
set(gca,'YTickLabel',[])


set(gcf,'color','w')
%}