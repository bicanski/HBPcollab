%% Lattice of hex coordinates

nvec = 0:5;

close all

for i = 1:length(nvec)
    
    n = nvec(i);
    
    % New points
    [im,jm] = meshgrid(-n:n);
    zm = abs(im(:)+jm(:));
    im = im(zm<=n);
    jm = jm(zm<=n);
    x = [im(:),jm(:)];
    
    % Old points
    [imo,jmo] = meshgrid(-(n-1):(n-1));
    zmo = abs(imo(:)+jmo(:));
    imo = imo(zmo<=(n-1));
    jmo = jmo(zmo<=(n-1));
    xo = [imo(:),jmo(:)];
    
    % Dicard the old points
    if n>0
        x = setdiff(x,xo,'rows');
    end
    
    im = x(:,1); jm = x(:,2);
    
    X = [im(:)*cosd(0) + jm(:)*cosd(60),...
        jm(:)*sind(0) + jm(:)*sind(60)];
    
    hold on
    scatter(X(:,1),X(:,2),500,'filled','MarkerFaceAlpha',1-n/length(nvec)); axis square
    
    for j = 1:size(X,1)
        text(X(j,1),X(j,2),num2str(n),'HorizontalAlignment','center','VerticalAlignment','middle');
    end
    
end

axis off
set(gcf,'color','w')

%legstr={}; for i=1:length(nvec); legstr{i}=num2str(nvec(i)); end; legend(legstr)

export_fig hexagonal_lattice_figure.fig
export_fig hexagonal_lattice_figure.png -m2.5