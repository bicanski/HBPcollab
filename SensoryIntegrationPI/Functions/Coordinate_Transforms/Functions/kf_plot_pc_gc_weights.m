% Plot the PC-GC weights for each place cell

function kf_plot_pc_gc_weights(par,W)

% Calculate the size of each PC plot such that none of them overlap
% each other
%xm=[]; ym=[]; for i=1:par.pc.npc; for j=i:par.pc.npc; if i~=j; xm=[xm,i]; ym=[ym,j]; end; end; end
%minDist = min(sqrt(sum(...
%    (par.pc.mu(xm(:),1)-par.pc.mu(ym(:),1)).^2 +...
%    (par.pc.mu(xm(:),2)-par.pc.mu(ym(:),2)).^2,2)...
%    ));

XM = par.pc.mu(:,1)-par.pc.mu(:,1)';
YM = par.pc.mu(:,2)-par.pc.mu(:,2)';
D = sqrt(XM.^2 + YM.^2);
D(1:size(D,1)+1:end)=Inf;
minDist = min(D(:));

% Define the minimum gap between PC plots as a proportion of the
% plot width
plotGap = 0.2;

% Scale the mu coordinates to space out the plots
beta = gather(par.grid.scale*(1+plotGap)/minDist);

coords = repmat(par.plot.grid_baseSheet,1,1,par.pc.npc);
coords = bsxfun(@plus,permute(coords,[3,2,1]),par.pc.mu*beta);
coords = permute(coords,[1,3,2]);
coords = gather(reshape(coords,[par.pc.npc*par.grid.NcellSheet,2]));

scatter(coords(:,1),coords(:,2),1e5/par.grid.NcellSheet^2,W(:),'filled')
axis equal off
set(gca,'YDir','reverse'); % Not sure why the y axis is the wrong way round but this is a quick fix

end