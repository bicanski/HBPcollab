function kfh_plot(varargin)

par = evalin('caller','par');
s = evalin('caller','s');

if ~isnumeric(varargin{1})

switch lower(varargin{1})
    % Plot estimate distribution
    case 'est'
        scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),1e7/par.grid.NcellSheet^2*ones(1,size(par.plot.grid_baseSheet,1)),gather(s.bys.p.P_est),'filled'); 
        axis equal tight off
    case 'sys'
        scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),1e7/par.grid.NcellSheet^2*ones(1,size(par.plot.grid_baseSheet,1)),gather(s.bys.p.P_sys),'filled'); 
        axis equal tight off
    case 'pos'
        scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),1e7/par.grid.NcellSheet^2*ones(1,size(par.plot.grid_baseSheet,1)),gather(s.bys.p.P_pos),'filled'); 
        axis equal tight off
end

else
    scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),1e7/par.grid.NcellSheet^2*ones(1,size(par.plot.grid_baseSheet,1)),gather(varargin{1}),'filled');
    axis equal tight off 
end

end
        