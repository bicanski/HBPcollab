
%% Update the grid cell firing

function stats = kf_update_grid_cells(s,par,stats)

%% Shortcuts
names = par.misc.names;

%% Intialises if not initialised
if ~isfield(stats,'firing_map')
    
    % Firing maps (we store one value for PI, apriori and measurement but
    % all values for the aposteriori estimate)
    for n=1:length(names)-1; stats.firing_map.(names{n})       = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1                    ); end
    stats.firing_map.aposteriori                               = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1,par.grid.NcellSheet);

    for n=1:length(names)-1; stats.firing_map_total.(names{n}) = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1                    ); end
    stats.firing_map_total.aposteriori                         = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1,par.grid.NcellSheet);

    % Rate maps
    for n=1:length(names)-1; stats.rate_map.(names{n})       = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1                    ); end
    stats.rate_map.aposteriori                               = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1,par.grid.NcellSheet);

    for n=1:length(names); stats.rate_map_total.(names{n}) = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1                    ); end
    stats.rate_map_total.aposteriori                       = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1,par.grid.NcellSheet);

    % Occupancy map
    stats.occupancy_map             = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1);
    stats.occupancy_map_total       = zeros(round(par.Y/par.plot.grid_res)+1,round(par.X/par.plot.grid_res)+1);    
    
end

%% Update

% Update the grid cell firing and occupancy map
stats = kf_update_maps(s,par,stats);

% Update the base grid firing pattern
if par.opts.drawfigs.CANsheet && par.opts.update.CANsheet
    stats.baseSheetActivity = s.bys.p.P_pos;
end

end