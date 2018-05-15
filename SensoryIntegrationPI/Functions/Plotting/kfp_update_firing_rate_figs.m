%% Reset the firing rate plots
function stats = kfp_update_firing_rate_figs(stats,par)

names = par.misc.names;
stats.occupancy_map(:) = zeros(size(stats.occupancy_map));
for n=1:length(names)
    
    if size(stats.firing_map.(names{n}),3)~=1
        mon_cell = par.plot.mon_cell;
    else
        mon_cell = 1;
    end
    
    par.plot.ax{1}{n}.Children(2).CData = stats.summary.rate_map.(names{n})(:,:,mon_cell);
    drawnow
    
    stats.occupancy_map = zeros(size(stats.occupancy_map));
    stats.firing_map.(names{n}) = zeros(size(stats.firing_map.(names{n})));
    stats.rate_map.(names{n}) = zeros(size(stats.rate_map.(names{n})));

end

end