%% Update the grid cell firing map

% Calculate the hazard of firing at a particular location according to a
% given grid cell and the distribution in certainty at a particular location.
%
% Generate spikes according to a Poisson process and update the firing rate
% map

function stats = kf_update_maps(s,par,stats)

%% Shortcuts
names = par.misc.names;

% Get the index of the animals true location
indx = round(s.X(1)/par.plot.grid_res)+1;
indy = round(s.X(2)/par.plot.grid_res)+1;

%% Update the firing maps
for n=1:length(names)
    spk = real(s.bys.p.f.(names{n}))*par.grid.firing_rate; %poissrnd(real(s.bys.p.f.(names{n}))*par.grid.firing_rate); % Convert this to number of spikes
    stats.firing_map      .(names{n})(indy,indx,:) = squeeze(stats.firing_map.      (names{n})(indy,indx,:)) + spk; 
    stats.firing_map_total.(names{n})(indy,indx,:) = squeeze(stats.firing_map_total.(names{n})(indy,indx,:)) + spk;     
end

%% Update the occupancy map

% True location
stats.occupancy_map(indy,indx)           = stats.occupancy_map(indy,indx)+1;
stats.occupancy_map_total(indy,indx)     = stats.occupancy_map_total(indy,indx)+1;

%% Update the rate maps
for n=1:length(names)
    stats.rate_map.      (names{n})(indy,indx,:) = squeeze(stats.firing_map.      (names{n})(indy,indx,:)) / stats.occupancy_map(indy,indx); 
    stats.rate_map_total.(names{n})(indy,indx,:) = squeeze(stats.firing_map_total.(names{n})(indy,indx,:)) / stats.occupancy_map(indy,indx);
end

end