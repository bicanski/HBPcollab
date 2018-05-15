
%% Calculate some statistics about the simulation

function stats = kf_calculate_stats(s,par,stats)

%% Shortcuts
names = par.misc.names;
kf_vars = {'xPI','xd','z_wrap','x'};

%% Intialises if not initialised
if ~isfield(stats,'err')
    
    % Error measurements
    for n=1:length(names); stats.err.(names{n}) = 0; end
    
end

%% Update

% Calculate the mean error in the various estimates
for n=1:length(names); stats.err.(names{n}) = (stats.err.(names{n})*(par.t-1) + kf_hex_sheet_dist(s.(kf_vars{n}),s.X_wrap(:),par))/par.t; end

% Calculate the error between the grid readout and the visual input
stats.measurement_stim_err = gather(sum(abs(s.pc.gc_u(:) - s.bys.p.P_pos(:)))/par.grid.NcellSheet);

end