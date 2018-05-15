function [s,par,stats] = kf_update_time_dependent_parameters(par,s,stats)

% Weighting of measurement estimate
s.pc.measurement_weight = s.pc.measurement_weight;

% Change the measurement estimate mechanism after a period of learning
%if par.t==5e4
%    par.opts.sim.visual_input = 'pc';
%    par.opts.RefreshRate = 10;
%end

% Reset the readout maps after a period of learning
if mod(par.t-3,par.opts.ResetGap)==0
% stats = kfp_update_firing_rate_figs(stats,par);
    names = par.misc.names;
    stats.occupancy_map(:) = zeros(size(stats.occupancy_map));
    for n=1:length(names)
        stats.occupancy_map = zeros(size(stats.occupancy_map));
        stats.firing_map.(names{n}) = zeros(size(stats.firing_map.(names{n})));
        stats.rate_map.(names{n}) = zeros(size(stats.rate_map.(names{n})));
        %par.plot.ax{1}{n}.Children(2).CData = stats.firing_map.(names{n});
        %drawnow
    end
end

% Update time dependent parameters from structure



end
