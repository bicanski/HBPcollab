%% Calculate summary stats every RefreshRate iterations

function stats = kf_calculate_summary_stats(S,par,stats)

% List of field names make writing the scripts neater
names = par.misc.names;
kf_vars = {'xPI','xd','z_wrap','x'};

%% Intialises if not initialised
if ~isfield(stats,'summary')
    for n=1:length(names)
        stats.summary.err.(names{n}) = [];
        for i=1:length(par.plot.env_subareas)
            stats.summary.gridness{i}.(names{n}) = [];
            stats.summary.scale{i}.(names{n}) = [];
            stats.summary.orientation{i}.(names{n}) = [];
        end
    end
end

%% Update the grid cell firing and occupancy map
for n=1:length(names)
    for n2=1:size(stats.rate_map.(names{n}),3)
        stats.summary.rate_map.(names{n})(:,:,n2) = filter2(ones(par.opts.rateMapSmoothing),stats.rate_map.(names{n})(:,:,n2));
    end
    AC = zeros((size(stats.summary.rate_map.(names{n}),1)-1)*2+1,...
               (size(stats.summary.rate_map.(names{n}),2)-1)*2+1,...
                size(stats.summary.rate_map.(names{n}),3));

    % Calculate the autocorrelograms separately for each subarea of the
    % environent (e.g. if we have multiple compartments)
    x={}; y={}; xAC={}; yAC={};
    for i = 1:length(par.plot.env_subareas)
        x{i} = par.plot.env_subareas{i}.x; xAC{i} = sort([2*(x{i}-1)+1,2*(x{i}-1)+2]); xAC{i} = xAC{i}(1:end-1);
        y{i} = par.plot.env_subareas{i}.y; yAC{i} = sort([2*(y{i}-1)+1,2*(y{i}-1)+2]); yAC{i} = yAC{i}(1:end-1);
        for n2=1:size(stats.summary.rate_map.(names{n}),3)
            AC(yAC{i},xAC{i},n2) = real(xPearson(stats.summary.rate_map.(names{n})(y{i},x{i},n2)));
        end
    end
    stats.summary.rate_map.([names{n},'_AC']) = AC;
    %stats.summary.rate_map.([names{n},'_AC']) = real(xPearson(stats.summary.rate_map.(names{n})));
end

%% Update stats on the grid readout
%
% Note that we don't calculate stats for all the grid cells here, this
% would slow down simulations too much
%
for n=1:length(names)
    for i=1:length(par.plot.env_subareas)
        if strcmpi(names{n},'aposteriori'); MON_CELL=par.plot.mon_cell; else; MON_CELL=1; end
        str.sac = stats.summary.rate_map.([names{n},'_AC'])(yAC{i},xAC{i},MON_CELL);
        tmp = autoCorrProps(str);
        if isnan(tmp.gridness); tmp.gridness=-1; end
        stats.summary.(names{n}){i} = tmp;
    end
end

%% Record the error, gridness, orientation etc.
for k=1:length(kf_vars) 
    stats.summary.err.(names{k})(end+1) = stats.err.(names{k});
    for i=1:length(par.plot.env_subareas)
        stats.summary.gridness   {i}.(names{k})(end+1) = stats.summary.(names{k}){i}.gridness;
        stats.summary.scale      {i}.(names{k})(end+1) = stats.summary.(names{k}){i}.scale;
        stats.summary.orientation{i}.(names{k})(end+1) = stats.summary.(names{k}){i}.orientation;
    end
end

end