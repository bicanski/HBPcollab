% Record summary stats in matrix rather than cell format for easy access later
function stats = kf_record_summary_stats(stats,par,S)

% Shortcuts etc.
sm = stats.summary;
names = par.misc.names;

%% Initialise
if ~isfield(stats,'stats_rec')
    stats.stats_rec=[];
    % Gridness, scale, orientation
    FN = {'gridness','scale','orientation'}; % Fields that we want to record
    for f=1:length(FN)
        for n=1:length(names)
            for i=1:length(par.plot.env_subareas)
                stats.stats_rec.(FN{f}){i}.(names{n})= [];
            end
        end
    end
    
    % Rate maps and ACs
    FN = fieldnames(stats.summary.rate_map);
    for fn=1:length(FN)
        for i=1:length(par.plot.env_subareas)
            stats.stats_rec.rate_map.(FN{fn}){i} = [];
        end
    end
    
    % PC->GC weights
    stats.stats_rec.weights = [];
    
end


%% Update

% Gridness, scale, orientation
FN = {'gridness','scale','orientation'}; % Fields that we want to record
for f=1:length(FN)
    for n=1:length(names)
        for i=1:length(par.plot.env_subareas)
            stats.stats_rec.(FN{f}){i}.(names{n})= [stats.stats_rec.(FN{f}){i}.(names{n}),sm.(names{n}){i}.(FN{f})(:)];
        end
    end
end

% Rate maps and autocorrelograms
FN = fieldnames(stats.stats_rec.rate_map);
for fn=1:length(FN)
    stats.stats_rec.rate_map.(FN{fn}) = horzcat(stats.stats_rec.rate_map.(FN{fn}),stats.summary.rate_map.(FN{fn})(:));
end

% Weights
%stats.stats_rec.weights = horzcat(stats.stats_rec.weights,S.pc.w(:));

end