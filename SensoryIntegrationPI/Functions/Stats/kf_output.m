%% Output the results of the simulation to file
function OUT = kf_output(par,stats,S_record)

%% Make a new directory for this project
sd = seshdate();
while (exist(['Data/',par.session_name,'/',sd,'/'],'dir')==7)
    sd = seshdate();
end
dirName = ['Data/',par.session_name,'/',sd,'/'];
    
if par.opts.save_stats || par.opts.save_plots
    mkdir(dirName)
end

%% Stats
OUT.kf_vars = S_record;
OUT.stats.record = stats.summary;
OUT.stats.summary = stats.stats_rec;
OUT.stats.final = rmfield(stats,{'summary','stats_rec'});

if par.opts.save_final_weights
    OUT.stats.final.weights = S_record{end}.pc.w;
end

OUT.par = par;
if par.opts.save_stats
    save([dirName,'OUT.mat'],'OUT')
end

%% Plots
if par.opts.save_plots
    saveallfigs('dir',dirName)
end

end