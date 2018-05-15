% Process output files into a manageable format

function STATS = kf_process_output_files(data_dir,varargin)

clc; disp('Initializing...')

% Defaults
mon_cell = 1;

parpool_opt = 1;

opt_fields = {'ac','rm','gr','or','or_amin','sc','sc_mean','ell'};
sections = 'last';

verbose = 0;

% Flags (don't modify)
fields_flag = 0; % If user specifies particular fields, empty the fields list and only record the fields specified

% Process input arguments
arg=1;
while arg<=length(varargin)
    switch lower(varargin{arrg})
        case 'mon_cell'
            mon_cell = varargin{arg+1};
            arg=arg+1;
        case opt_fields
            if fields_flag==0
                opt_fields={};
                fields_flag=1;
            end
            opt_fields{end+1} = varargin{arg};
        case {'sections','sects'}
            sectionlist = varargin{arg+1};
            arg=arg+1;
        case {'parpool','parpool_opt'}
            parpool_opt=varargin{arg+1};
            arg=arg+1;
        case 'verbose'
            verbose = varargin{arg+1};
            arg=arg+1;
    end
    arg=arg+1;
end

% Find the data

sim_struct = load([data_dir,'sim_struct.mat']); sim_struct=sim_struct.sim_struct; % Get the multi simulation structure

D = dir(data_dir); D=D(3:end-1); D = D([D.isdir]); % Get list of subfolders

%D = D(2:4);

% Preallocate
load([data_dir,D(1).name,'/OUT.mat']); clc % Load the first instance to get some info on length etc

% The optional fields to be collected 
for f=1:length(opt_fields)
    OUT_.(opt_fields{f}) = cell(length(sections),length(D));
end

% The fields defined in the simulation structure
sim_fields = {};
for f=1:length(sim_struct.param_names)
    tmp = strsplit(sim_struct.param_names{f},'.');
    sim_fields{f} = ['par_',num2str(f)];
    OUT_.(sim_fields{f}) = cell(1,length(D));
end

% If recording orientation or scale, also record amin and mean scale
if isfield(OUT_,'or'); OUT_.or_amin=cell(length(sections),length(D)); end
if isfield(OUT_,'sc'); OUT_.sc_mean=cell(length(sections),length(D)); end

% Store the datafile temporarily
data = cell(1,length(D));

% Loop through and collect the data
for d=1:length(D)
    
    % Load the date
    data{d} = load([data_dir,D(d).name,'/OUT.mat']); data{d}=data{d}.OUT; clc
    
    nsection = length(data{d}.stats.summary.rate_map.aposteriori_AC);
    if strcmpi(sections,'all'); sectionlist = 1:nsection; elseif strcmpi(sections,'last'); sectionlist=nsection; elseif length(sections)>nsection; error('Specified number of sections is greater than what exists.\n'); end
    
    for s=1:length(sectionlist)
        sect = sectionlist(s);
        
        % Progress
        clc; fprintf('Processing individual files...\nd=%i/%i\ns=%i/%i\n',d,length(D),s,length(sectionlist))
        
        % Load ac and calculate stat        
        ac_tmp{s,d}     = reshape(data{d}.stats.summary.rate_map.aposteriori_AC{sect},[2*data{d}.par.plot.gridsize-1,data{d}.par.grid.NcellSheet]);
        ac_tmp{s,d}     = ac_tmp{s,d}(:,:,mon_cell);
        in    {s,d}.sac = ac_tmp{s,d}; in{s,d}.VERBOSE = verbose;
        stats {s,d}     = autoCorrProps(in{s,d});
        
        for of = 1:length(opt_fields)
                       
            % Optional fields
            switch lower(opt_fields{of})
                case 'ac'
                    OUT_.ac{s,d} = ac_tmp{s,d};
                    OUT_.ac{s,d} = OUT_.ac{s,d}(:,:,mon_cell);
                case 'rm'
                    OUT_.rm{s,d} = reshape(data{d}.stats.summary.rate_map.aposteriori   {sect},[  data{d}.par.plot.gridsize  ,data{d}.par.grid.NcellSheet]);
                    OUT_.rm{s,d} = OUT_.rm{s,d}(:,:,mon_cell);
                case 'sc'
                    OUT_.sc{s,d}      = stats{s,d}.scale_all(:);
                    OUT_.sc_mean{s,d} = mean(OUT_.sc{s,d});
                case 'or'
                    OUT_.or     {s,d} = stats{s,d}.all_orientation(:);
                    OUT_.or_amin{s,d} = amin(OUT_.or{s,d}); 
                case 'gr'
                    OUT_.gr{s,d} = stats{s,d}.gridness;
                case 'ell'
                    OUT_.ell{s,d} = stats{s,d}.ellipticity;
            end
        end
        
        % Simulation parameters
        for sf = 1:length(sim_fields)
            eval(sprintf('OUT_.%s{%i,%i} = data{%i,%i}.par%s;',sim_fields{sf},s,d,s,d,sim_struct.param_names{sf}));
        end
        
        % Clear temporary variables
        in{s,d}     = [];
        %data{s,d}    = [];
        stats{s,d}  = [];
        ac_tmp{s,d} = [];

    end
end

% Re-order the data
disp('Re-ordering the data...')

% Get the size of the output structure
pv = sim_struct .param_vec;
pn = sim_struct.param_names;

outsize = []; 
for s=1:length(pv)
    outsize(end+1) = length(pv{s});
end

% Make another output structure
FN = fieldnames(OUT_);
for fn=1:length(FN)
    STATS.(FN{fn}) = zeros([outsize,size(OUT_.(FN{fn}){1})]);
end

% Loop through the stored data
for d=1:length(D)
    for s=1:length(sectionlist)
        
        % Get indices
        inds = [];
        for p=1:length(pv)
            inds(end+1) = find(pv{p}==OUT_.(['par_',num2str(p)]){s,d});
        end
        
        % Loop over fields
        indstr=[]; for i=1:length(inds); indstr=[indstr,num2str(inds(i)),',']; end; indstr=[indstr,':,:,:,:,:'];
        for fn=1:length(FN)
            eval(sprintf('STATS.(FN{%i})(%s) = OUT_.(FN{%i}){%i,%i};',fn,indstr,fn,s,d))
            
            OUT_.(FN{fn}){s,d} = [];
        end
        
    end
        
end

% Store parameter names that were varied during the trial
STATS.sim_struct = sim_struct;

disp('Done!')

end