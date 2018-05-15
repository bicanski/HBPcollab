% Function to parse multi-simulation parameters into a structure cell
%
% simargs     : A structure containing parameters that don't change over iterations
% param_vec   : A cell containing vectors of parameters that need to be changed over iterations
% param_names : A list of parameters names (e.g. '.grid.scale')

function simargs = kf_parse_multi_input(param_vec,simargs,param_names)

% Meshgrid the parameters
param_mat = cell(size(param_vec));
[param_mat{:}] = ndgrid(param_vec{:});

% Create a cell array of simargs structures
simargs_ = cell(size(param_mat{1}));
for i = 1:numel(simargs_)
    simargs_{i}=simargs;
    
    for j = 1:length(param_names)
        evalc(['simargs_{',num2str(i),'}',param_names{j},'=param_mat{',num2str(j),'}(',num2str(i),')']);
    end
    
    % Add the multi-simualtion structure
    simargs_{i}.multi.param_vec   = param_vec;
    simargs_{i}.multi.param_names = param_names;
    
end
simargs = simargs_; clear simargs_

end