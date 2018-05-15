function W = kf_initialise_recurrent_weights(par)

% Initialise the weights
switch lower(par.opts.pc_remapping.rec_conn_type)
    
    % Uniformly distributed weights 
    case 'uniform'
        
        W = par.opts.pc_remapping.mean_strength/0.5 * rand(par.pc.npc,par.pc.npc); % Normalise by 0,5 since that's the mean of the uniform distribution
        
    % Lognormal distribution of weights
    % See Mizuseki and Buzsaki, 2014 (Nature Neuro), Fig. 5 and
    % accompanying text for references
    case 'lognormal'
        
    % Should the connection probability be distance depdendent?
    case 'distance_dependent'
        
    % Some other connectivity scheme...
    case 'some other connectivity scheme...'
        
end

% Randomly set some fraction of the connection to zero
W(randperm(numel(W),numel(W)-round(par.opts.pc_remapping.rec_conn_frac*numel(W)))) = 0; 

end