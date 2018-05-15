function par = kf_bvc_param_file(par)

%% Simulation (overwrite the ones from the KF params file)
par.T = 1e3;

%% Environment
par.env.X = 1;                                                             % Size X
par.env.Y = 1;                                                             % Size Y

par.env.grid_res = 0.01;                                                   % Resolution of environment for BVCs
par.env.type = 'walls';                                                    % Type of environment

%% BVCs
par.bvc.nCopies = 10;                                                      % Number of copies of each cell at angle phi
par.bvc.phivec = [0,pi/2,pi,3*pi/2];                                       % List of angles
par.bvc.tuning_distance_min_max = [0,par.env.X/2];                         % Tuning distance
par.bvc.model =  'bvc_full';                                               % Type of BVC response
par.bvc.strength = 1;                                                      % Strength of BVC input

%% PCs
par.pc.nPCs = 100;                                                         % Number of PCs
par.pc

%% Learning
par.bvcpc.learning_rate = 0.2;                                             % Learning rate
par.bvcpc.learning_rule = 'bcm';                                           % Learning rule
par.bvcpc.threshold_time_constant = 10*par.bvcpc.learning_rate;            % Note this should be faster than the learning rate
par.bvcpc.thresh = 0;

%% Initialisation
par.var.x0 = [0.5,0.5];                                                    % Initial position
par.var.hd0 = 0;                                                           % Initial head direction

par.bvc.phi = repmat(par.bvc.phivec,1,par.bvc.nCopies);                    % Make n copies of each phi value
par.bvc.nBVC = length(par.bvc.phi);                                        % Total number of BVCs
par.bvc.tuning_distance = par.bvc.tuning_distance_min_max(1) + ...
                            rand(1,par.bvc.nBVC)*range(par.bvc.tuning_distance_min_max);
end