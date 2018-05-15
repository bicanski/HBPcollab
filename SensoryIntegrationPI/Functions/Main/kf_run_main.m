function [stats,S_record,par] = kf_run_main(par,s)

%% Loop storage preallocation 
S_record = {};                                                             % Array to hold the state vector
x = zeros(3,par.T); x(:,1) = [s.x;0];                                      % Array to hold the trajectory
z = zeros(2,par.T); z(:,1) = s.x(1:2) + sqrt(diag(s.Q)).*randn;            % Array to hold the measurement

S_record{1} = s;                                                           % Initialise first cell
H = [];                                                                    % Empty plot
stats = [];                                                                % Initialise stats array

%% Run in a loop
for t=2:par.T
    
    % Keep a track of the current iteration to pass to subfunctions
    par.t = t;     
    clc; fprintf('t=%i/%i\n',par.t,par.T)
    %pause(0.1); % Slow down the simulation if desired
    
    %% Update paramaters that are time-dependent
    [s,par,stats] = kf_update_time_dependent_parameters(par,s,stats);
     
    %% Update the states
    
    % Update the true position for the current time step
    [x(:,t),par] = kf_trajectory_func(x(:,t-1),par);     
    s.dx = sqrt((x(1,t)-x(1,t-1)).^2 + (x(2,t)-x(2,t-1)).^2);
    s.X = x(1:2,t); 
    s.X_wrap = wrap_xy_in_hex(s.X(:)',par.grid.phi,par.grid.scale,par.grid.offset);
    
    % Update the control input 
    % The process noise is used to account for the noise associated with
    % the PI input. We need to add this here to simulate the uncertainty
    s.Q = kf_update_process_noise(s,par);  % Update the process noise at time t
    
    % Modify the noise according to the speed of the animal (faster speed = more noise)
    vel = x(1:2,t) - x(1:2,t-1);
    s.absVel = sqrt(sum(vel.^2));
    if ~par.opts.perfect_tracking
        noise = sqrt(diag(s.Q)).*s.absVel.*randn(2,1);                     % Multiplication term is SD not VAR (VAR = SD^2)
        s.u = vel+noise;                                                   % The change in position at time t relative to t-1
    else
        s.u = x(1:2,t)-x(1:2,t-1);
    end
    
    % Take a noisy measurement (noise value would change depending on reliability of sensory input)
    s.R = kf_update_measurement_noise(s,par); % Update the measurement noise
    
    % !!!!!!!! The PCs should be receiving a noisy estimate of position?
    % I.e. each x input to each PC should be slightly different? %%%%%%%
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! %
    if ~par.opts.perfect_tracking
        if par.opts.independent_sensory_measurements
            s.z = x(1:2,t) + sqrt(...
                [s.R(1:4:numel(par.pc.C));...
                 s.R(4:4:numel(par.pc.C))])...
                .*randn(2,par.pc.npc);
        else
            s.z = z(1:2,t) + sqrt(diag(s.R(:,:,1))).*randn(2,1);
        end
    else
        s.z = x(1:2,t);
    end
    
    % Compute the firing of the sensory inputs (PCs/BVCs) and get a
    % measurement estimate
    if par.opts.update.pc_firing
        s.pc.F = kf_compute_pc_firing(s,par);
    end
    
    if strcmpi(par.opts.sim.visual_input,'pc')
        s = kf_get_measurement_from_sensory(par,s);
    end
    
    % Update step
    
    % a. Movement update
    [s,par] = kf_do_movement_update(par,s);

    % Update the Sensory - GC connections AFTER
    if strcmpi(par.learn.learning_order,'inbetween')
        s = kf_update_sensory_gc_connections(stats,par,s);
    end    
    
    % b. Measurement update
    [s,par] = kf_do_measurement_update(par,s);
    
    % Old version...
    %[s,par] = kf_update_state_bayesian_periodic(par,s);
    
    % Update the Sensory - GC connections AFTER
    if strcmpi(par.learn.learning_order,'after')
        s = kf_update_sensory_gc_connections(stats,par,s);
    end
    
    %% If doing replay, do a few new things
    if par.opts.replay.option
        
        % Update the PC-PC connections
        s = kf_update_P2P_connections(s,par);
        
        % Monitor prediction error
        s.pred_err = kf_compute_prediction_error(s.bys.p.P_est,s.bys.p.P_apriori);
        s.spwr     = kf_replay_condition(s.pred_err,par);
        
        if any(par.t==par.opts.replay.spwr_time)
            s.spwr=1; 
        else
            s.spwr=0; 
        end
        
        % Do replay update if appropriate
        if s.spwr
            s = kf_do_replay_realspace(s,par);
        end
        
        % Record the inferred pairwise PC-PC weights to compare with the
        % real case
        if mod(par.t-2,par.opts.replay.pw_dist_rec_gap)==0 %|| some_condition
            stats = kf_record_pairwise_error(s,par,stats);
        end
        
        % Record replay related stats
        stats = kf_record_replay_stats(stats,s,par);
        
    end
    
    %% Output and plotting
    
    % Update the readouts from the grid cells
    stats = kf_update_grid_cells(s,par,stats);
    
    % Calculate some simulation statistics at each iteration
    stats = kf_calculate_stats(s,par,stats);       
    
    % Calculate summary statistics every StatsGap iterations
    % Lighter stuff, e.g. gridness, orientations etc.
    if (mod(par.t,par.opts.StatsGap)==2)
        stats = kf_calculate_summary_stats(s,par,stats);
        S_record{end+1} = s; %#ok<AGROW>
    end
    
    % Record some stuff every SummaryGap iterations
    % e.g weights, firing maps etc.
    if (mod(par.t,par.opts.SummaryGap)==2)
        stats = kf_record_summary_stats(stats,par,s);
    end
    
    % Plot the current state and calculate summary statistics
    if ((mod(par.t,par.opts.RefreshRate)==2) && par.opts.plot) || (par.opts.RefreshRate==1) || (par.t==par.T)
        [H,par] = kf_plot_state(H,s,par,stats);
    end
                      
end

% Output
stats = kf_calculate_summary_stats(s,par,stats);
S_record{end+1} = s;

% Close figures
for ax=1:length(par.plot.ax)
    h = par.plot.ax{ax}{1}.Parent;
    close(h)
end

end
    