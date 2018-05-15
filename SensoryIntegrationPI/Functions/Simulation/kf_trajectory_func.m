%% Trajectory function

% x = [x,y,th] where x and y are the 2D position and th is the current
% heading direction

function [xnew,par] = kf_trajectory_func(xold,par)

switch lower(par.traj.type)
    case 'uniform_sample'
        xnew = [par.X*rand,par.Y*rand];
        xnew(3) = atan2d(xnew(2)-xold(2),xnew(1)-xold(1));
    case 'random_exploration'
        
        xnew = [Inf,Inf]; % Set the new coordinate at infinity so the function enters the while loop
        xold = xold(:)';
        ang_var = par.traj.ang_var; % Get the angular variation in heading direction
        
        xind = [1,1];
        
        while any([xnew(1)<0,xnew(1)>par.X,xnew(2)<0,xnew(2)>par.Y]) || (par.plot.env_mask(xind(2),round(xind(1)))==1)
            hd = mod(xold(3) + ang_var*randn,2*pi); % Calculate new random heading direction
            xnew(1:2) = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]; % Calculate a new trial coordinate
            ang_var = ang_var + par.learn.dt*0.1*par.traj.ang_var; % Increase the angular variance. If the solver can't find a position within the bounds, increase the angular variance to switch direction more suddenly
            xind = (floor(xnew/par.plot.grid_res))+1;
        end
        
        xnew(3) = hd; % Update current heading direction
        
    case 'edge'
        hd = round(xold(3)/(pi/2))*(pi/2); % Round t the nearest 90 degrees
        xnew = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]';
        if any([xnew(1)<0,xnew(1)>par.X,xnew(2)<0,xnew(2)>par.Y]) % If at the edge, turn 90 degrees and carry on
            hd = hd + 90;
            xnew = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]';
        end
        xnew(3) = hd;
        
    case 'alternate_edge_plus_middle'
        if ~isfield(par.traj,'alternate')
            par.traj.alternate = 1;
        end
        
        if mod(par.t,par.traj.alternate_time)==0
            par.traj.alternate = ~par.traj.alternate; % Swithc behaviour every par.alternate iterations
        end
        
        if par.traj.alternate
            % Run around the edges
            hd = round(xold(3)/(pi/2))*(pi/2); % Round t the nearest 90 degrees
            xnew = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]';
            while any([xnew(1)<0,xnew(1)>par.X,xnew(2)<0,xnew(2)>par.Y]) % If at the edge, turn 90 degrees and carry on
                hd = hd + 90;
                xnew = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]';
            end
            xnew(3) = hd;
        else
            % Random trajectory
            xnew = [Inf,Inf]; % Set the new coordinate at infinity so the function enters the while loop
            xold = xold(:)';
            ang_var = par.traj.ang_var; % Get the angular variation in heading direction
            
            while any([xnew(1)<0,xnew(1)>par.X,xnew(2)<0,xnew(2)>par.Y])
                hd = mod(xold(3) + ang_var*randn,2*pi); % Calculate new random heading direction
                xnew(1:2) = xold(1:2) + par.learn.dt*par.traj.vel*[cos(hd),sin(hd)]; % Calculate a new trial coordinate
                ang_var = ang_var + par.learn.dt*0.1*par.traj.ang_var; % Increase the angular variance. If the solver can't find a position within the bounds, increase the angular variance to switch direction more suddenly
            end
            
            xnew(3) = hd; % Update current heading direction
        end
    case 'spiral'
        [xnew(1:2),par] = kf_spiral(xold(1:2),par);
        xnew(3) = atan2d(xnew(2)-xold(2),xnew(1)-xold(1));
    case 'real_traj'
        if ~isfield(par.traj,'X')
            par.traj.X = get_sample_rat_trajectory(par,par.traj.real_ind);
            xnew = par.traj.X(1,:);
            xnew(3) = atan2d(xnew(2)-xold(2),xnew(1)-xold(1));
        else            
            xnew = par.traj.X(par.t,:);
            xnew(3) = atan2d(xnew(2)-xold(2),xnew(1)-xold(1));
        end
end
        
end