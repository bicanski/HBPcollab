

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% REF
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


% this function updates the linear translational movement. Rotation
% is resolved in the agent model script.
% The function needs to take care not to run into boundaries
% We use a list of targets to be visited in the arena. By arranging
% the order of the targets appropriately we avoid running into boundaries
% and hence any complicated calculations of geometry.


function [Xnew,Ynew,ang2f_tot,toturn,dist2f_cur,CWturn,CCWturn,dwell,target_no,locateNextTarget,move2nextTarget,rotResolved] = ...
    BB_subr_trajectory(dt,Xold,Yold,HD,Vlin,dist2f_cur,percep_flag,imag_flag,targetlist,target_no,dwell,DWELL,locateNextTarget, ...
            move2nextTarget,rotResolved,CWturn,CCWturn,toturn,mental_nav)


if percep_flag==1 || mental_nav     % agent explores environment and is perceptually driven, no memory recall, only top down information flow. only under this condition the agent moves

    CWturn     = 0;           % in case none of the ifs is executed
    CCWturn    = 0;
    Xnew       = Xold;
    Ynew       = Yold;
    ang2f_tot  = 0;

    if move2nextTarget == 0
        
        targetlist
        target_no
        
        if locateNextTarget == 1                                    % determine amount to turn
            Xgoal       = targetlist(target_no,1);
            Ygoal       = targetlist(target_no,2);
            ag2goal_vec = [Xgoal-Xold Ygoal-Yold];                  % vector from agent to goal
            dist2f_cur  = sqrt((Xgoal-Xold)^2+(Ygoal-Yold)^2);
            HD          = HD+pi/2;                                  % because BBB 0 deg is north.
            HDag_vec    = [cos(HD) sin(HD)];                        % must always be shorter than ag2goal
            HDag_vecP90 = [cos(HD+pi/2) sin(HD+pi/2)];
            xtan2       = sum(ag2goal_vec.*HDag_vec);               % projection on locally horizontal component
            ytan2       = sum(ag2goal_vec.*HDag_vecP90);            % projection on locally vertical component
            ang2f_tot   = atan2(ytan2,xtan2);                       % agent must turn this much
            ang2f_tot   = round(ang2f_tot*100)/100;
            if ang2f_tot==0
                rotResolved     = 1;
                move2nextTarget = 1;                                % no rot, move directly to target
                CWturn  = 0;
                CCWturn = 0;
            else
                rotResolved = 0;
                if ang2f_tot>=0
                    CWturn  = 0;
                    CCWturn = 1;                                    % turn will go left/CCW
                else                                                % turn will go right/CW
                    ang2f_tot =-ang2f_tot;
                    CWturn  = 1;
                    CCWturn = 0;
                end
            end
            locateNextTarget = 0;
 
            Xnew        = Xold;
            Ynew        = Yold;
        end                                                         % done here, first resolve turning in the main BBBB script way outside of this script

    end

    if move2nextTarget == 1   % this if is executed when turning is resolved, needs the flag to be set
        
        dist2f_cur      = dist2f_cur - Vlin*dt;                 % total distance minus linear velocity times time
        Xgoal           = targetlist(target_no,1);
        Ygoal           = targetlist(target_no,2);
        ag2goal_int_vec = [Xgoal-Xold Ygoal-Yold]/(sqrt(sum([Xgoal-Xold Ygoal-Yold].*[Xgoal-Xold Ygoal-Yold])))*Vlin*dt;   % intermediate ag2goal, Xag and Yag change along the way
        Xnew            = Xold+ag2goal_int_vec(1);
        Ynew            = Yold+ag2goal_int_vec(2);              % actual Xnew and Ynew can be float.
        if dist2f_cur<=0                                        % reached drop location => dwell
            move2nextTarget  = 0;                               % done, don't re-enter this if next timestep
            locateNextTarget = 0;
            dwell            = DWELL;                           % arbitrary dwell period
            dist2f_cur       = 0;
            Xnew             = Xold;
            Ynew             = Yold;
        end
        ang2f_tot = 0;
        CWturn    = 0;
        CCWturn   = 0;
        
    end
    
else     % if the agent is recalling or imagining, it does not move in the 'real' world

    ang2f_tot  = 0;
    dist2f_cur = 0;
    CWturn     = 0;
    CCWturn    = 0;
    Xnew       = Xold;
    Ynew       = Yold;
    dwell      = 0;

end


