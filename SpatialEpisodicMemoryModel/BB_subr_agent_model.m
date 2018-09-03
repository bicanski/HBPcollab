

% Subroutine for the BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


function [ Xag            ,Yag                ,ang2f_tot                 ,toturn                   ,dist2f_cur           ,dwell          , ...
           Xag_imag_nav   ,Yag_imag_nav       ,ang2f_tot_imag_nav        ,toturn_imag_nav          ,dist2f_cur_imag_nav  ,dwell_imag_nav , ...
           navi_flag      ,target_no          ,locateNextTarget          ,move2nextTarget          ,rotResolved          , ...
           navi_flag_imag ,target_no_imag_nav ,locateNextTarget_imag_nav ,move2nextTarget_imag_nav ,rotResolved_imag_nav ,CWturn ,CCWturn ] = ...
   BB_subr_agent_model(...
           step, Vlin ,HDag ,HDestim_prev ,HDestim ,imag_cuedur_nav ,percep_flag ,imag_flag ,dt ,CWturn ,CCWturn , H_rate, NHx, NHy, ...
           Xag            ,Yag                       ,ang2f_tot                ,toturn               ,dist2f_cur          ,targetlist          ,target_no         , dwell      , ...
           Xag_imag_nav   ,Yag_imag_nav              ,ang2f_tot_imag_nav       ,toturn_imag_nav      ,dist2f_cur_imag_nav ,targetlist_imag_nav ,target_no_imag_nav, dwell_imag_nav, ...
           DWELL          ,locateNextTarget          ,move2nextTarget          ,rotResolved          ,navi_flag , ...
           DWELL_imag_nav ,locateNextTarget_imag_nav ,move2nextTarget_imag_nav ,rotResolved_imag_nav ,navi_flag_imag ,encoding, imag_start_nav, switchdur)

       
       
       
% returns real and imagnined location estimates based of translation and rotation
% this script should be rewritten for clarity

if ~imag_cuedur_nav   % for normal navigation
    
    if ~imag_flag && navi_flag                          % move around only when not imagining and after reestablishing bottom-up perceptual representation
        if dwell<0 && rotResolved==1 && ~encoding       % ~encoding >>> Don't move while setting up new weights for encountered object
                   [Xag,Yag,ang2f_tot,toturn,dist2f_cur,CWturn,CCWturn,dwell,target_no, ...
                   locateNextTarget,move2nextTarget,rotResolved] = ... 
            BB_subr_trajectory(dt,Xag,Yag,HDag, ...
                   Vlin,dist2f_cur,percep_flag,imag_flag,targetlist,target_no,dwell,DWELL, ...
                   locateNextTarget,move2nextTarget,rotResolved,CWturn,CCWturn,toturn);
        else
            dwell = dwell-1;
        end
    end
    % k_HD: here we also track roation and compare to agent model
    if ~imag_flag && navi_flag
        if dwell==0                        % only executed once, because of decrement
            dwell            = dwell-1;
            locateNextTarget = 1;
            target_no        = target_no + 1;
        end
        if ang2f_tot>0 && rotResolved==0
            toturn    = ang2f_tot;
            ang2f_tot = 0;                  % assures that this if gets executed only once
        end
        if toturn>0 && rotResolved==0
            toturn = toturn - abs(min(abs(HDestim_prev-HDestim),abs(abs(HDestim_prev-HDestim)-(2*pi))));
            %disp(toturn);
        end
        if toturn<=0 && rotResolved==0
            move2nextTarget = 1;
            rotResolved     = 1;
            CWturn          = 0;
            CCWturn         = 0;
        end
    end
end



if imag_cuedur_nav  % if at some point imagined movement is planned use this code
    
    if ~imag_flag && navi_flag                          % move around only when not imagining and after reestablishing bottom-up perceptual representation
        if dwell<0 && rotResolved==1 && ~encoding       % ~encoding >>> Don't move while setting up new weights for encountered object
                   [Xag,Yag,ang2f_tot,toturn,dist2f_cur,CWturn,CCWturn,dwell,target_no, ...
                   locateNextTarget,move2nextTarget,rotResolved] = ...
            BB_subr_trajectory(dt,Xag,Yag,HDag, ...
                   Vlin,dist2f_cur,percep_flag,imag_flag,targetlist,target_no,dwell,DWELL, ...
                   locateNextTarget,move2nextTarget,rotResolved,CWturn,CCWturn,toturn,encoding);
        else
            dwell = dwell-1;
        end
        % k_HD: here we also track roation and compare to agent model
        if dwell==0                        % only executed once, because of decrement
            dwell            = dwell-1;
            locateNextTarget = 1;
            target_no        = target_no + 1;
        end
        if ang2f_tot>0 && rotResolved==0
            toturn    = ang2f_tot;
            ang2f_tot = 0;                  % assures that this if gets executed only once
        end
        if toturn>0 && rotResolved==0
            toturn = toturn - abs(min(abs(HDestim_prev-HDestim),abs(abs(HDestim_prev-HDestim)-(2*pi))));
            %disp(toturn);
        end
        if toturn<=0 && rotResolved==0
            move2nextTarget = 1;
            rotResolved     = 1;
            CWturn          = 0;
            CCWturn         = 0;
        end
    end
    
    if imag_flag && navi_flag_imag && step>imag_start_nav+switchdur    % imagined movement, only start when imag representation is established
        
        if step == imag_start_nav+switchdur+1     % do this only once, then the routine below takes over in creating the trajectory
            Htmp = reshape(H_rate,NHx,NHy)';
            Ytmp = sum(Htmp,2);
            Xtmp = sum(Htmp,1);
            [i,j] = find(Htmp==max(max(Htmp)));
            Xag_imag_nav = find(Xtmp==max(Xtmp))/2;
            Yag_imag_nav = find(Ytmp==max(Ytmp))/2;
            locateNextTarget_imag_nav = 1;
            rotResolved_imag_nav      = 1;
            move2nextTarget_imag_nav  = 0;
        end
        
        if dwell_imag_nav<0 && rotResolved_imag_nav==1 && ~encoding       % ~encoding >>> Don't move while setting up new weights for encountered object
                   [Xag_imag_nav,Yag_imag_nav,ang2f_tot_imag_nav,toturn_imag_nav, ...
                   dist2f_cur_imag_nav,CWturn,CCWturn,dwell_imag_nav,target_no_imag_nav, ...
                   locateNextTarget_imag_nav,move2nextTarget_imag_nav,rotResolved_imag_nav] = ...
            BB_subr_trajectory(dt,Xag_imag_nav,Yag_imag_nav,HDag, ...
                   Vlin,dist2f_cur_imag_nav,percep_flag,imag_flag,targetlist_imag_nav, ...
                   target_no_imag_nav,dwell_imag_nav,DWELL_imag_nav,locateNextTarget_imag_nav, ...
                   move2nextTarget_imag_nav,rotResolved_imag_nav,CWturn,CCWturn,toturn_imag_nav,imag_cuedur_nav);
        else
            dwell_imag_nav = dwell_imag_nav-1;
        end
        % k_HD: here we also track roation and compare to agent model
        if dwell_imag_nav==0                        % only executed once, because of decrement
            dwell_imag_nav            = dwell_imag_nav-1;
            locateNextTarget_imag_nav = 1;
            target_no_imag_nav        = target_no_imag_nav + 1;
        end
        if ang2f_tot_imag_nav>0 && rotResolved_imag_nav==0
            toturn_imag_nav    = ang2f_tot_imag_nav;
            ang2f_tot_imag_nav = 0;                  % assures that this if gets executed only once
        end
        if toturn_imag_nav>0 && rotResolved_imag_nav==0
            toturn_imag_nav = toturn_imag_nav - abs(min(abs(HDestim_prev-HDestim),abs(abs(HDestim_prev-HDestim)-(2*pi))));
            %disp(toturn);
        end
        if toturn_imag_nav<=0 && rotResolved_imag_nav==0
            move2nextTarget_imag_nav = 1;
            rotResolved_imag_nav     = 1;
            CWturn          = 0;
            CCWturn         = 0;
        end
    end
    
end


