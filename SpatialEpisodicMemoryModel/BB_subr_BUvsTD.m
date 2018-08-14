

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% REF
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


function [oPRCue,HDCue,oHDCue,HDag,old_HD,percep_flag,navi_flag,navi_flag_imag,imag_flag,recallobj,plt_ttl_flag,imag2] = ...
             BB_subr_BUvsTD(step,imag_start,imag_start_nav,imag_cuedur,imag_cuedur_nav,switchdur,HDag,old_HD,first_imag_HD, ...
             percep_flag,navi_flag,navi_flag_imag,imag_flag,recallobj,plt_ttl_flag,imag2,oPRCue,HDCue,oHDCue,simflag,load_flag,oPRCuescale)

         
    % static imagination
    if ~imag_cuedur_nav
        if step>=imag_start && step<=imag_start+imag_cuedur                         % 1. imagination phase.  makes only sense after we have encoded something
            if step==imag_start
                old_HD   = HDag;                                                    % remember HD for reset later
            end
            percep_flag  = 0;                                                       % "stop and think"
            navi_flag    = 0;
            imag_flag    = 1;
            recallobj    = 1;
            if load_flag                                                            % here we load data, test recall of all encoded objects
                recallNO       = ceil(recallcount/imagPerObj);
                ObjInd         = Obj2recall(min(recallNO,length(Obj2recall)));
                oPRCue         = zeros(9,1);
                oPRCue(ObjInd) = 160;
                plt_ttl_flag   = 5;
            else
                if simflag == 10 || simflag == 100 || simflag == 12 || simflag == 11
                    oPRCue       = 160;                                                 % cue with only memorized objects. Could be any.
                    plt_ttl_flag = 2;                                                   % title of plot in video
                end
                if simflag == 21
                    plt_ttl_flag = 2;                                                   % title of plot in video
                end
                if simflag == 31
                    oPRCue       = [160; 0];                                            % cue with only memorized objects. Could be any.
                    plt_ttl_flag = 2;                                                   % title of plot in video
                end
                if simflag == 241 || simflag == 242
                    if step>=imag_start && step<=imag_start+imag_cuedur/2 
                        oPRCue       = [160; 0];                                        % cue with only memorized objects. Could be any.
                    end
                    if step>=imag_start+imag_cuedur/2 && step<=imag_start+imag_cuedur 
                        oPRCue       = [0; 160];                                        % cue with only memorized objects. Could be any.
                    end
                    plt_ttl_flag = 241;                                                 % title of plot in video
                end
            end
        end
        if step>=imag_start+imag_cuedur && step<imag_start+imag_cuedur+switchdur    % 2. Give it time to reestablish perception before continuing to navigate
            navi_flag    = 0;                                                       % switchduration should be kept, but in the VR we should have an external signal
            percep_flag  = 1;                                                       % which tells us when to recall an object (object cue also needs to be supplied,
            imag_flag    = 0;                                                       % meaning the decision which object to recall is not part of the BBBB. So imag_start
            recallobj    = 0;                                                       % etc are just for demonstration
            oPRCue       = 0 * oPRCue;
            HDag         = old_HD;                                                  % reinstate old HD
            HDCue        = 60 * BB_subr_cue_HDact(old_HD);                          % recalculate correct HDC activities
            plt_ttl_flag = 3;
        end
        if step==imag_start+imag_cuedur+switchdur                                   % 3. resume navigation
            HDCue        = 0* HDCue;
            navi_flag    = 1;
            plt_ttl_flag = 1;
        end
    end
    
    % imagined movement
    if imag_cuedur_nav 
        if step>=imag_start_nav && step<=imag_start_nav+imag_cuedur_nav             % 1. imagination phase. makes only sense after we have encoded something
            if step==imag_start_nav
                old_HD   = HDag;                                                    % remember HD for reset later
            end
            if imag2 == 0 && isnan(first_imag_HD) && step==imag_start_nav
                first_imag_HD = HDag;
            end
            if imag2 == 1 && ~isnan(first_imag_HD)                                  % if there is a second recall phase in which we want to repeat the first
                HDag = first_imag_HD;
                oHDCue = 60 * BB_subr_cue_HDact(HDag);
                imag2 = 2;
            end
            percep_flag    = 0;                                                     % "stop and think"
            navi_flag      = 0;   % !!!
            navi_flag_imag = 1;
            imag_flag      = 1;
            if simflag == 50
                recallobj = 0;
            end
            if simflag == 40
                recallobj = 1;
                if step>imag_start_nav && step<imag_start_nav+switchdur/2
                    oPRCue       = oPRCuescale * [1, 0, 0]';                        % cue with only one memorized objects. then mental navigation.
                else
                    oPRCue       = [0, 0, 0]';                                      % in order to release HD via oPR2HD weights
                end
            end
            plt_ttl_flag = 6;                                                       % title of plot in video
        end 
        if step>=imag_start_nav+imag_cuedur_nav && step<imag_start_nav+imag_cuedur_nav+switchdur    % 2. Give it time to reestablish perception before continuing to navigate
            navi_flag    = 0;                                                       % switchduration should be kept, but in the VR we should have an external signal
            navi_flag_imag = 0;
            percep_flag  = 1;                                                       % which tells us when to recall an object (object cue also needs to be supplied,
            imag_flag    = 0;                                                       % meaning the decision which object to recall is not part of the BBBB. So imag_start
            recallobj    = 0;                                                       % etc are just for demonstration
            oPRCue       = 0 * oPRCue;
            HDag         = old_HD;                                                  % reinstate old HD
            HDCue        = 60 * BB_subr_cue_HDact(old_HD);                          % recalculate correct HDC activities
            plt_ttl_flag = 3;
        end
        
        if step==imag_start_nav+imag_cuedur_nav+switchdur                           % 3. resume navigation
            HDCue        = 0* HDCue;
            navi_flag    = 1;
            navi_flag_imag = 0;
            plt_ttl_flag = 1;
        end      
    end
    
    
    