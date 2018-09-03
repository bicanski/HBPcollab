

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


% When we imagine a object in its context, associated PCs will drive OVCs
% for all objects encoded from that location. Meaning, even if we cue with
% only one object, the other associated OVCs (for other objects relative to 
% the current location) will receive some input from PCs, leading to a 
% residual peak in OVCs not coding the position of the currently imagined
% object. This activity spreads to the PWo and can be located to allocate
% attention there. This referes to simflag 31. The function returns a
% cueing current which "allocates attention" to the residual peak.

function [ObjCue_allocATT] = BB_subr_allocATTinPW(r0,dir,line,step,imag_start,imag_cuedur,H_rate,NHx,NHy,oPW_rate, ...
                                                            NBVCR,NBVCTheta,polarDistRes,polarAngRes,HDag,CurrObjCenX,CurrObjCenY,polarDist)

if step>=imag_start+imag_cuedur/2+1    % do this only once or the next timesteps overwrites the residual maximum
    Htmp = reshape(H_rate,NHx,NHy)';
    Ytmp = sum(Htmp,2);
    Xtmp = sum(Htmp,1);
    Xag_imag = find(Xtmp==max(Xtmp))/2;
    Yag_imag = find(Ytmp==max(Ytmp))/2;
    oPWtmp = reshape(oPW_rate,NBVCR,NBVCTheta)';
    [row col] = find(oPWtmp==max(max(oPWtmp)));
    rowinds = row-5:row+5;  % row-4:row+4;
    rowinds(rowinds<=0) = rowinds(rowinds<=0)+51;
    rowinds(rowinds>51) = rowinds(rowinds>51)-51;
    colinds = col-5:col+5;
    colinds(colinds<=0) = colinds(colinds<=0)+16;
    colinds(colinds>16) = colinds(colinds>16)-16;
    oPWtmp(rowinds,colinds) = 0;
    [row col] = find(oPWtmp==max(max(oPWtmp)));             % this should be the residual max
    if polarDist(16)-polarDist(15) == 1
        r   = col*polarDistRes;
    else
        r   = polarDist(col);
    end
    ang = row*polarAngRes;
    Ovec = [r*cos(ang) r*sin(ang)];
    Rtmp = [cos(HDag) -sin(HDag); sin(HDag)  cos(HDag)];
    Ovec = Ovec*Rtmp';                                      % because PW is in egocentric perspective, rotate back
    CurrObjCenY = Ovec(2) + Yag_imag;
    CurrObjCenX = Ovec(1) + Xag_imag;
end

CurrObjAllX = (CurrObjCenX-0.25:0.1:CurrObjCenX+0.25)';     % extended object
CurrObjAllY = (CurrObjCenY-0.25:0.1:CurrObjCenY+0.25)';

if step>=imag_start+imag_cuedur/2+1
    [OBJcuesIM,~,~,~,~,~,~,~] = BB_subr_PdrivePW_withObj(r0,dir,line,CurrObjAllX,CurrObjAllY,Xag_imag,Yag_imag,HDag,0,0,1);
    ObjCue_allocATT = 20 * BB_subr_cue_PWact_multiseg(0.5,OBJcuesIM);
else
    ObjCue_allocATT = 0;
end
