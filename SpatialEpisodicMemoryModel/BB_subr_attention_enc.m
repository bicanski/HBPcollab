

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


function [ObjCue_percep,oPR_rate,ObjEncoded,H2LVCwts,LVC2Hwts,LVC2BVCwts,BVC2LVCwts,oPR2LVCwts, ...
          LVC2oPRwts,oPR2HDwts,oPR2Hwts,H2oPRwts,Xag_encode,Yag_encode,CurrObjCenX,CurrObjCenY] = BB_subr_attention_enc( ...
          ObjCenX,Xag,ObjCenY,Yag,ObjAttThresh,ATTcycle,r0,dir,line,HDag,step,ObjCue_percep, ...
          ObjEncoded,ObjEncThresh,HD_rate,LVC_rate,H_rate,NHx,NHy,BVC_rate,oPR_rate,H2LVCwts,LVC2Hwts, ...
          LVC2BVCwts,BVC2LVCwts,oPR2LVCwts,LVC2oPRwts,oPR2Hwts,H2oPRwts,oPR2HDwts,N_oPR,N_BVC,LVC2oPR_inhib_phi,simflag)

% return Object cueing current for oPW and oPR rate
% latter driven by putative recognition process along ventral vis stream
% and modulated by heuristic attention

% Attending to things does NOT mean we will commit them to memory. 
% That is detemined by a separate criterion (relevance of object, our current proxy is distance)

% With multiple objects we don't want to
% sample all of them all the time, the attention cycle
% would be too long. Also we memorize each object from a single location 
% attend sequentially to everything in a reasonably large (but not too large) hemicircle with yourself at origin. Can be changed easily

oPR_rate = oPR_rate * 0;

D2obj       = sqrt((ObjCenX-Xag).^2+(ObjCenY-Yag).^2);   % Distance to object
Obj2attend  = find(D2obj<ObjAttThresh);
NobjATT     = length(Obj2attend);
ATTperObj   = ATTcycle/NobjATT;
ATTcounter  = mod(step,ATTcycle);
Xag_encode  = [];                                        % memorize encoding location
Yag_encode  = [];
CurrObjCenX = [];
CurrObjCenY = [];

for l=1:NobjATT                                                                  % looks like a loop but actually only selects the right Obj
    ifstring = ['ATTcounter>' num2str(l-1) '*ATTperObj && ATTcounter<=' num2str(l) '*ATTperObj'];
    if eval(ifstring)
        
        CurrObjCenX = ObjCenX(Obj2attend(l));                                    % centres
        CurrObjCenY = ObjCenY(Obj2attend(l));
        CurrObjAllX = (CurrObjCenX-0.2:0.1:CurrObjCenX+0.2)';                    % extended object
        CurrObjAllY = (CurrObjCenY-0.2:0.1:CurrObjCenY+0.2)';
        
        if sqrt((CurrObjCenX-Xag).^2 + (CurrObjCenY-Yag).^2)>0.5                 % minimum distance from agent is 1 distance units
            [OBJcues,~,~,~,~,~,~,~] = BB_subr_PdrivePW_withObj(r0,dir,line,CurrObjAllX,CurrObjAllY,Xag,Yag,HDag,0,0,1);
            % OBJcues is empty if object not in visual field, i.e. it has dropped out
            if ~isempty(OBJcues)
                oPR_rate                = oPR_rate * 0;
                oPR_rate(Obj2attend(l)) = 1;
            else
                oPR_rate                = oPR_rate * 0;
            end
        else
            OBJcues  = [];
            oPR_rate = oPR_rate * 0;
        end
        
        ObjCue_percep = 100 * BB_subr_cue_PWact_multiseg(0.5,OBJcues);
        
        if round(mod(ATTcounter,ATTperObj)) == round(ATTperObj-1)                 % update weights at end of cycle, i.e. after buildup
            oPR_rate(Obj2attend(l)) = 1;                                          % recognition at end of cylce
            if ObjEncoded(Obj2attend(l))==0 && D2obj(Obj2attend(l))<=ObjEncThresh && max(LVC_rate)>0.5 % when the object is closer than X distance unit it is "behaviorally relevant" (aritrarily defined) and encoded
                        [H2LVCwts,LVC2Hwts,LVC2BVCwts,BVC2LVCwts,oPR2LVCwts,LVC2oPRwts,oPR2HDwts,oPR2Hwts,H2oPRwts] = ...
                BB_subr_updateWTS(HD_rate,LVC_rate,H_rate,BVC_rate,oPR_rate, ...
                         H2LVCwts,LVC2Hwts,LVC2BVCwts,BVC2LVCwts,oPR2LVCwts,LVC2oPRwts,oPR2Hwts,oPR2HDwts,step,N_oPR,N_BVC,LVC2oPR_inhib_phi);
                disp(Xag); disp(Yag);
                ObjEncoded(Obj2attend(l)) = 1;
                if simflag == 21
                    [Xag_encode,Yag_encode] = BB_subr_XYestimFromPCs(H_rate,NHx,NHy);   % estimate position from PCs
                end
            end
        end
        
    end
end


