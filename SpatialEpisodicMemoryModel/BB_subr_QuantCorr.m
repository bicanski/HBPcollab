
function [PRcorr_IM, oPRcorr_IM, LVCcorr_IM, BVCcorr_IM, PCcorr_IM, PWcorr_IM, oPWcorr_IM, ...
          PRcorr_PE, oPRcorr_PE, LVCcorr_PE, BVCcorr_PE, PCcorr_PE, PWcorr_PE, oPWcorr_PE, ...
          PRcorr_PErand, oPRcorr_PErand, LVCcorr_PErand, BVCcorr_PErand, PCcorr_PErand, PWcorr_PErand, oPWcorr_PErand, ...
          CCnt_rand] = BB_subr_QuantCorr(step, ObjEncoded, ObjEncoded_prev, ObjRecalled, ObjRecalled_prev, CCnt_rand, ...
                                         PR_rates, oPR_rates, LVC_rates, BVC_rates, PC_rates, PW_rates, oPW_rates, ...
                                         PRcorr_IM, oPRcorr_IM, LVCcorr_IM, BVCcorr_IM, PCcorr_IM, PWcorr_IM, oPWcorr_IM, ...
                                         PRcorr_PE, oPRcorr_PE, LVCcorr_PE, BVCcorr_PE, PCcorr_PE, PWcorr_PE, oPWcorr_PE, ...
                                         PRcorr_PErand, oPRcorr_PErand, LVCcorr_PErand, BVCcorr_PErand, PCcorr_PErand, PWcorr_PErand, oPWcorr_PErand)
                                     
% two situations should trigger registering firing rates for comparision:
% 1. at encoding, the array ObjEncoded is incremented elsewhere in that case, 
% and only if firing rates are reasonably high, that's fine for the purpose of this subroutine
% 2. at recall we should register firing rate, but we must make sure there
% was enough time for buildup of activity


% threshold rates below 10% of max
%if sum(ObjEncoded-ObjEncoded_prev)>0 || sum(ObjRecalled-ObjRecalled_prev)>0
th = 0.1;
PR_rates(PR_rates<th)   = 0;
oPR_rates(oPR_rates<th) = 0;
LVC_rates(LVC_rates<th) = 0;
BVC_rates(BVC_rates<th) = 0;
PC_rates(PC_rates<th)   = 0;
PW_rates(PW_rates<th)   = 0;
oPW_rates(oPW_rates<th)  = 0;
%end

% during perception
if sum(ObjEncoded-ObjEncoded_prev)>0                   % an object has just been encoded, there is a change in number of encoded objects
    objind = find(ObjEncoded-ObjEncoded_prev);
    PRcorr_PE(:,objind)  = PR_rates;
    oPRcorr_PE(:,objind) = oPR_rates;
    LVCcorr_PE(:,objind) = LVC_rates;
    BVCcorr_PE(:,objind) = BVC_rates;
    PCcorr_PE(:,objind)  = PC_rates;
    PWcorr_PE(:,objind)  = PW_rates;
    oPWcorr_PE(:,objind) = oPW_rates;
else                                                   % else will compare imag representation to perceptual representation at random times
    if mod(step,100) == 0
        PRcorr_PErand(:,CCnt_rand)  = PR_rates;
        oPRcorr_PErand(:,CCnt_rand) = oPR_rates;
        LVCcorr_PErand(:,CCnt_rand) = LVC_rates;
        BVCcorr_PErand(:,CCnt_rand) = BVC_rates;
        PCcorr_PErand(:,CCnt_rand)  = PC_rates;
        PWcorr_PErand(:,CCnt_rand)  = PW_rates;
        oPWcorr_PErand(:,CCnt_rand) = oPW_rates;
        CCnt_rand                   = CCnt_rand + 1;   % counter for random snapshots
    end
end

% during recall
if sum(ObjRecalled-ObjRecalled_prev)>0               % there is a change in number of recalled objects
    objind = find(ObjRecalled-ObjRecalled_prev);
    PRcorr_IM(:,objind)  = PR_rates;
    oPRcorr_IM(:,objind) = oPR_rates;
    LVCcorr_IM(:,objind) = LVC_rates;
    BVCcorr_IM(:,objind) = BVC_rates;
    PCcorr_IM(:,objind)  = PC_rates;
    PWcorr_IM(:,objind)  = PW_rates;
    oPWcorr_IM(:,objind) = oPW_rates;
end

