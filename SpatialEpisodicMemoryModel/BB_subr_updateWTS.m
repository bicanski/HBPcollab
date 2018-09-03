

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


% this function implemets simple Hebbian associations between MTL model
% components at the chosen time of encoding and saves them to the respective weight matrices


function [H2OVCwts,OVC2Hwts,OVC2BVCwts,BVC2OVCwts,oPR2OVCwts,OVC2oPRwts,oPR2HDwts,oPR2Hwts,H2oPRwts] = ...
    BB_subr_updateWTS(HDact,OVCact,Hact,BVCact,oPRact,H2OVCwts,OVC2Hwts,OVC2BVCwts,BVC2OVCwts,oPR2OVCwts, ...
    OVC2oPRwts,oPR2Hwts,oPR2HDwts,step,N_oPR,N_BVC,OVC2oPR_inhib_phi)



disp('                ')
disp('****************')
disp('updating weights')
disp(['time_' int2str(step) '_'])
disp('****************')
disp('                ')



% OVC2H and H2OVC weights
OVC2Hwts_after    = OVC2Hwts;
Hact_for_update   = max(Hact,0);                                   % because of negative activations
OVCact_for_update = max(OVCact,0);   
OVCact_for_update(OVCact_for_update<0.05) = 0;                     % ...
OVC_x_H_act       = Hact_for_update*OVCact_for_update';
OVC_x_H_act       = OVC_x_H_act/max(max(OVC_x_H_act));             % normalize
tmp1              = (OVC_x_H_act>0.05);                            % rate threshold, eliminate background 
OVC2Hwts_after(tmp1) = OVC2Hwts(tmp1) + OVC_x_H_act(tmp1);
OVC2Hwts             = OVC2Hwts_after;
H2OVCwts = OVC2Hwts';



% oPR2OVC and OVC2oPR weights
OVC2oPRwts_after    = OVC2oPRwts;
oPRact_for_update   = max(oPRact,0);                               % because of negative activations
OVCact_for_update   = max(OVCact,0);                               % ...
OVCact_for_update(OVCact_for_update<0.05) = 0;
OVC_x_oPR_act       = oPRact_for_update*OVCact_for_update';
OVC_x_oPR_act       = OVC_x_oPR_act/max(max(OVC_x_oPR_act));       % normalize
tmp1              = (OVC_x_oPR_act>0.05);                          % rate threshold, eliminate background
OVC2oPRwts_after(tmp1) = OVC2oPRwts(tmp1) + OVC_x_oPR_act(tmp1);
OVC2oPRwts             = OVC2oPRwts_after;
oPR2OVCwts = OVC2oPRwts';



% oPR2H  weights

oPR2Hwts_after    = oPR2Hwts;
oPRact_for_update = max(oPRact,0);                                 % because of negative activations
Hact_for_update   = max(Hact,0);                                   % ...
H_x_oPR_act       = oPRact_for_update*Hact_for_update';
H_x_oPR_act       = H_x_oPR_act/max(max(H_x_oPR_act));             % normalize
oPR_x_H_act       = H_x_oPR_act';
tmp1              = (oPR_x_H_act>0.2);                             % individual thresh
oPR2Hwts_after(tmp1) = oPR2Hwts(tmp1) + oPR_x_H_act(tmp1);
oPR2Hwts             = oPR2Hwts_after;
H2oPRwts = oPR2Hwts';



% oID to HD 
oPR2HDwts_after   = oPR2HDwts;
oPRact_for_update = max(oPRact,0);                                 % because of negative activations
HDact_for_update  = max(HDact,0);                                  
HD_x_oPR_act      = oPRact_for_update*HDact_for_update';
oPR_x_HD_act      = HD_x_oPR_act';
oPR_x_HD_act      = oPR_x_HD_act/max(max(oPR_x_HD_act));           % normalize
tmp1              = (oPR_x_HD_act>0.2);                            % individual thresh
oPR2HDwts_after(tmp1) = oPR2HDwts(tmp1) + oPR_x_HD_act(tmp1);
oPR2HDwts             = oPR2HDwts_after;


% BVC2OVC, could be necessary to support OVC firing in imagery as contentextual input
BVC2OVCwts_after    = BVC2OVCwts;
OVCact_for_update   = max(OVCact,0);                               % because of negative activations
BVCact_for_update   = max(BVCact,0);                               % ...
BVCact_for_update(BVCact_for_update<0.05) = 0;
BVC_x_OVC_act       = OVCact_for_update*BVCact_for_update';
BVC_x_OVC_act       = BVC_x_OVC_act/max(max(BVC_x_OVC_act));       % normalize
tmp1              = (BVC_x_OVC_act>0.07);                          % ...
BVC2OVCwts_after(tmp1) = BVC2OVCwts(tmp1) + BVC_x_OVC_act(tmp1);
BVC2OVCwts             = BVC2OVCwts_after;


