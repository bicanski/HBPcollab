

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


% a simple routine to cue with the wall identity, which we assume the agent
% can recognize. We determine the minimum distance to each wall (N,S,E,W)
% and how many boundary points of each wall are visible. Both factors
% contribute to the PRcue during perception, which drives the PR neurons.
% This helps the model remove ambiguity in the simple 4 wall environment.

% hear for more boundaries, but should be unifie with the scripts 'BB_subr_PR_BoundaryCueing.m'


function [PRcue_percep] = BB_subr_PR_BoundaryCueing_with_Barrier(VBX_rt,VBY_rt,HD,Xag,Yag,Ybarr_beg,Ybarr_end,Xbarr_beg)

R    = [cos(HD) -sin(HD); sin(HD)  cos(HD)];
tmp  = [VBX_rt ; VBY_rt];
tmp2 = R*tmp;
Xtmp = tmp2(1,:)+Xag;
Ytmp = tmp2(2,:)+Yag;

Xtmp = Xtmp(VBY_rt>0);%-min(Xtmp(VBY_rt>0));
Ytmp = Ytmp(VBY_rt>0);

W = length(find(Xtmp==1));    % amount of West wall pts

E3 = length(find(Xtmp==21&Ytmp>Ybarr_end));   % amount of East wall pts
E4 = length(find(Ytmp==Ybarr_end));
E5 = length(find(Xtmp==Xbarr_beg));
E6 = length(find(Ytmp==Ybarr_beg));
E7 = length(find(Xtmp==21&Ytmp<Ybarr_beg));

S = length(find(Ytmp==1));   % amount of South wall pts
N = length(find(Ytmp==21));  % amount of North wall pts

if W>0   W = W/abs(min(Xtmp(Xtmp==1))-Xag);   end   % the closer to the wall the more input

if E3>0   E3 = E3/abs(min(Xtmp(Xtmp==21&Ytmp>Ybarr_end))-Xag);   end
if E4>0   E4 = E4/abs(min(Ytmp(Ytmp==Ybarr_end))-Yag);   end
if E5>0   E5 = E5/abs(min(Xtmp(Xtmp==Xbarr_beg))-Xag);   end
if E6>0   E6 = E6/abs(min(Ytmp(Ytmp==Ybarr_beg))-Yag);   end
if E7>0   E7 = E7/abs(min(Xtmp(Xtmp==21&Ytmp<Ybarr_beg))-Xag);   end

if S>0   S = S/abs(min(Ytmp(Ytmp==1))-Yag);   end
if N>0   N = N/abs(min(Ytmp(Ytmp==21))-Yag);   end

totmax = max([W E3 E4 E5 E6 E7 S N]);

if totmax == 0
    totmax = 1;
end

PRcue_percep = [N; W; S; E3; E4; E5; E6; E7]*50/totmax;
