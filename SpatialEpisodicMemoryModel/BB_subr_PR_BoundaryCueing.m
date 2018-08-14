

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% REF
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


% a simple routine to cue with the wall identity, which we assume the agent
% can recognize. We determine the minimum distance to each wall (N,S,E,W)
% and how many boundary points of each wall are visible. Both factors
% contribute to the PRcue during perception, which drives the PR neurons.
% This helps the model remove ambiguity in the simple 4 wall environment.


function [PRcue_percep] = BB_subr_PR_BoundaryCueing(VBX_rt,VBY_rt,HD,Xag,Yag)

R    = [cos(HD) -sin(HD); sin(HD)  cos(HD)];
tmp  = [VBX_rt ; VBY_rt];
tmp2 = R*tmp;
Xtmp = tmp2(1,:)+Xag;
Ytmp = tmp2(2,:)+Yag;
Xtmp = Xtmp(VBY_rt>0);%-min(Xtmp(VBY_rt>0));
Ytmp = Ytmp(VBY_rt>0);

W = length(find(Xtmp==1));    % amount of West wall pts
E = length(find(Xtmp==21));   % amount of East wall pts
S = length(find(Ytmp==1));   % amount of South wall pts
N = length(find(Ytmp==21));  % amount of North wall pts

if W>0   W = W/abs(min(Xtmp(Xtmp==1))-Xag);   end   % the closer to the wall the more input
if E>0   E = E/abs(min(Xtmp(Xtmp==21))-Xag);   end
if S>0   S = S/abs(min(Ytmp(Ytmp==1))-Yag);   end
if N>0   N = N/abs(min(Ytmp(Ytmp==21))-Yag);   end

totmax = max([W E S N]);

if totmax == 0
    totmax = 1;
end

PRcue_percep = [N; W; S; E]*50/totmax;
