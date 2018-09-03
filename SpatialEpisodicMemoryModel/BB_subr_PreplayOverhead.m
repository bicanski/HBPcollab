

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

if simflag == 50   
    
    % extra overhead for preplay/planning simulation
    % changes for sealed-off part of arena introduce all this messy
    % overhead, because the reservoir place cells, random connections
    % from grid cell, variables for tracking activity and for plotting 
    % etc. need to be defined
    
    % coords of boundaries: BX,BY are coords to "black out":
    minXb = (find(roomGridPrePlay(12,3:end),1,'first')+2)*2;         % plus 2 because we start at 3, times 2 because 2 PCs per unit
    maxXb = (find(roomGridPrePlay(12,3:end),1,'last')+2)*2;
    minYb = (find(roomGridPrePlay(3:end-3,16),1,'first')+2)*2;
    maxYb = (find(roomGridPrePlay(3:end-3,16),1,'last')+2)*2;
    
    tmp1         = reshape(zeros(N_H,1),NHx,NHy);
    tmp1(minXb:maxXb,minYb:maxYb) = nan;
    tmp2         = reshape(tmp1,NHx*NHy,1);
    unexploredXY = (isnan(tmp2)-1)*-1;      
    
    N_env = sum(unexploredXY);
    N_res = sum((unexploredXY-1)*-1);
    
    Xunexp = repmat(unexploredXY',[NHx*NHy,1]);
    Yunexp = repmat(unexploredXY,[1,NHx*NHy]);
    
    all2unexp2all = logical(((Xunexp.*Yunexp)-1)*-1);                % indices for connections 
    unexp2unexp   = (((Xunexp+Yunexp)-1)*-1)==1;                     % variable names hopefully make sense
    unexp2all     = logical(Xunexp-1);                               % unexp = unexplored
    unexp2fam     = logical(unexp2all-unexp2unexp);                  % fam = familiar part of environment
    fam2unexp     = unexp2fam';
    
    H2Hwts(all2unexp2all) = min(min(H2Hwts));                        % HPC global inhibition inherited from normal weight matrix
    H2Hwts(unexp2fam)     = min(min(H2Hwts))*1.5;
    H2Hwts(fam2unexp)     = min(min(H2Hwts));                        % this includes the 1.5 from line above
    %H2Hwts(unexp2unexp)   = min(min(H2Hwts))*2;
    
    H2Hcap                = mean(sum(H2Hwts(H2Hwts>0),2))*1;  
    tmp_wts               = unexp2unexp.*rand(size(unexp2unexp));
    tmp_wts(tmp_wts<0.94) = 0;                                       % 0.94 set sparseness via threshold
    capvec                = sum(tmp_wts,2);                          % values capped
    capmat                = repmat(capvec,1,length(tmp_wts(1,:)));
    cmL                   = capmat>0;
    tmp_wts(cmL)          = tmp_wts(cmL)./capmat(cmL)*H2Hcap;        % sum toal of wts onto a reservoir PC should be constant
    tmp_wts(tmp_wts==0)   = min(min(H2Hwts));
    H2Hwts(unexp2unexp)   = tmp_wts(unexp2unexp);  
    
    XYwts_ind_H2BVC = logical((repmat(unexploredXY',[N_BVC,1])-1)*-1);
    XYwts_ind_BVC2H = logical((repmat(unexploredXY,[1,N_BVC])-1)*-1);
    XYwts_ind_H2PR  = logical((repmat(unexploredXY',[N_PR,1])-1)*-1);
    XYwts_ind_PR2H  = logical((repmat(unexploredXY,[1,N_PR])-1)*-1);
    XYwts_ind_GC2PC = logical((repmat(unexploredXY,[1,N_GC])-1)*-1);
    
    H2BVCwts(XYwts_ind_H2BVC) = 0;
    BVC2Hwts(XYwts_ind_BVC2H) = 0;
    H2PRwts(XYwts_ind_H2PR)   = 0;
    PR2Hwts(XYwts_ind_PR2H)   = 0;
    GC2PCwts(XYwts_ind_GC2PC) = 0;                                   % set previously learned GC2PC weights to zero, then ...
    
    tmp_sum  = sum(GC2PCwts,2);                                      % make new random weights to reservoir PCs
    GC2PCcap = mean(tmp_sum(1:100));
    tmp_wts  = XYwts_ind_GC2PC.*rand(size(XYwts_ind_GC2PC));
    tmp_wts(tmp_wts<0.97) = 0;                                       % 0.97 set sparseness via threshold
    tmp_wts(tmp_wts>0)    = 1;
    capvec   = sum(tmp_wts,2);
    capmat   = repmat(capvec,1,length(tmp_wts(1,:)));
    cmL      = capmat>0;
    tmp_wts(cmL) = tmp_wts(cmL)./capmat(cmL)*GC2PCcap;               % sum toal of wts onto a reservoir PC should be constant
    GC2PCwts(XYwts_ind_GC2PC) = tmp_wts(XYwts_ind_GC2PC)*1.8; 
    GC2PCwts(logical((XYwts_ind_GC2PC-1)*-1)) = GC2PCwts(logical((XYwts_ind_GC2PC-1)*-1));
    
    Hm1Count   = 1;
    Hm2Count   = 1;
    Hm3Count   = 1;                                                 % counter for normalizing rate maps by occupancy
    Cflag      = 0;
    Hm1Cmean   = [];                                                % to track firing rates
    Hm2Cmean   = [];
    Hm3Cmean   = [];
  
    Rlininds   = logical(unexploredXY-1);                           % linear indices of unexplored part of env
    ENVlininds = logical(unexploredXY);
    Rcartinds  = logical(reshape((unexploredXY-1),NHx,NHy)');       % cartesian indices of unexplored part of env
    toswap     = zeros(size(H_rate));
    
    % for pre-replay renormalization later
    resPCs2BVCs = repmat(unexp2unexp(605,:),N_BVC,1);
    famPCs2BVCs = logical((resPCs2BVCs-1)*-1);
    BVCs2resPCs = resPCs2BVCs';
    BVCs2famPCs = logical((BVCs2resPCs-1)*-1);
    
end

% lrate_H2H       = 0.0001;
lrate_GC2H        = 0.0000025;   % learning rates for very (!) simple learning setup 
lrate_H2BVC       = 0.0000025;
lrate_BVC2H       = 0.0000025;

Hmem1            = [];                            % ~32MB array to track firing rates along imagined route
Hmem2            = [];
Hmem3            = [];


