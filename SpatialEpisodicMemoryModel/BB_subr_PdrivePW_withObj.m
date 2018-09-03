

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


function [egocue,VisBndryPtsX_rottrans,VisBndryPtsY_rottrans,Lmarks_rot,BndryPtX,BndryPtY,TrainX,TrainY] = BB_subr_PdrivePW_withObj(r0,dir,line,BndryPtX,BndryPtY,X,Y,HD,step,modstep,obj)


% In the original BBB model translation is modeled in the following way (direct quote 
% from the paper, except for parentheses): Motor efference drives the
% spatial updating of the egocentric representation of the locations of
% landmarks (meaning spatial updating of the egocentric PW representation).
% Specifically, modulation of the posterior parietal egocentric-allocentric
% transformation by motor efference causes allocentric locations to be
% mapped to the egocentric locations pertaining after the current segment
% of movement. The reactivation of the BVCs by this shifted egocentric
% representation then updates the mendial temporal representation to by
% consistent with the (egocentric) parietal representation.
%
%
% Schematic:
%
% BVCs activity profile before translational movement --> via BVCtoPW
% transformation + translation(!) weights --> translated PW repres. -->
% back to BVC via normal transformation weigths  -->  BVC activity profile
% after translation. 
%
% Comment:
% 
% This was necessary (in the BBB model) for a form of self-projection in imagery, 
% but for real navigation ('eyes open'), it should not be used. The above quote means that 
% - in the old BBB (!) - there was no direct, continuously updating, perceptual drive 
% to the PW which can function without the MTL component.
%
% Implementation:
% 
% X,Y will be given from a trajectory update. Then we calculate
% from X,Y the distance to all landmarks in front of the agent
% (meaning determined by its egocentric location), i.e. landmarks in its
% field of view centered on HD. The return variable egocue will then be
% the input for the cueing current, with the help of the function.
% BB_subr_cue_PWact_multiseg(egocue). That is, this script effectively
% computes into which neurons of the PW sheet current needs to be
% injected to represent the local sensory environment (sensory drive)



total_lines = line;                   
occ_one     = ones(total_lines,1);
TrainX      = X;                      % location at which we calculate the PW current
TrainY      = Y;
NTrainPts   = 1;                      % size(TrainX,1); % only 1 training location here 



num_bndry_pts = size(BndryPtX,1);
bpone         = ones(num_bndry_pts,1);                  % helper arrays
bpzero        = zeros(num_bndry_pts,1);                 % ...
VisXY         = NaN*ones(2,num_bndry_pts,NTrainPts);    % NaN*ones(2,num_bndry_pts,NTrainPts); 

% This determines which boundary/line points can be seen from a given
% training location

posx                     = TrainX;
posy                     = TrainY;                                   % xy coords of current training loc.
TrainingLoc              = [posx posy];                              
local_r0                 = r0 - occ_one*[posx posy 0];               % Transform all boundary coordinates so that current training loc. is origin
Loc_bndry_pts            = [BndryPtX-posx BndryPtY-posy bpzero];     % adding thrid dim with 0s because fct crossprod in routing intersection needs 3D vectors
occluded_pts             = bpzero;

if ~obj
    for occ_bndry = 1:total_lines
                     [alpha_pt,alpha_occ] = ...
        BB_prep_subr_intersection([bpzero bpzero bpzero],bpone*local_r0(occ_bndry,:), ...
                     Loc_bndry_pts,bpone*dir(occ_bndry,:));                                                % The intersection routine figures out where two lines intersect and returns both alphas
        occluded_pts = occluded_pts | ( alpha_pt<1-10^(-5) & alpha_pt>0 & alpha_occ<=1 & alpha_occ>=0 );   % This variable accumulates boundary pts that are occluded by some other boundary from current training loc.
    end
end

unocc_ind                = find(occluded_pts==0);                                                % Indices of bndry points that are not occluded, subset of all pts. 
num_vis_pts              = prod(size(unocc_ind));                                                % ...
VisXY(:,1:num_vis_pts,1) = Loc_bndry_pts(unocc_ind,1:2)' + [posx posy]'*ones(1,num_vis_pts);     % xy coords of unoccluded bndry pts - Accumulates over all locations - transforms back to previous origin

VisX = permute(VisXY(1,:,:),[3,2,1]); % Puts x,y coords of unoccluded pts in separate variables
VisY = permute(VisXY(2,:,:),[3,2,1]);

posX = TrainingLoc(1,1);                          % Get x,y coords for this location
posY = TrainingLoc(1,2);

tmp               = find(isnan(VisX(1,:))==0);    % Clean up some arrays. This will remove trailing NaN entries
VisBndryPtsX      = VisX(1,tmp)-posX;             % Get coords of bndry points visible from current location
VisBndryPtsY      = VisY(1,tmp)-posY;

% rotate and translate unoccluded boundary pts to match agent rotation
if ~obj
    R = [cos(HD) -sin(HD); sin(HD)  cos(HD)];           % set up rotation matrix
else
    R = [cos(HD) -sin(HD); sin(HD)  cos(HD)];           % ...
    R = R';
end

object_vectors        = [VisBndryPtsX; VisBndryPtsY];   % array of vectors, note, object refers to boundary elements here, don't confuse with discrete objects
rotated_vectors       = R * object_vectors;             % rotation
VisBndryPtsX_rottrans = rotated_vectors(1,:);   
VisBndryPtsY_rottrans = rotated_vectors(2,:);  



% now we want all boundary pts with Y'>0, corresponding to a 180° field of view

ind1 = VisBndryPtsY_rottrans>0;      % select pts then use diff to separate landmark segments

diffvec = [diff(VisBndryPtsX(ind1))' diff(VisBndryPtsY(ind1))'];   % use unrotated
Lmarks  = [];                                                      % setup variable for output
FOVy    = VisBndryPtsY(ind1);                                      % use unrotated counterparts   
FOVx    = VisBndryPtsX(ind1);

c2=1;
counter=1;
while c2<length(find(ind1))    
    diffsum = round((abs(diffvec(c2,1))+abs(diffvec(c2,2)))*10);
    if diffsum==3-obj                                                  % depends on res 3
        Lmarks = [Lmarks [FOVx(c2) ; FOVy(c2)]];                       % enter start pt of landmark
        while diffsum==3-obj && c2+counter<length(find(ind1)) 
            diffsum = round((abs(diffvec(c2+counter,1))+abs(diffvec(c2+counter,2)))*10);
            counter=counter+1;
        end
        Lmarks = [Lmarks [FOVx(c2+counter-1) ; FOVy(c2+counter-1)]];   % enter end pt of landmark
    end
    c2 = c2+counter;
    counter=1;
end

if isempty(Lmarks)
    egocue     = [];
    Lmarks_rot = [];
else
    Lmarks_rot = R*Lmarks;   % rotate according to agent orientation
    egocue  = [];
    counter = 1;
    for i = 1:length(Lmarks_rot(1,:))/2
        egocue = [egocue ; reshape(Lmarks_rot(:,counter:counter+1),[1,4])];
        counter = counter + 2;
    end
    
end



