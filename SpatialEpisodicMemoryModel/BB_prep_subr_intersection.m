
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


function [alpha1,alpha2] = BB_prep_subr_intersection(r0_1,r0_2,dir1,dir2)


% Figures out how far along each line two lines intersect.

den2 = cross(dir1,dir2);
NaN_ind = find(den2==0);
den2(NaN_ind) = NaN;
den1 = -den2;
alpha2 = cross(r0_2-r0_1,dir1)./den2;
alpha1 = cross(r0_1-r0_2,dir2)./den1;

% cross product will tell you how this calculation works

alpha1 = alpha1(:,3);
alpha2 = alpha2(:,3);

