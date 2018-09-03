
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


function [Activity] = BB_subr_cue_HDact(bump_locs)


% bump_locs is the initial HD, i.e. the location of the activity bump

NHD = 100;
sig = 0.1885;
amp = 1;

% bump_locs must be from 0 to 2*pi minus one neuron's resolution
bump_locs = NHD*bump_locs/(2*pi);
sig       = NHD*sig/(2*pi);
Activity  = zeros(NHD,1);

x                       = [1:1:NHD];
Wide_x(1,1:NHD)         = x-NHD;
Wide_x(1,NHD+1:2*NHD)   = x;
Wide_x(1,2*NHD+1:3*NHD) = x+NHD;

for bump = 1:size(bump_locs,2)
        x0 = bump_locs(bump);
        Gaussian = amp.*(exp(-((Wide_x-x0)/sig).^2) + ...
                   exp(-((Wide_x-x0-NHD)/sig).^2) + ...
                   exp(-((Wide_x-x0+NHD)/sig).^2));
        Activity = Activity + Gaussian(1,NHD+1:2*NHD)';
end



