%% Get a measurement of location from the sensory inputs

function s = kf_get_measurement_from_sensory(par,s)

% Simply sum the inputs
%s.pc.gc_u = s.pc.F'*s.pc.w;

% Multiplication (or, summing the log-probability and then exponentiate)
%s.pc.gc_u = exp(sum(log(bsxfun(@times,s.pc.w,s.pc.F)),1));
s.pc.gc_u = bsxfun(@times,s.pc.w,s.pc.F); % Get the input to each of the grid cells
%s.pc.gc_u = bsxfun(@times,s.pc.gc_u,1./(mean(s.pc.gc_u,2)+eps)); % Make sure that each PC has a total input of 1 across all grid cells
%s.pc.gc_u = prod(s.pc.gc_u,1);
s.pc.gc_u = sum(s.pc.gc_u,1);

s.pc.gc_u = s.pc.gc_u / sum(s.pc.gc_u(:)); % Normalise

%% Normalise the sensory input accoridng to its importance weighting
% A measurement of no importance is equivalent to a convolution by a ones
% matrix. To do this, we add 0.5 then take some root (according to the 
% weighting parameter) such that all values are squeezed around 1. Then,
% subtract 0.5 again. Finally, divide by the sum to make it a probability
% distribution again.
s.pc.gc_u = s.pc.gc_u / sum(s.pc.gc_u);

end