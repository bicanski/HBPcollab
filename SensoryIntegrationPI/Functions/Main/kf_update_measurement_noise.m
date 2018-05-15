%% Update the measurement noise
%
% The measurement noise may be a function of the current position, for
% example
%
% The measurement noise is assumed to be independent of previous values

function R = kf_update_measurement_noise(S,par)

R = S.R;

end