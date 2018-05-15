%% Update the process noise
%
% The process noise accounts for the noise associated with uncertain PI
% update (control input). This can be seen from the following:
% 
% https://stats.stackexchange.com/questions/134920/kalman-filter-with-input-control-noise
%
% The process noise may change at each timestep, and may be a function of
% e.g. position
%
% The process noise is assumed to be independent of previous values

function Q = kf_update_process_noise(S,par)

Q = S.Q;

end

