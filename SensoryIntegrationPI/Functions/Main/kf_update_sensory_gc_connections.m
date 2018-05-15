function s = kf_update_sensory_gc_connections(stats,par,s)

% Record the change in weights
s.pc.dw = s.pc.w;

switch lower(par.learn.rule)
    
    case 'none'
        s = s;
    case 'bcm'
        % The BCM rule:
        % tau_w* dw   /dt = v*u*(v-Th_v) ___(1)
        % tau_Th*dTh_v/dt = v^2 - Th_v   ___(2)
        % 
        % Note that tau_Th > tau_W, since the change in the threshold Th
        % needs to happen faster that the rate at which the weights change
        %
        tau_w =  1/par.learn.pc.rate; % Learning time constant (inverse of learning-rate)
        tau_Th = tau_w/100; % Must be > tau_w (see Dayan and Abbott, BCM rule
        
        % Pre- and post-synaptic firing are u and v respectively, where we
        % assume the grid cells to be the presynaptic cells
        
        %% PC->GC 
        s.pc.w    = par.learn.wact(s.pc.w    + (par.learn.dt/tau_w) * bsxfun(@times,(s.pc.F.*(s.pc.F - s.pc.Th_v)),s.bys.p.P_pos'));
        
        % Calculate the threshold, which varies throughout the trial, by tracking change using an adaptive threshold                
        s.pc.Th_v = s.pc.Th_v + (par.learn.dt/tau_Th)*(s.pc.F.^2-s.pc.Th_v);
        
        % Threshold the weights to be positive
        %s.pc.w = (s.pc.w>0).*s.pc.w;
        
                
        %% GC->PC
        %s.pc.w2   = par.learn.wact(s.pc.w2    + (par.learn.dt/tau_w) * bsxfun(@times,(s.pc.F.*(s.pc.F - s.pc.Th_v2)),s.bys.p.P_pos'));
        
        % Calculate the threshold, which varies throughout the trial, by tracking change using an adapttive threshold                
        %s.pc.Th_v = s.pc.Th_v + (par.learn.dt/tau_Th)*(s.pc.F.^2-s.pc.Th_v);
        
        % Threshold the weights to be positive
        %s.pc.w = (s.pc.w>0).*s.pc.w;
        
    case 'squared_loss'
        %{
        % Adjust the error term between the sensory and PI distributions
        % 
        % L = || G - (W*P) / ||W*P|| ||
        % 
        % Where ||x|| = Sum(x.^2) = x'*x
        
        t_0 = np.dot(W, P)
        t_1 = np.linalg.norm(t_0, 1)
        t_2 = (G - ((1 / t_1) * t_0))
        t_3 = np.linalg.norm(t_2)
        functionValue = t_3
        
        
        gradient = -(((1 / (t_1 * t_3)) * np.multiply.outer(t_2, P)) - (((1 / ((t_1 ** 2) * t_3)) * np.dot(P, np.dot(W.T, t_2))) * np.multiply.outer(np.sign(t_0), P)))
%}
        
    case 'oja'
        % The Oja Rule:
        %
        % dw/dt = v*u - alpha*v^2*w
        %
        % Where alpha is the learning rate
        %
        % The Oja rule applies multiplicative normalisation. The SSQ of the
        % weights is constrained to tend->1.
        s.pc.w = par.learn.wact(...
            s.pc.w + bsxfun(@times,s.pc.F,s.bys.p.P_pos') + bsxfun(@times,par.learn.pc.rate*par.learn.dt.*s.pc.F.^2,s.pc.w));

end

        
%% Record the change in weights
s.pc.dw = s.pc.w - s.pc.dw;

end