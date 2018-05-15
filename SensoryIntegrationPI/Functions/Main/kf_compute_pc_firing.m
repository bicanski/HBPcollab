% Compute the pc firing of the population of PCs at each iteration
%
% PCs are assumed to be responding to sensory input only, so they fire in
% reponse to the true position of the animal

function F = kf_compute_pc_firing(S,par)

if strcmpi(par.pc.distribution,'bvc')
    B = bvc_func(S.X(:)',par.bvc);
    F = par.bvc.W_BVC2PC'*B + S.pc.w*S.bys.p.P_pos; % Input from both BVCs and GCs
    F(F<par.bvc.thresh)=0;
else
    F = bvnpdf(S.X(:)',par.pc.mu,par.pc.C) .* par.pc.fmax;
end

end