
function [xnew,par] = kf_spiral(xold,par)

xold = xold(:);

xold(1:2) = xold(1:2) - [par.X/2;par.Y/2];

if ~isfield(par.traj,'R')
    par.traj.R0 = 0.99*max([par.X/2,par.Y/2]);
    par.traj.R = par.traj.R0;
    par.traj.pn = -1;
    par.traj.Th = 0; 
    xold = [sqrt(abs(cos(par.traj.Th))).* par.traj.R .* sign(cos(par.traj.Th)); sqrt(abs(sin(par.traj.Th))).* par.traj.R .* sign(sin(par.traj.Th))];
    par.traj.Rstep = par.traj.R*par.traj.vel/20;
end
    
    par.traj.R = par.traj.R + par.traj.pn*par.traj.Rstep;
    
    if (par.traj.R<(par.traj.vel)) 
        par.traj.pn = -par.traj.pn;
        par.traj.R = par.traj.vel;
    end
    
    if (par.traj.R>par.traj.R0)
        par.traj.pn = -par.traj.pn;
        par.traj.R = par.traj.R0;
    end
    
    ds=0;
    Tnew = par.traj.Th;
    DS = 2*par.traj.vel*rand();
    while (ds<DS) 
        Tnew = mod(Tnew+par.traj.vel/100,2*pi);
    
        xnew(1) = sqrt(abs(cos(Tnew))).* par.traj.R .* sign(cos(Tnew));
        xnew(2) = sqrt(abs(sin(Tnew))).* par.traj.R .* sign(sin(Tnew));
        
        ds = sqrt((xnew(1)-xold(1)).^2 + (xnew(2)-xold(2)).^2);
    end
    
    par.traj.Th = Tnew;
    
    xnew = xnew(:);
    
    xnew = xnew + [par.X/2;par.Y/2];
        
end