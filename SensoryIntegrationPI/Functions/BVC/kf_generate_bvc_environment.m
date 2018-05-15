function par = kf_generate_bvc_environment(par)

switch lower(par.env.type)
    case 'walls'
        
        % Define the environment in which the animal is roaming, with 1s where
        % objects / boundaries exist
        par.env.env_mask = zeros(par.env.X/par.env.grid_res+1,...
                                 par.env.Y/par.env.grid_res+1);
        
        % Set the edges to 1
        par.env.env_mask(1,:)=1; par.env.env_mask(end,:)=1; par.env.env_mask(:,1)=1; par.env.env_mask(:,end)=1; 
        
        % Generate a grid of coordinates
        par.env.yvec = 0:par.env.grid_res:par.env.Y;
        par.env.xvec = 0:par.env.grid_res:par.env.X;
        [ym,xm] = meshgrid(par.env.yvec,par.env.xvec);
        
        % Characteristic width of a boundary element
        dx = par.env.xvec(2)-par.env.xvec(1);
        dy = par.env.yvec(2)-par.env.yvec(1);
        par.env.dEl = sqrt(dx^2+dy^2);
        
        % Coordinates of the boundary elements
        par.env.x_bound = xm(par.env.env_mask==1); % Only pick out the coordinates where
        par.env.y_bound = ym(par.env.env_mask==1); % an element of boundary exists
        
        % Convert to gpu arrays if required
        if par.opts.gpu_option
            FN = fieldnames(par.env);
            for f=1:length(FN)
                if isnumeric(par.env.(FN{f}))
                    par.env.(FN{f}) = gpuArray(par.env.(FN{f}));
                end
            end
        end
        
    case 'another_type_eg_with_objects'
        
end

end