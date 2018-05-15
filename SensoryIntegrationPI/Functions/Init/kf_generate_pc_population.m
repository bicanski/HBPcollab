function par = kf_generate_pc_population(par)

% Defaults
par.pc.sort_inds = 1:par.pc.npc;

% Generate the centers of the place cells
switch lower(par.pc.distribution)
    case 'uniform'
        % Uniform distribution of PC centers
        par.pc.mu = (rand(2,par.pc.npc).*[par.X;par.Y])';
        par.pc.sort_inds = 1:par.pc.npc;
    case 'grid'
        % A uniform grid of place cells maybe
        nx = round(sqrt(par.X/par.Y*par.pc.npc));
        ny = round(par.pc.npc/nx);
        [ym,xm] = meshgrid(linspace(0,par.Y,ny),linspace(0,par.X,nx));
        par.pc.mu = [xm(:),ym(:)];
        par.pc.npc = size(par.pc.mu,1);
        par.pc.sort_inds = 1:par.pc.npc;
    case 'boundaries'
        % Place cells distributed mainly near the boundaries perhaps
    case 'bvc'
        % Generate PC from BVC model
        par = kf_generate_pcs_from_bvcs(par);
        %MU = par.pc.sort_coords2D;
        %MU = bsxfun(@plus,MU,-min(MU,[],1));
        %MU = bsxfun(@times,MU,1./max(MU,[],1));
        %par.pc.mu = MU;
        nx = ceil(sqrt(par.X/par.Y*par.pc.npc));
        ny = ceil(par.pc.npc/nx);
        [ym,xm] = meshgrid(linspace(0,par.Y,ny),linspace(0,par.X,nx));
        par.pc.mu = [xm(:),ym(:)];
        par.pc.mu = par.pc.mu(1:par.pc.npc,:);
        par.pc.mu = par.pc.mu(par.pc.sort_inds,:);
    case 'nonuniform'
        % Non-uniform distribution of PC centres
        gr = 0.01; % Default initial discretization resolution
        par.pc.mu = nonuniform_sample(par.pc.distribution_func.f,par.pc.distribution_func.params,par.X,par.Y,gr,par.pc.npc);
        par.pc.sort_inds = 1:par.pc.npc;
    case 'predefined'
        % Do nothing
        par.pc.npc = size(par.pc.mu,1);
        par.pc.sort_inds = 1:par.pc.npc;
        
end

if ~strcmpi(par.pc.distribution,'bvc')
    if ~strcmpi(par.pc.distribution,'predefined')
        
        % Only keep those cells that the animal can travel to, if required
        if par.pc.only_within_boundaries
            MU = floor(par.pc.mu/par.plot.grid_res)+1; % Convert to matrix coordinates
            MUind=nan(1,length(MU));
            for p=1:length(MU)
                MUind(p)=~par.plot.env_mask(MU(p,2),MU(p,1));
            end
            par.pc.mu = par.pc.mu(find(MUind),:);
            par.pc.npc = sum(MUind);
            [~,par.pc.sort_inds] = sort(par.pc.sort_inds(find(MUind))); % This is aslightly confusing operation: basically, it does the equivalent of reordering a sequence like [1,4,5,3] to [1,3,4,2], since we're thrhowing away some of the entries
        end
        
    end
    
    % Generate the covariance matrices of the place cells
    switch lower(par.pc.tuning)
        case 'constant'
            % Shape of PCs is uniform across the environment
            par.pc.C = repmat(par.pc.C0*[1,0;0,1],1,1,par.pc.npc);
        case 'boundaries'
            % PCs may get wider away from the walls, for example
    end
    
end

% Convert to gpuArrays if required
if par.opts.gpu_option
    par.pc.mu = gpuArray(par.pc.mu);
    par.pc.C  = gpuArray(par.pc.C);
end

% Compute pairwise distances in real and grid space
par.pc.Dpw0_euc = sqrt( (par.pc.mu(:,1)-par.pc.mu(:,1)').^2 + (par.pc.mu(:,2)-par.pc.mu(:,2)').^2 );

par.pc.hex_coords = wrap_xy_in_hex(par.pc.mu',0,par.grid.scale,0)';

%gm = par.plot.grid_baseSheet';
%d = (hex_coords(1,:)-gm(1,:)').^2 + (hex_coords(2,:)-gm(2,:)').^2;
%[~,ind] = min(d,[],1);
%hex_coords = gm(:,ind);

par.pc.Dpw0_hex = kf_hex_dist(...
    par.pc.hex_coords,...
    par.pc.hex_coords,...
    par.grid.scale);


end

% Generate sample over an arbitrary PDF
function mu = nonuniform_sample(pdf,pdf_params,X,Y,gr,nsamples)

% Generate discretized sample
xvec = 0:gr:X; yvec = 0:gr:Y;
[xm,ym] = meshgrid(xvec,yvec);

% Generate new samples
mu = zeros(nsamples,2);
for s=1:nsamples
    [mu(s,1),mu(s,2)] = pinky(xvec,yvec,pdf(pdf_params,xm,ym));
end

end