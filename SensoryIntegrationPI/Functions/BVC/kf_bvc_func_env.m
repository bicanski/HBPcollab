%% BVC firing function over entire environment or single point
%
% I wrote a second function, kf_bvc_func() to handle multiple bvcs in one
% more efficiently than this script using matrix operations instead of for
% loops. This is possuble in the other script since it only takes one query
% point at a time (but multiple BVCs)

% sigR(d) = ((d/beta) + 1)*sig0
%
% In Barry and Burgess (2007), values are sigA = 0.2 radians, beta = 1830 mm, and sig0 = 122 mm.
%
% The preferred firing direction (phi) for each BVC set was selected randomly from the continuous range 0�2pi.
%
% The preferred firing distance (d) for each BVC set was selected randomly from the following values:
% 81.0, 169.0, 265.0, 369.0, 482.5, 606.5, and 741.0 mm.
% It can be seen that BVCs with shorter preferred ?ring distance, and hence narrower tuning curves, are more densely represented.
%
% sigR - radial firing variance
% r - allocentric radial distance
% d - tuning distance
% the - current angle to boundary
% phi - tuning angle

% Simulate a multi-layered neural network that has head direction inputs at
% the bottom in addition to head direction sensitive BVC inputs. Hopefully,
% grid cells come out at the bottom that are directionally sensitive
% (conjunctive GCs), whereas they lose direction sensitivity as we go up
% the layers.
%
% In a biological sense, the directional sensitivity is inherent in the BVC
% representations since they're likely to be visually driven, and so it's
% necessary to include it in the model. However, it's not clear whether
% there is a functional advantage to directional snesitity in GCs if
% they're only meant as a coding mechanism. So, if they get more stable as
% we go from presubiculum (PrS) to parasubiculum (PaS) to mEC, they might
% lose their directional sensitivity.
%
% Environmental parameters to supply:
%
% env_mask : A square binary matrix containing 1s where a boundary exists, and 0s
% elsewhere
%
% Cell parameters to supply:
%
% d : The tuning distance
% sig0 : The basal tuning width (this is used to calculate sigR, which
% varies with distance)
% phi : The allocentric tuning direction

function F = kf_bvc_func_env(env_mask,d,phi,bvc_model,varargin)

% Defaults (chnaged by varargin parser)
query_pos = 0; % Unless querying a particular position calculate the firing map over all positions in the environment
gpu_option = 0;

%% Constant parameters (set according to literature)
sigA = 0.2;
sig0 = 122e-3;
beta = 1830e-3;

%% Parse optional arguments
arg=1; while arg<length(varargin) %#ok<ALIGN>
    switch lower(varargin{arg})
        case {'pos','position'}
            xpos = varargin{arg+1}(1);
            ypos = varargin{arg+1}(2);
            query_pos = 1;
            arg=arg+1;
        case 'gpu_option'
            gpu_option = varargin{arg+1};
            arg=arg+1;
    end
    arg=arg+1;
end

if strcmpi(bvc_model,'simple_exp_decay'); d = 0; end % have to set sigA to a very small value to prevent NaNs

%% Do main function
%% Generate firing map across environment unless otherwise specified
sigR = (d/beta + 1)*sig0;

if ~query_pos
    
    switch lower(bvc_model)
        case 'bvc_full'
            
            [NY,NX] = size(env_mask);
            
            xvec = linspace(0,1,size(env_mask,2));
            yvec = linspace(0,1,size(env_mask,1));
            
            dx = xvec(2)-xvec(1);
            dy = yvec(2)-yvec(1);
            dEl = sqrt(dx^2+dy^2); % Characteristic width of a boundary element
            
            
            if gpu_option
                
                [xm,ym] = meshgrid(xvec,yvec); % Generate a matrix of coordinates according to the env_map provided
                xm = gpuArray(xm); ym = gpuArray(ym);
                
                x_bound = gpuArray(permute(repmat(xm(env_mask==1),[1,NY,NX]),[2,3,1])); % Only pick out the coordinates where
                y_bound = gpuArray(permute(repmat(ym(env_mask==1),[1,NY,NX]),[2,3,1])); % an element of boundary exists
                
            else
                
                [xm,ym] = meshgrid(xvec,yvec); % Generate a matrix of coordinates according to the env_map provided
                
                x_bound = permute(repmat(xm(env_mask==1),[1,NY,NX]),[2,3,1]); % Only pick out the coordinates where
                y_bound = permute(repmat(ym(env_mask==1),[1,NY,NX]),[2,3,1]); % an element of boundary exists
                
            end
            
            xm = repmat(xm,[1,1,size(x_bound,3)]); % Repmat so we can do all the calculations in one big matrix operation later
            ym = repmat(ym,[1,1,size(y_bound,3)]); % These should have dimensions [NX,NY,numBoundEls]
            
            % Calculate the radial distance between the query points and the element of
            % boundary
            %r = sqrt( (xq-x_bound).^2 + (yq-y_bound).^2 );
            
            % Calculate the allocentric direction
            %dAng = -pi+mod(atan2(yq-y_bound,xq-x_bound)-phi+pi,2*pi);
            
            %F = exp( -(  r-d  ).^2 / (2*sigR^2) ) / sqrt(2*pi*sigR^2).*...
            %    exp( -(  dAng ).^2 / (2*sigA^2) ) / sqrt(2*pi*sigA^2);
            
            DX = xm-x_bound;
            DY = ym-y_bound;
            R = sqrt(DX.^2+DY.^2);
            
            % Angle that the element subtends is given by:
            % dTheta = 2*atan(dEl./(2*R)) where dx is the pixel width
            
            
            F = 2*atan(dEl./(2*R)).*...
                exp( -(  R-d  ).^2 / (2*sigR^2) ) .*...% / sqrt(2*pi*sigR^2).*...
                exp( -(  -pi+mod(atan2(DY,DX)-phi+pi,2*pi) ).^2 / (2*sigA^2) ); % / sqrt(2*pi*sigA^2);
            
        case 'simple_exp_decay'
            
            if gpu_option
                xvec = gpuArray(linspace(0,1,size(env_mask,2)));
                yvec = gpuArray(linspace(0,1,size(env_mask,1)));
                [xm,ym] = gpuArray(meshgrid(xvec,yvec)); % Generate a matrix of coordinates according to the env_map provided
            else
                xvec = linspace(0,1,size(env_mask,2));
                yvec = linspace(0,1,size(env_mask,1));
                [xm,ym] = meshgrid(xvec,yvec); % Generate a matrix of coordinates according to the env_map provided
            end
            
            if phi==0
                F = exp( -(xm).^2 ./ (2*sigR^2));
            elseif phi==(90*pi/180)
                F = exp( -(ym).^2 ./ (2*sigR^2));
            elseif phi==(180*pi/180)
                F = exp( -(1-xm).^2 ./ (2*sigR^2));
            elseif phi==(270*pi/180)
                F = exp( -(1-ym).^2 ./ (2*sigR^2));
            else F = zeros(size(xm));
            end
    end
    
    F = sum(F,3); % Sum the contributions of all cells
    
    % Set the points that are inside the boundary to a specific value. I
    % found that if I set them to zero, the decoding tends to predict
    % values inside the walls, as sometimes there are no spikes
    edgeVal = max(F(:));
    if prod(env_mask(1,:))==1; F(1,:)=edgeVal; end
    if prod(env_mask(end,:))==1; F(end,:)=edgeVal; end
    if prod(env_mask(:,1))==1; F(:,1)=edgeVal; end
    if prod(env_mask(:,end))==1; F(:,end)=edgeVal; end
    
    %% Only query one point
else
    
    switch lower(bvc_model)
        case 'bvc_full'
            
            if gpu_option
                xvec = gpuArray(linspace(0,1,size(env_mask,2)));
                yvec = gpuArray(linspace(0,1,size(env_mask,1)));
                
            else
                xvec = linspace(0,1,size(env_mask,2));
                yvec = linspace(0,1,size(env_mask,1));
            end
            
            dx = xvec(2)-xvec(1);
            dy = yvec(2)-yvec(1);
            dEl = sqrt(dx^2+dy^2); % Characteristic width of a boundary element
            
            [xm,ym] = meshgrid(xvec,yvec); % Generate a matrix of coordinates according to the env_map provided
            
            x_bound = xm(env_mask==1); % Only pick out the coordinates where
            y_bound = ym(env_mask==1); % an element of boundary exists
            
            DX = xpos-x_bound;
            DY = ypos-y_bound;
            R = sqrt(DX.^2+DY.^2);
            
            % Angle that the element subtends is given by:
            % dTheta = 2*atan(dEl./(2*R)) where dx is the pixel width

            F = 2*atan(dEl./(2*R)).*...
                exp( -(  R-d  ).^2 / (2*sigR^2) ) .*... / sqrt(2*pi*sigR^2).*...
                exp( -(  -pi+mod(atan2(DY,DX)-phi+pi,2*pi) ).^2 / (2*sigA^2) ); % / sqrt(2*pi*sigA^2);
            
        case 'simple_exp_decay'
            
            if phi==0
                F = exp( -(xpos).^2 ./ (2*sigR^2));
            elseif phi==(90*pi/180)
                F = exp( -(ypos).^2 ./ (2*sigR^2));
            elseif phi==(180*pi/180)
                F = exp( -(1-xpos).^2 ./ (2*sigR^2));
            elseif phi==(270*pi/180)
                F = exp( -(1-ypos).^2 ./ (2*sigR^2));
            else F = 0;
            end
    end
    
    F = sum(F); % Sum the contributions of all cells
    
end

F(isnan(F))=0;

F = gather(F);

end