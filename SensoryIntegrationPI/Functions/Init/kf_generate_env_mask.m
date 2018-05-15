% Generate the environment mask

function par = kf_generate_env_mask(par)

switch lower(par.opts.env_type)
    % No environmental features
    case 'empty'
        par.plot.env_mask = zeros(par.plot.gridsize)';
        par.plot.env_subareas{1}.x = 1:size(par.plot.env_mask,2);
        par.plot.env_subareas{1}.y = 1:size(par.plot.env_mask,1);
        
        % Francis' two-compartment environment
    case 'two_compartment'
        
        XG = reshape(par.plot.grid(:,1),par.plot.gridsize)';
        YG = reshape(par.plot.grid(:,2),par.plot.gridsize)';
        GR = par.plot.grid_res;
        
        msk = zeros(par.plot.gridsize)';
        
        xc = 0.5;
        EWw = 0.4/1.3;
        DW = 0.1/1.8;
        
        msk(round(YG/par.Y/GR)*GR>=EWw & round(XG/par.X/GR)*GR == 0.5) = 1; % The N-S dividing wall between the two walls
        msk(round(YG/par.Y/GR)*GR==round(EWw/GR)*GR) = 1; % The E-W wall leading to the corridor
        msk(round(XG/par.X/GR)*GR>=(0.25-DW/2) & round(XG/par.X/GR)*GR<=(0.25+DW/2)) = 0; % Create a doorway in the left compartment
        msk(round(XG/par.X/GR)*GR>=(0.75-DW/2) & round(XG/par.X/GR)*GR<=(0.75+DW/2)) = 0; % Create a doorway in the right compartment
        
        msk(:,1)   = 1; % West wall
        msk(:,end) = 1; % East wall
        msk(1,:)   = 1; % North wall
        msk(end,:) = 1; % South wall
        
        par.plot.env_mask = msk;
        
        % Top corridor
        par.plot.env_subareas{1}.x = 1:par.plot.gridsize(1);
        par.plot.env_subareas{1}.y = 1:round(EWw/GR+1);
        
        % Left box
        par.plot.env_subareas{2}.x = 1:round(0.5*par.plot.gridsize(1)+1);
        par.plot.env_subareas{2}.y = round(EWw/GR+2):par.plot.gridsize(2);
        
        % Right box
        par.plot.env_subareas{3}.x = round(0.5*par.plot.gridsize(1)+2):par.plot.gridsize(1);
        par.plot.env_subareas{3}.y = round(EWw/GR+2):par.plot.gridsize(2);

        % Francis' two-compartment environment
    case 'circle'
        
        XG = reshape(par.plot.grid(:,1),par.plot.gridsize)';
        YG = reshape(par.plot.grid(:,2),par.plot.gridsize)';
        GR = par.plot.grid_res;
        
        par.plot.env_mask = ones(par.plot.gridsize)';    
        
        R = min([par.X,par.Y])/2;
        XC = [par.X/2,par.Y/2];
        
        spaceInd = sqrt((XG-XC(1)).^2 + (YG-XC(2)).^2)<=R;
        
        par.plot.env_mask(spaceInd) = 0;
                
        par.plot.env_subareas{1}.x = 1:size(par.plot.env_mask,2);
        par.plot.env_subareas{1}.y = 1:size(par.plot.env_mask,1);
        
case 'trapezoid'
        
        XG = reshape(par.plot.grid(:,1),par.plot.gridsize)';
        YG = reshape(par.plot.grid(:,2),par.plot.gridsize)';
        
        par.plot.env_mask = ones(par.plot.gridsize)';    
        
        R = min([par.X,par.Y])/2;
        XC = [par.X/2,par.Y/2];
        
        % Define polygon
        
        % Long/short axis ratio = 0.9/0.2
        small_end = par.Y*(0.2/0.9);
        
        % Define bounding polygon
        xq = [0                  ,par.X,par.X,0                  ,0                  ];
        yq = [par.Y/2+small_end/2,par.Y,0    ,par.Y/2-small_end/2,par.Y/2+small_end/2];
        
        % Points in polygon are free space
        par.plot.env_mask(inpolygon(XG,YG,xq,yq))=0;
                                
        par.plot.env_subareas{1}.x = 1:size(par.plot.env_mask,2);
        par.plot.env_subareas{1}.y = 1:size(par.plot.env_mask,1);
        
        % Add another type here...
    case 'derdikman type, for example...'
        
end

end