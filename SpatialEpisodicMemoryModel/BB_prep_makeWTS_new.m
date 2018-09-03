

% Subroutine for the BB model of spatial cogniton (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)


% This script takes care of the geometry of the environment, and generates
% desired weights. 


GEOMflag = 0;  % set to 1 to setup new geometry (prereq. for some other calculations)
MTLflag  = 0;  % set to 1 to make MTL weights,
HDflag   = 0;  % ... make HD weights
TRflag   = 1;  % ... make transformation weights

% choose type of environment, don't forget to update namestring below
% accordingly
two_room           = 0;  % an alternative environment from the original BBB model
square_room        = 1;  % main environment from the article
inserted_barrier   = 0;  % for simflag 2X
preplay_env_closed = 0;  % for simflag 50
preplay_env_open   = 0;  % ...

%namestrg = 'inserted_barrier_BVCcorr';  % name this geometry/setup
%namestrg = 'Two_Room';
%namestrg = 'square_room';
namestrg = 'square_room_BVCcorr';
%namestrg = 'preplay_env_closed_EXT_BVCcorr';
%namestrg = 'preplay_env_open_EXT_BVCcorr';



tic;

if GEOMflag
    
    % Read environment specific data:
    % START OF envdata_Two_Room
    
    if two_room
        
        % These two variables set up the corners of a grid of possible exploration points:
        maxXY = 12.5;
        minXY = -12.5;
        
        % These four set up the exact rectangle, points within which will actually be used for training the model
        minTrainX = -12.5;
        minTrainY = -12.5;
        maxTrainX = 12.5;
        maxTrainY = 6.25;
        
        % These variable contain the boundaries of the "buildings" in the environment
        maxNObjPoints = 5;
        nObjects = 6;
        polyn = nObjects;
        objectX = zeros(maxNObjPoints,nObjects);
        objectY = zeros(maxNObjPoints,nObjects);
        
        nVertices(1,1) = 5;
        objectX(1:5,1) = [-10 -0.8 -0.8 -10 -10]';   % each boundary is a four-sided polygon. Start pt = end pt
        objectY(1:5,1) = [4.05 4.05 5 5 4.05]';
        
        nVertices(2,1) = 5;
        objectX(1:5,2) = [-0.5 0.5 0.5 -0.5 -0.5]';
        objectY(1:5,2) = [-2.5 -2.5 5 5 -2.5]';
        
        nVertices(3,1) = 5;
        objectX(1:5,3) = [0.8 10 10 0.8 0.8]';
        objectY(1:5,3) = [4.05 4.05 5 5 4.05]';
        
        nVertices(4,1) = 5;
        objectX(1:5,4) = [9.1 10 10 9.1 9.1]';
        objectY(1:5,4) = [3.7 3.7 -10 -10 3.7]';
        
        nVertices(5,1) = 5;
        objectX(1:5,5) = [8.8 8.8 -8.8 -8.8 8.8]';
        objectY(1:5,5) = [-9.1 -10 -10 -9.1 -9.1]';
        
        nVertices(6,1) = 5;
        objectX(1:5,6) = [-9.1 -10 -10 -9.1 -9.1]';
        objectY(1:5,6) = [-10 -10 3.7 3.7 -10]';
        
        objectX=1.25*objectX;   % scale to full width after coords were between -10 an 10
        objectY=1.25*objectY;
    end
    
    if square_room
        
        Xrange   = 22;
        Yrange   = 22;

        % These two variables set up the corners of a grid of possible exploration points:
        maxXY = Xrange; 
        minXY = 0; 
        
        % These four set up the exact rectangle, points within which will actually be used for training the model
        minTrainX = 0.0;
        minTrainY = 0.0;
        maxTrainX = Xrange;
        maxTrainY = Yrange;
        
        % These variable contain the boundaries of the "buildings" in the environment
        maxNObjPoints = 5;
        nObjects = 4;                           
        polyn = nObjects;
        
        objectX = zeros(maxNObjPoints,nObjects);
        objectY = zeros(maxNObjPoints,nObjects);
        
        nVertices(1,1) = 5;
        objectX(1:5,1) = [   1      Xrange-1   Xrange-1    1       1    ]';    % North wall
        objectY(1:5,1) = [Yrange-1  Yrange-1   Yrange   Yrange  Yrange-1]';
        
        nVertices(2,1) = 5;
        objectX(1:5,2) = [Xrange-1  Xrange  Xrange    Xrange-1  Xrange-1]';    % West wall
        objectY(1:5,2) = [  1          1    Yrange-1  Yrange-1      1   ]';
        
        nVertices(3,1) = 5;
        objectX(1:5,3) = [  1    Xrange-1   Xrange-1    1     1    ]';         % South Wall
        objectY(1:5,3) = [  0       0          1        1     0    ]';
        
        nVertices(4,1) = 5;
        objectX(1:5,4) = [  0      1       1         0          0   ]';        % East Wall
        objectY(1:5,4) = [  1      1    Yrange-1  Yrange-1      1   ]';
        
    end
    
    if inserted_barrier
        
        Xrange   = 22;
        Yrange   = 22;

        % These two variables set up the corners of a grid of possible exploration points:
        maxXY = Xrange;  
        minXY = 0; 
        
        % These four set up the exact rectangle, points within which will actually be used for training the model
        minTrainX = 0.0;
        minTrainY = 0.0;
        maxTrainX = Xrange;
        maxTrainY = Yrange;
        
        % These variable contain the boundaries of the "buildings" in the environment
        maxNObjPoints = 5;
        nObjects = 5;                             % 4 walls for now. maybe treat bed etc also as boundary?
        polyn = nObjects;
        objectX = zeros(maxNObjPoints,nObjects);
        objectY = zeros(maxNObjPoints,nObjects);
        
        nVertices(1,1) = 5;
        objectX(1:5,1) = [   1      Xrange-1   Xrange-1    1       1    ]';    % North wall
        objectY(1:5,1) = [Yrange-1  Yrange-1   Yrange   Yrange  Yrange-1]';
        
        nVertices(2,1) = 5;
        objectX(1:5,2) = [Xrange-1  Xrange  Xrange    Xrange-1  Xrange-1]';    % West wall
        objectY(1:5,2) = [  1          1    Yrange-1  Yrange-1      1   ]';
        
        nVertices(3,1) = 5;
        objectX(1:5,3) = [  1    Xrange-1   Xrange-1    1     1    ]';         % South Wall
        objectY(1:5,3) = [  0       0          1        1     0    ]';
        
        nVertices(4,1) = 5;
        objectX(1:5,4) = [  0      1       1         0          0   ]';        % East Wall
        objectY(1:5,4) = [  1      1    Yrange-1  Yrange-1      1   ]';
        
        nVertices(5,1) = 5;
        objectX(1:5,5) = [  8    14   14    8     8    ]';         % middle barrier
        objectY(1:5,5) = [  8     8    9    9     8    ]';
          
    end
    
    if preplay_env_closed
        
        %load('roomGridPrePlay_ext.mat');
        load('roomGridPrePlay.mat');
        
        red_grid = roomGridPrePlay(2:end-1,2:end-1);
        
        RoomSize = size(roomGridPrePlay);
        Xrange   = RoomSize(1,1);
        Yrange   = RoomSize(1,2);
        
        Xbarrier_top_min = find(red_grid(Yrange/2,:)==1,1,'first')+1; % max is at wall
        Ybarrier_top = find(red_grid(:,end)==1,1,'last')+1;
        Xbarrier_bot_min = Xbarrier_top_min;  % max is at wall
        Ybarrier_bot = find(red_grid(:,end)==1,1,'first')+1;
        
        Xrange   = 22;
        Yrange   = 22;

        % These two variables set up the corners of a grid of possible exploration points:
        maxXY = Xrange;  
        minXY = 0; 
        
        % These four set up the exact rectangle, points within which will actually be used for training the model
        minTrainX = 0.0;
        minTrainY = 0.0;
        maxTrainX = Xrange;
        maxTrainY = Yrange;
        
        % These variable contain the boundaries of the "buildings" in the environment
        maxNObjPoints = 5;
        nObjects = 8;                            
        polyn = nObjects;
        objectX = zeros(maxNObjPoints,nObjects);
        objectY = zeros(maxNObjPoints,nObjects);
        
        nVertices(1,1) = 5;
        objectX(1:5,1) = [   1      Xrange-1   Xrange-1    1       1    ]';    % North wall
        objectY(1:5,1) = [Yrange-1  Yrange-1   Yrange   Yrange  Yrange-1]';
        
        nVertices(2,1) = 5;
        objectX(1:5,2) = [0   1   1          0          0]';                   % West Wall
        objectY(1:5,2) = [1   1   Yrange-1   Yrange-1   1]';
        
        nVertices(3,1) = 5;
        objectX(1:5,3) = [  1    Xrange-1   Xrange-1    1     1    ]';         % South Wall
        objectY(1:5,3) = [  0       0          1        1     0    ]';
        
        nVertices(4,1) = 5;
        objectX(1:5,4) = [Xrange-1       Xrange         Xrange     Xrange-1   Xrange-1]';    % East wall above barrier
        objectY(1:5,4) = [Ybarrier_top   Ybarrier_top   Yrange-1   Yrange-1   Ybarrier_top]';
        
        nVertices(5,1) = 5;
        objectX(1:5,5) = [   Xbarrier_top_min+1  Xrange-1          Xrange-1        Xbarrier_top_min+1    Xbarrier_top_min+1]';      % Barrier North
        objectY(1:5,5) = [   Ybarrier_top-1      Ybarrier_top-1    Ybarrier_top    Ybarrier_top          Ybarrier_top-1]';
        
        nVertices(6,1) = 5;
        objectX(1:5,6) = [Xbarrier_top_min   Xbarrier_top_min+1   Xbarrier_top_min+1   Xbarrier_top_min   Xbarrier_top_min]';       % Barrier West
        objectY(1:5,6) = [Ybarrier_top-1+1   Ybarrier_top-1+1     Ybarrier_bot+1-1     Ybarrier_bot+1-1   Ybarrier_top-1+1]';
        
        nVertices(7,1) = 5;
        objectX(1:5,7) = [Xbarrier_top_min+1  Xrange-1       Xrange-1         Xbarrier_top_min+1  Xbarrier_top_min+1]';             % Barrier South
        objectY(1:5,7) = [Ybarrier_bot        Ybarrier_bot   Ybarrier_bot+1   Ybarrier_bot+1      Ybarrier_bot]';
        
        nVertices(8,1) = 5;
        objectX(1:5,8) = [Xrange-1   Xrange   Xrange         Xrange-1       Xrange-1]';    % East wall below barrier
        objectY(1:5,8) = [1          1        Ybarrier_bot   Ybarrier_bot   1   ]';
             
    end
    
    if preplay_env_open
        
        %load('roomGridPrePlay_ext.mat');
        load('roomGridPrePlay.mat');
        
        red_grid = roomGridPrePlay(2:end-1,2:end-1);
        
        RoomSize = size(roomGridPrePlay);
        Xrange   = RoomSize(1,1);
        Yrange   = RoomSize(1,2);
        
        Xbarrier_top_min = find(red_grid(Yrange/2,:)==1,1,'first')+1; % max is at wall
        Ybarrier_top = find(red_grid(:,end)==1,1,'last')+1;
        Xbarrier_bot_min = Xbarrier_top_min;  % max is at wall
        Ybarrier_bot = find(red_grid(:,end)==1,1,'first')+1;
        
        Xrange   = 22;
        Yrange   = 22;

        % These two variables set up the corners of a grid of possible exploration points:
        maxXY = Xrange;  
        minXY = 0; 
         
        % These four set up the exact rectangle, points within which will actually be used for training the model
        minTrainX = 0.0;
        minTrainY = 0.0;
        maxTrainX = Xrange;
        maxTrainY = Yrange;
       
        % These variable contain the boundaries of the "buildings" in the environment
        maxNObjPoints = 5;
        nObjects = 7;                             % 4 walls for now. maybe treat bed etc also as boundary?
        polyn = nObjects;
        objectX = zeros(maxNObjPoints,nObjects);
        objectY = zeros(maxNObjPoints,nObjects);
         
        nVertices(1,1) = 5;
        objectX(1:5,1) = [   1      Xrange-1   Xrange-1    1       1    ]';    % North wall
        objectY(1:5,1) = [Yrange-1  Yrange-1   Yrange   Yrange  Yrange-1]';
        
        nVertices(2,1) = 5;
        objectX(1:5,2) = [0   1   1          0          0]';                   % West Wall
        objectY(1:5,2) = [1   1   Yrange-1   Yrange-1   1]';
   
        nVertices(3,1) = 5;
        objectX(1:5,3) = [  1    Xrange-1   Xrange-1    1     1    ]';         % South Wall
        objectY(1:5,3) = [  0       0          1        1     0    ]';
               
        nVertices(4,1) = 5;
        objectX(1:5,4) = [Xrange-1       Xrange         Xrange     Xrange-1   Xrange-1]';    % East wall above barrier
        objectY(1:5,4) = [Ybarrier_top   Ybarrier_top   Yrange-1   Yrange-1   Ybarrier_top]';
        
        nVertices(5,1) = 5;
        objectX(1:5,5) = [Xbarrier_top_min     Xbarrier_top_min+1     Xbarrier_top_min+1     Xbarrier_top_min     Xbarrier_top_min]';        % Barrier West
        objectY(1:5,5) = [Ybarrier_top-1+1     Ybarrier_top-1+1       Ybarrier_bot+1-1       Ybarrier_bot+1-1     Ybarrier_top-1+1   ]';
        
        nVertices(6,1) = 5;
        objectX(1:5,6) = [Xrange-1   Xrange  Xrange   Xrange-1   Xrange-1]';              % Barrier WestWest
        objectY(1:5,6) = [Ybarrier_bot       Ybarrier_bot         Ybarrier_top         Ybarrier_top       Ybarrier_bot   ]';
        
        nVertices(7,1) = 5;
        objectX(1:5,7) = [Xrange-1   Xrange   Xrange         Xrange-1       Xrange-1]';    % East wall below barrier
        objectY(1:5,7) = [1          1        Ybarrier_bot   Ybarrier_bot   1   ]';
        
    end
    
        
        
    figure;plot(objectX,objectY);    
        
        
   
    % Each building has a unique numerical identity, or texture One more for the background (which isn't used) and one
    % more for the two-room env. because one "wall" has two identities (one for each side).
    
    if two_room
        nTextures = polyn + 2;                                        % polyn = nObjects; in envdata_Two_Room
    end
    if square_room
        nTextures = polyn;
    end
    if inserted_barrier
        nTextures = polyn + 1;                                        % polyn = nObjects; in envdata_Two_Room
    end
    if preplay_env_closed
        nTextures = polyn;                                            % polyn = nObjects; in envdata_Two_Room
    end
    if preplay_env_open
        nTextures = polyn;                                            % polyn = nObjects; in envdata_Two_Room
    end
    
    res           = 0.3;                                              % Resolution of the grid of possible training locations
    gridx         = [minXY:res:maxXY];                                % Create cartesian grid of possible locations over env.
    gridy         = [minXY:res:maxXY];
    [GRIDX,GRIDY] = meshgrid(gridx,gridy);                            % Forms matrices with repeated arrays of the vectors gridx,gridy
    COMPLEXGRID   = GRIDX + i * GRIDY;                                % Convert cartesian coordinates to complex numbers
    COMPLEXGRID   = reshape(COMPLEXGRID,prod(size(COMPLEXGRID)),1);   % Vector of grid points:  x & y values (complex):

    % This for loop breaks all buildings up into sets of lines/boundaries, each associated with the building's identity:
    
    line = 0;            % Each "building" in the environment is composed of boundaries or "lines" (4 each for TR env.)
    ForgroundPts = [];   % Will be the possible locations from above that are inside buildings

    for poly=1:polyn     % Loop over polygons (buildings), polyn = nObjects in env_Two_Room
        
        Vertices     = [];
        Vertices     = [objectX(:,poly)' + i*objectY(:,poly)']';
        Vertices     = Vertices(1:nVertices(poly,1));                                                        % Complex vector of x,y coords of the building vertices
        inPolyPts    = find(inpolygon(real(COMPLEXGRID),imag(COMPLEXGRID),real(Vertices),imag(Vertices)));   % Locations inside this building
        ForgroundPts = [ForgroundPts [inPolyPts]'];                                                          % Locations inside this building, foreground in the sense of looking at the map from above, the ground is background, buildings are foreground
        
        % Loop over "lines" of building (same as # of vertices)
        % Subloop turns lines into start pt and vector and associates identity
        for polyline = 1:nVertices(poly,1)-1
            
            line = line + 1;
            xi   = objectX(polyline,poly);     % ***, see below
            xf   = objectX(polyline+1,poly);
            yi   = objectY(polyline,poly);
            yf   = objectY(polyline+1,poly);
            
            % Special for TwoRoom and inserted barrier because one bld has two identities. This associates identity (stored in line_tex)
            
            if two_room
                if poly >= 3
                    line_tex(line) = poly+1;
                elseif  poly == 2
                    if xf > 0 & xi > 0;
                        line_tex(line) = poly+1;
                    else
                        line_tex(line) = poly;
                    end
                else
                    line_tex(line) = poly;
                end
            end
            
            if square_room || preplay_env_open || preplay_env_closed
                line_tex(line) = poly;
            end
            
            if inserted_barrier
                if poly >= 5
                    line_tex(line) = poly+1;
                elseif  poly == 5
                    if xf == 14 && xi == 8;
                        line_tex(line) = poly;
                    end
                    if xf == 8 && xi == 14;
                        line_tex(line) = poly+1;
                    end
                else
                    line_tex(line) = poly;
                end
            end
            
            dir(line,:) = [xf-xi yf-yi 0];   % line vectors, from one vertex of a bld to the next, see ***, but only from start to end, discretized below
            r0(line,:) = [xi yi 0];          % Line start
            
        end
        
    end
    
    % The above lines basically create the coordinates of all the vertices that
    % make up a bld, and then go on to create all the line segments of one bld
    % vertex to the next. Each line segments is assigned an identity of the texture, 
    % with teh special case of building two, where it depends on the side. 
    % The "points within polygon" for each building are used below.

    total_lines = line;                                      % was loop variable above
    occ_one     = ones(total_lines,1);
    
    BakgrndX               =  GRIDX;
    BakgrndY               =  GRIDY;%-GRIDY;                 % x & y values of locations outside of buildings:
    BakgrndX(ForgroundPts) = NaN;                            % Convert the elements which lie within the
    BakgrndY(ForgroundPts) = NaN;                            % buildings into non-numbers
    BakgrndX               = BakgrndX(isfinite(BakgrndX));   % Remove the non-numbers from the arrays -
    BakgrndY               = BakgrndY(isfinite(BakgrndY));   % forms a column vector
    
    train_ind = find(BakgrndX>minTrainX&BakgrndX<maxTrainX&BakgrndY>minTrainY&BakgrndY<maxTrainY);   % Indices of locations outside of bldgs, but inside training rect.:
    TrainX    = BakgrndX(train_ind);                                                                 % Actual coords
    TrainY    = BakgrndY(train_ind);
    
    clear BakgrndX BakgrndY
    
    NTrainPts = size(TrainX,1);
    
    BB_plt_lineplot(r0,dir)   % Plot lines
    plot(TrainX,TrainY,'*')   % Plot training locations

    % This loop turns the building lines or boundaries into discrete points. (Discretizes it)
    strt = 1;
    for bndry = 1:total_lines                            % total_lines is the sum of the loop counter further above, total number of line segments, multiple segments to one bld wall. counter goes over all buildings
        
        x         = [];
        y         = [];
        Bndry_len = norm(dir(bndry,:));                  % Length of line
        Dir_unit  = dir(bndry,:)/Bndry_len;              % Unit vector along line
        alpha     = 0;                                   % alpha is how far along line, ranges zero to one
        
        while alpha < Bndry_len;
            x     = [x r0(bndry,1)+alpha*Dir_unit(1)];   % x & y are coords of points
            y     = [y r0(bndry,2)+alpha*Dir_unit(2)];
            alpha = alpha + res;                         % Boundary points are 'res' apart
        end
        
        stp                        = prod(size(x)) + strt - 1;                
        BndryPtX(strt:stp,1)       = x';                                      % These variables accumulate the x & y coords of all bndry pts, a vecctor of growing length
        BndryPtY(strt:stp,1)       = y';
        Bndry_Textures(strt:stp,1) = (ones(1,stp-strt+1)*line_tex(bndry))';   % This variable has associated identities
        strt                       = stp + 1;
        
    end

    num_bndry_pts = size(BndryPtX,1);
    bpone         = ones(num_bndry_pts,1);                  % helper arrays
    bpzero        = zeros(num_bndry_pts,1);                 % ...
    VisXY         = NaN*ones(2,num_bndry_pts,NTrainPts);    % ...
    Texture       = NaN*ones(NTrainPts,num_bndry_pts);

    % This loop determines which boundary/line points can be seen from a given training location
    for location=1:NTrainPts   % Loops over locations
        
        if mod(location,10) == 0;
            location  % Prints out which location the algorithm is currently at (so I know it is doing something!)
        end
        
        posx                    = TrainX(location);
        posy                    = TrainY(location);                       % xy coords of current training loc.
        TrainingLoc(location,:) = [posx posy];                            % This variable will be saved to the training data file
        local_r0                = r0 - occ_one*[posx posy 0];             % Transform all boundary coordinates so that current training loc. is origin
        Loc_bndry_pts           = [BndryPtX-posx BndryPtY-posy bpzero];   % adding thrid dim with 0s because fct crossprod in routing intersection needs 3D vectors
        occluded_pts            = bpzero;
        
        % This loop figures out which boundary pts are occluded by other boundaries relative to the origin (current training loc)
        for occ_bndry = 1:total_lines;
                         [alpha_pt,alpha_occ] = ...
            BB_prep_subr_intersection([bpzero bpzero bpzero], ...
                         bpone*local_r0(occ_bndry,:),Loc_bndry_pts,bpone*dir(occ_bndry,:));   % The intersection routine figures out where two lines intersect and returns both alphas
            occluded_pts         = occluded_pts | ( alpha_pt<1-10^(-5) & alpha_pt>0 & alpha_occ<=1 & alpha_occ>=0 );      % This variable accumulates boundary pts that are occluded by some other boundary from current training loc.
        end
        
        unocc_ind                       = find(occluded_pts==0);                                              % Indices of bndry points that are not occluded
        num_vis_pts                     = prod(size(unocc_ind));
        VisXY(:,1:num_vis_pts,location) = Loc_bndry_pts(unocc_ind,1:2)' + [posx posy]'*ones(1,num_vis_pts);   % xy coords of unoccluded bndry pts - Accumulates over all locations - transforms back to previous origin
        Texture(location,1:num_vis_pts) = Bndry_Textures(unocc_ind,1)';
        
    end

    VisX = permute(VisXY(1,:,:),[3,2,1]); % Puts x,y coords of unoccluded pts in separate variables
    VisY = permute(VisXY(2,:,:),[3,2,1]);
    
    clear VisXY
    
    n = size(TrainingLoc,1);
    
    % This loop randomly arranges training locations
    for row = 1:n
        
        swaprow                = ceil(n * rand(1,1));
        tempTL                 = TrainingLoc(row,:);
        tempVX                 = VisX(row,:);
        tempVY                 = VisY(row,:);
        tempTX                 = Texture(row,:);
        TrainingLoc(row,:)     = TrainingLoc(swaprow,:);
        VisX(row,:)            = VisX(swaprow,:);
        VisY(row,:)            = VisY(swaprow,:);
        Texture(row,:)         = Texture(swaprow,:);
        TrainingLoc(swaprow,:) = tempTL;
        VisX(swaprow,:)        = tempVX;
        VisY(swaprow,:)        = tempVY;
        Texture(swaprow,:)     = tempTX;
        
    end

    % Saving
    savename = ['save TrainingData_' namestrg ' TrainingLoc VisX VisY Texture nTextures minTrainX minTrainY maxTrainX maxTrainY'];
    eval(savename);
    
    toc;
    disp('done setting up geometry');
    
    
    
    % This for loop breaks all buildings up into sets of lines/boundaries, each associated with the building's identity:
    
    line = 0;                                      % Each "building" in the environment is composed of boundaries or "lines" (4 each for TR env.)
    for poly=1:polyn                               % Loop over polygons (buildings), polyn = nObjects above
        for polyline = 1:nVertices(poly,1)-1       % Subloop turns lines into start pt and vector and associates identity
            line = line + 1;
            xi   = objectX(polyline,poly);         % ***, see below
            xf   = objectX(polyline+1,poly);
            yi   = objectY(polyline,poly);
            yf   = objectY(polyline+1,poly);
            dir(line,:) = [xf-xi yf-yi 0];         % line vectors, from one vertex of a bld to the next, see ***, but only from start to end, discretized below
            r0(line,:) = [xi yi 0];                % Line start
        end
    end
    total_lines = line;
    
    % The above lines basically create the coordinates of all the vertices that
    % make up a bld, and then go on to create all the line segments of one bld
    % vertex to the next (later used to see if interesected?). Each line
    % segments is assigned an identity of the texture, with the special case of
    % building two, where it depends on the side. The "points within polygon"
    % for each building are used below.
    
    % This loop turns the building lines or boundaries into discrete points. (Discretizes it)
    strt = 1;
    for bndry = 1:total_lines                            % total_lines is the sum of the loop counter further above, total number of line segments, multiple segments to one bld wall. counter goes over all buildings
        x         = [];
        y         = [];
        Bndry_len = norm(dir(bndry,:));                  % Length of line
        Dir_unit  = dir(bndry,:)/Bndry_len;              % Unit vector along line
        alpha     = 0;                                   % alpha is how far along line, ranges zero to one
        while alpha < Bndry_len;
            x     = [x r0(bndry,1)+alpha*Dir_unit(1)];   % x & y are coords of points
            y     = [y r0(bndry,2)+alpha*Dir_unit(2)];
            alpha = alpha + res;                         % Boundary points are 'res' apart
        end
        stp                        = prod(size(x)) + strt - 1;                
        BndryPtX(strt:stp,1)       = x';                       % These variables accumulate the x & y coords of all bndry pts, a vecctor of growing length
        BndryPtY(strt:stp,1)       = y';
        strt                       = stp + 1;
    end
    
    savename = ['save Roomdataforperceptiondrive_' namestrg ' r0 dir line BndryPtX BndryPtY'];
    eval(savename);
    toc;
    disp('done setting up room data for perception');

    clearvars -except GEOMflag MTLflag HDflag TRflag two_room square_room inserted_barrier namestrg

    pause(5);
    tic;
    
end



% From here on we make weights



% This part generates the MTL weights. That is the weights from BVCs and
% PCs, from PCs to BVCs, from BVCs to perirhinal neurons (PRNs), from PRNs to BVCs,
% from PRNs to PCs, from PCs to PRNs. among PRNs, among BVCs, and among PCs
%
% it uses the info about unoccluded landmarks (VisX,VisY) at training locations

if MTLflag
    
    loadname = ['load TrainingData_' namestrg ''];        % Read in the training data
    eval(loadname);

    Hres         = 0.5;       % Place cells on grid with 2 cells per unit length
    maxR         = 16;        % BVCs to cover space up to 16 units from origin
    maxX         = 12.5;
    maxY         = 6.25;
    minX         = -12.5;
    minY         = -12.5;
    polarDistRes = 1;         % One BVC per unit radial length, only used for number of radial steps, not scaling
    polarAngRes  = 2*pi/51;   % Angular separation of neurons on radial grid
    HSig         = 0.5;       % Width of desired/training activity hill
    
    % Assign cartesian coords to each HPC neuron. Numbering moves along x-dir.
    NHx   = round((maxTrainX-minTrainX)/Hres);                  % Num of H neurons in x-dir
    NHy   = round((maxTrainY-minTrainY)/Hres);                  % " " in y-dir
    NH    = NHx*NHy;                                            % Total H neurons
    x     = [minTrainX+Hres/2:Hres:minTrainX+(NHx-0.5)*Hres];
    y     = [minTrainY+Hres/2:Hres:minTrainY+(NHy-0.5)*Hres];
    [X,Y] = meshgrid(x,y);                                      % x,y cords for all H neurons
    Hx    = reshape(X',prod(size(X)),1);                        % Same thing, but as column vector
    Hy    = reshape(Y',prod(size(Y)),1);
    
    % Assign polar coordinates to each BVC neuron. Numbering moves outward along radius, starting at 0 radians
    NBVCR        = round((maxR)/polarDistRes);                               % Num BVCs along a radius
    NBVCTheta    = floor((2*pi-0.01)/polarAngRes)+1;                         % Num BVCs in a ring
    NBVC         = NBVCR*NBVCTheta;                                          % Total number
    %polarDist    = [polarDistRes/2:polarDistRes:(NBVCR-0.5)*polarDistRes];
    [polarDist]  = BB_subr_radscaling(maxR);
    
    polarAng     = [0:polarAngRes:(NBVCTheta-1)*polarAngRes];
    [pDist,pAng] = meshgrid(polarDist,polarAng);                             % polar coords of all BVC neurons
    BVCDist      = reshape(pDist',prod(size(pDist)),1);                      % Same, but in column vector
    BVCAng       = reshape(pAng',prod(size(pAng)),1);
    BVCAng       = BVCAng - 2*pi*(BVCAng > pi);                              % Make sure angles in correct range
    
    % Perirhinal neuron and possible activations setup
    NPR           = nTextures;   % One perirhinal neuron for each identity/texture
    PRactivations = eye(NPR);    % identity matrix
    
    % Create HPC to HPC weights (each neuron excites neighbors in proportion to a Gaussian of separation
    
    H_XSeparations = ones(NH,1)*reshape(X',1,NH)-reshape(X',NH,1)*ones(1,NH);   % x separations between all pairs of H neurons
    H_YSeparations = ones(NH,1)*reshape(Y',1,NH)-reshape(Y',NH,1)*ones(1,NH);   % y " " " "
    H_SqDistances  = (H_XSeparations.^2+H_YSeparations.^2);
    H2Hwts         = exp(-H_SqDistances./HSig^2);                               % H2Hwts are easy to calculate!
    
    % Now must loop over Training locations. This is mainly for the BVC connections, since HPC already calculated
    
    BVC2Hwts  = zeros(NH,NBVC);                                                                   % Initialize weight matrices
    BVC2PRwts = zeros(NPR,NBVC);
    PR2Hwts   = zeros(NH,NPR);
    H2PRwts   = PR2Hwts';
    
    for location = 1:size(TrainingLoc,1)                                                          % Loop over training locations
        
        disp(location);                                                                           % Let me know if you are doing something!
        posX = TrainingLoc(location,1);                                                           % Get x,y coords for this location
        posY = TrainingLoc(location,2);
        
        tmp               = find(isnan(VisX(location,:))==0);                                     % Clean up some arrays. This will remove trailing NaN entries
        VisBndryPtsX      = VisX(location,tmp)-posX;                                              % Get coords of bndry points visible from current location
        VisBndryPtsY      = VisY(location,tmp)-posY;
        TextureOfBndryPts = Texture(location,tmp);                                                % Get identities of each
        
        [VisBndryPtsTheta,VisBndryPtsR]               = cart2pol(VisBndryPtsX,VisBndryPtsY);      % Switch to polar coords
        VisBndryPtsR(find(VisBndryPtsR<polarDistRes)) = polarDistRes;                             % Don't let any bndry pts be closer than this to origin/self	or things blow-up!
        
        Hact = exp(-((Hx-posX).^2+(Hy-posY).^2)/HSig^2);                                          % What place cell activity should be at this location
        
        % Next, the BVC and PR layer activation correlations are used to generate remaining weights. Must loop over visible boundary points
        % Note that texture cells are associated with BVC cells as attention is payed to each visible boundary point sequentially.
        % This could be done for BVC2H associations as well. The results would, however, be the same as assuming all BVC cells are 
        % activated at once and associated with corresponding H cells.
        % We do the latter here for speed of calculation.
        
        BVCact = zeros(NBVC,1);
        
        contribs = zeros(NBVC,1);
        
        for bndryPtNum = 1:prod(size(VisBndryPtsX))                                       % Loop over/attend to each bndryPt in sequence
            
            AngDiff1  = abs(BVCAng-VisBndryPtsTheta(bndryPtNum));                         % These three lines find smallest angular separation
            AngDiff2  = 2*pi-abs(-BVCAng+VisBndryPtsTheta(bndryPtNum));                   % between the bndryPt and each radial grid point (neuron)
            AngDiff   = (AngDiff1<pi).*AngDiff1 + (AngDiff1>pi).*AngDiff2;
            sigmaTH   = sqrt(0.05);
            sigmaR0   = 0.08;
            sigmaR    = (VisBndryPtsR(bndryPtNum)+8)*sigmaR0;
            DelBVCact = 1/VisBndryPtsR(bndryPtNum) * ( exp(-(AngDiff/sigmaTH).^2 )  .*  exp( - ( (BVCDist-VisBndryPtsR(bndryPtNum) )./sigmaR ) .^2 )).*(BVCact<=1);
            BVCact    = BVCact + DelBVCact;                                                       % Accumulates BVC activity
            BVC2PRwts = BVC2PRwts + PRactivations(:,TextureOfBndryPts(bndryPtNum))*DelBVCact';    % Updates BVC2PRwts after each event
            H2PRwts   = H2PRwts + PRactivations(:,TextureOfBndryPts(bndryPtNum))*Hact';           % Update H2PRwts
            
        end
        
        BVC2Hwts = BVC2Hwts + Hact*BVCact';                                                       % These done all at once as discussed above
        
    end
    
    % activations are effectively used as weights
    
    H2BVCwts  = BVC2Hwts';                                                             % These are easy to calculate now!
    PR2BVCwts = BVC2PRwts';
    PR2Hwts   = H2PRwts';
    
    BVC2Hwts  = BVC2Hwts./(sum(BVC2Hwts,2)*ones(1,size(BVC2Hwts,2)));                  % Post-synaptic ``normalization''. Sum of weights terminating on single neuron is one
    H2BVCwts  = H2BVCwts./(sum(H2BVCwts,2)*ones(1,size(H2BVCwts,2)));

    BVC2PRwts  = BVC2PRwts./((sum(BVC2PRwts,2)+0.000001)*ones(1,size(BVC2PRwts,2)));   % The small number is to avoid infinity for the unused backgroundneuron
    PR2BVCwts  = PR2BVCwts./(sum(PR2BVCwts,2)*ones(1,size(PR2BVCwts,2)));
    H2PRwts    = H2PRwts./((sum(H2PRwts,2)+0.000001)*ones(1,size(H2PRwts,2)));
    PR2Hwts    = PR2Hwts./(sum(PR2Hwts,2)*ones(1,size(PR2Hwts,2)));
    PR2PRwts   = zeros(NPR,NPR);
    BVC2BVCwts = zeros(NBVC,NBVC);
    
    savename = ['save Weights_VR_MTL_' namestrg ' BVC2Hwts H2BVCwts BVC2PRwts PR2BVCwts PR2Hwts H2PRwts PR2PRwts BVC2BVCwts H2Hwts'];
    eval(savename);
    
    clearvars -except GEOMflag MTLflag HDflag TRflag two_room square_room inserted_barrier namestrg
    
    toc;
    disp('done creating MTL weights');
    pause(5);
    tic;
    
end



% This part generates the HD weights.

if HDflag
    
    NHD      = 100;                                                    % 100 Head-direction neurons
    amp      = 1;                                                      % Max weight value
    sig      = 0.1885;                                                 % 0.1885 radians corresponds to 3 neurons!
    sig      = NHD*sig/(2*pi);                                         % Neuron number measure as opposed to radian measure
    HD2HDwts = zeros(NHD,NHD);
    
    x                       = [1:1:NHD];
    Wide_x(1,1:NHD)         = x-NHD;                                   % A more complicated than necessary way to implement periodic bndy conditions
    Wide_x(1,NHD+1:2*NHD)   = x;
    Wide_x(1,2*NHD+1:3*NHD) = x+NHD;
    
    for x0 = 1:NHD                                                     % Create weights for each neuron
        
        Gaussian       = amp.*(  exp(-((Wide_x-x0)/sig).^2) ...
            + exp(-((Wide_x-x0-NHD)/sig).^2) ...
            + exp(-((Wide_x-x0+NHD)/sig).^2) );
        HD2HDwts(:,x0) = Gaussian(1,NHD+1:2*NHD)';
        
    end
    
    save HDWeights HD2HDwts
    clear all
    
    toc;
    disp('done creating HD-HD weights');
    pause(5);
    tic;

    % Next, create the rotational integrator head-direction weights
    % Does so by performing a Hebbian learning between current head-dir activity
    % and a sliding, weighted window of past HD activity
    
    NSteps = 100;                               % Num. of training steps
    dt     = 0.05;                              % A time-step
    NHD    = 100;                               % Number of head direction neurons
    sig    = 0.1885;                            % 0.1885 rads is about 3 neurons
    sig    = NHD*sig/(2*pi);                    % Switch to neuron measure
    record = zeros(NHD,100);                    % This is the window, which contains a record of the past 100 steps of activity
    decay  = 1;                                 % Weight/Importance of past activity decays with time within the sliding window
    
    Rotwts                  = zeros(NHD,NHD);
    x                       = [1:1:NHD];
    Wide_x(1,1:NHD)         = x-NHD;            % Odd way to implement periodic bndry conditions
    Wide_x(1,NHD+1:2*NHD)   = x;
    Wide_x(1,2*NHD+1:3*NHD) = x+NHD;
    
    rec_ind = 1;
    
    for step = 1:NSteps
        
        disp(step);                                                      % Display which step we are on
        x0  = 2*pi*rand;                                                 % Random location to put current bump of HD activity
        vel = 1*rand+0.5;                                                % Pick a random velocity for this step
        
        for time = 0:dt:2*pi/abs(vel)                                    % Move bump once around full rotation at current velocity
            xt   = x0+time*vel;                                          % Update bump position
            xt   = xt-2*pi*(xt>2*pi);
            xt   = NHD*xt/(2*pi);
            
            Gaussian = ( exp(-((Wide_x-xt)/sig).^2) ...
                + exp(-((Wide_x-xt-NHD)/sig).^2) ...
                + exp(-((Wide_x-xt+NHD)/sig).^2) );
            record             = record*(1-dt/decay);                    % Perform decay on window
            current_activation = Gaussian(1,NHD+1:2*NHD)';               % Calculate current activity
            
            Rotwts = Rotwts + current_activation*sum(record');           % Do Hebbian update between current activity and sliding window
            
            record(:,rec_ind) = current_activation;                      % Put current activation in sliding window
            rec_ind           = rec_ind+1-100*(rec_ind>100);             % Update where we are in window
        end
        
        if mod(step,20)==0
            Rotwts = Rotwts./(ones(NHD,1)*max(Rotwts));                  % Normalize occasionally as in Stringer et al.
        end
        
    end
    
    Rotwts = Rotwts./(ones(NHD,1)*max(Rotwts));                          % Do final normalization
    
    save RotIntWeights Rotwts
    
    clearvars -except GEOMflag MTLflag HDflag TRflag two_room square_room inserted_barrier namestrg
    
    
    toc;
    disp('done creating HD weights for angular PI');
    pause(5);
    tic;
    
end



% Make transformation circuit weights 

if TRflag
    
    Hres         = 0.5;       % Place cells on grid with 2 cells per unit length
    maxR         = 16;        % BVCs to cover space up to 16 units from origin
    maxX         = 12.5;
    maxY         = 6.25;
    minX         = -12.5;
    minY         = -12.5;
    polarDistRes = 1;         % One BVC per unit radial length
    polarAngRes  = 2*pi/51;   % Angular separation of neurons on radial grid
    HSig         = 0.5;       % Width of desired/training activity hill
    
    N_HD     = 100;      % Number of HD neurons
    NSteps   = 400000;   % Lots of training steps!
    HDActSig = 0.1885;
    
    % Env. edge resolution (i.e. How dense are wall segment edges?)
    TRres = pi/10;                  % There will be 20 TR sublayers - these are for the rotated versions of the environment
    res   = 0.2;                    % Line segment discretization resolution
    
    % Each TR layer corresponds to one unit of TRres (measured in radians)
    N_TR     = floor(2*pi/TRres);           % Needs to be this many uncommented lines of separate TR entries before the save command at the end of the file.
    TRangles = [0:TRres:(N_TR-1)*TRres];
    
    % How many BVC neurons? (Will be same as num of PW neurons and each TR sublayer)
    NBVCR     = round((maxR)/polarDistRes);
    NBVCTheta = floor((2*pi-0.01)/polarAngRes)+1;
    NBVC      = NBVCR*NBVCTheta;

    % Empty weight matricies
    BVC2TRwts = zeros(NBVC,NBVC);          % Assume BVC and TR layers have same res!!!! And that same weights are used to connect BVC to all TR layers.
    TR2PWwts  = zeros(NBVC,NBVC,N_TR);     % Assume TR and PW layers have same res!!!!
    HD2TRwts  = zeros(NBVC,N_HD,N_TR);
    PW2TRwts  = zeros(NBVC,NBVC,N_TR);
       
    for count = 1:NSteps

        if mod(count,100)==0
            disp(count);
        end
        
        % Generate random edge, first: pick a starting point
        thetaI = 2*pi*rand;
        DistI  = maxR*rand;
        xi     = DistI*cos(thetaI);       % x,y coords of beginning of line seg. in allo ref frame
        yi     = DistI*sin(thetaI);        
        % Next, end point:
        thetaF = 2*pi*rand;
        xf     = xi+DistI*cos(thetaF);    % x,y coords of end point
        yf     = yi+DistI*sin(thetaF);
        
        % Generate the BVC grid activity for the edge
        BVCrate = BB_subr_cue_PWact_multiseg(res,[xi yi xf yf]);
        
        % Pick random HD from TRangles. We only train these directions and hope the intermediate ones ``fill in'' correctly.
        TRlayer = floor(N_TR*rand+1);
        HD      = TRangles(TRlayer);      % In training we only use twenty head directions (corresponding to each TR sublayer. Model generalizes to other HDs
        
        % Generate rotated edge by rotating coords of initial/final points
        rxi = xi*cos(HD)+yi*sin(HD);      % Now we have line coords in ego frame
        ryi = -xi*sin(HD) + yi*cos(HD);
        rxf = xf*cos(HD)+yf*sin(HD);
        ryf = -xf*sin(HD) + yf*cos(HD);
        
        % Generate ego layer (PW) act.
        PWrate = BB_subr_cue_PWact_multiseg(res,[rxi ryi rxf ryf]);

        % Weight Updates
        if mod(count,20)==0
            BVC2TRwts = BVC2TRwts + BVCrate*BVCrate';                        % These weights will be the same for each TR sublayer, therefore they will get 20 times as much training!
        end
        TR2PWwts(:,:,TRlayer) = TR2PWwts(:,:,TRlayer) + PWrate*BVCrate';     % ...
        
    end
    

    % we can generate HD2TR weights algorithmically
    for i = 1:N_TR
        HD                  = TRangles(i);
        HDrate              = BB_subr_cue_HDact(HD);
        HDrate(HDrate<0.01) = 0;   
        HDrate              = sparse(HDrate);
        HD2TRwts(:,:,i)     = ones(816,1)*HDrate';
    end
    
   
    % save in case want to change something below
    save TRtmp_new_BVCcorr 
    %load TRtmp_new 
 
    
    % Get remaining weights with transpose:
    for i = 1:20
        PW2TRwts(:,:,i) = TR2PWwts(:,:,i)';
    end
    TR2BVCwts = BVC2TRwts';
    
    
    % rescale all weigths to max 1 before clipping
    for i = 1:20
        PW2TRwts(:,:,i) = PW2TRwts(:,:,i)/max(max(PW2TRwts(:,:,i)));  % do this globally, not per column to preserve relative weight differences ontu target neurons (per row)
        TR2PWwts(:,:,i) = TR2PWwts(:,:,i)/max(max(TR2PWwts(:,:,i)));
        HD2TRwts(:,:,i) = HD2TRwts(:,:,i)/max(max(HD2TRwts(:,:,i)));
    end
    TR2BVCwts = TR2BVCwts/max(max(TR2BVCwts));
    BVC2TRwts = BVC2TRwts/max(max(BVC2TRwts));

    
    % clip and make sparse, HD2TR weights already clipped above 
    sparseness = 18000;
    BTclip = 0.01; % start by clipping 1 percent of weights relative to global max 1
    TPclip = 0.01;
    for i = 1:20
        TPspa = sum(sum(TR2PWwts(:,:,i)>0)); % sparseness
        while TPspa>sparseness
            TPclip          = TPclip + 0.01;
            TR2PWwts(:,:,i) = sparse(TR2PWwts(:,:,i).*(TR2PWwts(:,:,i)>TPclip));
            TPspa           = sum(sum(TR2PWwts(:,:,i)>0));
        end
        PW2TRwts(:,:,i) = sparse(PW2TRwts(:,:,i).*(PW2TRwts(:,:,i)>TPclip));
    end
    BTspa = sum(sum(BVC2TRwts>0)); % sparseness
    while BTspa>sparseness
        BTclip    = BTclip + 0.01;
        BVC2TRwts = sparse(BVC2TRwts.*(BVC2TRwts>BTclip));
        BTspa     = sum(sum(BVC2TRwts>0));
    end
    TR2BVCwts = sparse(TR2BVCwts.*(TR2BVCwts>BTclip));  
  
  
    % finally normalize such that sum of all weights targetting a given neuron is 1   
    %tmp = sum(TR2PWwts,3);
    for i = 1:20;
        divtmp1 = sum(TR2PWwts(:,:,i),2)*ones(1,NBVC);   divtmp1(divtmp1==0) = 1;   %catch NaNs
        divtmp2 = sum(PW2TRwts(:,:,i),2)*ones(1,NBVC);   divtmp2(divtmp2==0) = 1;
        %HD2TRwts(:,:,i) = HD2TRwts(:,:,i)./(sum(HD2TRwts(:,:,i),2)*ones(1,N_HD));
        TR2PWwts(:,:,i) = TR2PWwts(:,:,i)./divtmp1;
        PW2TRwts(:,:,i) = PW2TRwts(:,:,i)./divtmp2;
    end
    divtmp3   = sum(BVC2TRwts,2)*ones(1,NBVC);   divtmp3(divtmp3==0) = 1;   %catch NaNs
    divtmp4   = sum(TR2BVCwts,2)*ones(1,NBVC);   divtmp4(divtmp4==0) = 1;
    BVC2TRwts = BVC2TRwts./divtmp3;
    TR2BVCwts = TR2BVCwts./divtmp4;

   
    % export as single matrices and save fo use in BB_main
        eval(['PW2TRwts' num2str(i) ' = sparse(PW2TRwts(:,:,i));']);
        eval(['TR2PWwts' num2str(i) ' = sparse(TR2PWwts(:,:,i));']);
    end
    
    
    save TRWeights_NEW2_18kSP_BVCcorr BVC2TRwts TR2BVCwts HD2TRwts TR2PWwts1 TR2PWwts2 TR2PWwts3 TR2PWwts4 TR2PWwts5...
        TR2PWwts6 TR2PWwts7 TR2PWwts8 TR2PWwts9 TR2PWwts10 TR2PWwts11 TR2PWwts12 TR2PWwts13 ...
        TR2PWwts14 TR2PWwts15 TR2PWwts16 TR2PWwts17 TR2PWwts17 TR2PWwts18 TR2PWwts19 TR2PWwts20 ...
        PW2TRwts1  PW2TRwts2  PW2TRwts3 PW2TRwts4 PW2TRwts5 PW2TRwts6 PW2TRwts7 PW2TRwts8 ...
        PW2TRwts9 PW2TRwts10 PW2TRwts11 PW2TRwts12 PW2TRwts13 PW2TRwts14 PW2TRwts15 PW2TRwts16 ...
        PW2TRwts17 PW2TRwts18 PW2TRwts19 PW2TRwts20
    
    
    clearvars -except GEOMflag MTLflag HDflag TRflag two_room square_room inserted_barrier namestrg
    toc;
    disp('done creating Transformation weights, HD-TR, TR-BVC, BVC-TR, PW-TR, TR-PW');
    
end


