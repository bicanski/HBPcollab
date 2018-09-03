
% Subroutine for the BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%
% Some code has been ported from a the predecessor of this model, the BBB model.
% (Patrick Byrne, Sue Becker and Neil Burgess, code available on request)

if simflag == 0   %%% for FR maps
    Xag              = 3; 
    Yag              = 3; 
    HDag             = 3*pi/2;          % initial HD
    Vlin             = 15.0;            % linear velocity start value, faster for better coverage for rate maps
    percep_flag      = 1;               % as long as this is 1 we are in bottom-up mode
    imag_flag        = 0;               % when this is 1, the imagery setup is used, top-down mode
    navi_flag        = 0;               % when 1 the agent starts to move
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;               % set to 1 after rotation is resolved, same at start
    target_no        = 0;               % index of next target, init with 0 
    tl = 1 + 20*rand(250,2);            % lots of random targets for this sim
    tl = round(tl*10)/10;
    targetlist = tl;
    SandT = 0;                          % "stop and think", engage top-down mode near trajectory point Xst,Yst
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP  = [13, 14; ...           
                  15,  7; ...
                   8, 8];               % object coordinates when not using externally supplied file
    Nobj         = length(CoordsTMP(:,1));                     
    ObjEncThresh = 0.000000001;         % small value essentailly prevents encoding in this sim
end

if simflag == 11   %%% Papez Circuit lesion
    HD2TRphi         = 0;   
    GC2PCphi         = 0;
    GC2PCphi_base    = 0;
    Xag              = 5; 
    Yag              = 10; 
    HDag             = pi;
    Vlin             = 2.5;             
    percep_flag      = 1;               
    imag_flag        = 0;              
    navi_flag        = 0;               
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;               
    targetlist = [ 5, 6; ...
                  10, 10; ...
                  16, 8; ...
                  16, 12];
    SandT = 3;                          
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [ 12, 12];               
    Nobj         = length(CoordsTMP(:,1));                   
    ObjEncThresh = 4.0;                    
end

if simflag == 12    %%% Papez Circuit lesion with preexisting memroy and removed object
    HD2TRphi         = 0;  
    GC2PCphi         = 0;
    GC2PCphi_base    = 0;
    Xag              = 5; 
    Yag              = 10; 
    HDag             = pi;
    Vlin             = 2.5;             
    percep_flag      = 1;               
    imag_flag        = 0;               
    navi_flag        = 0;               
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;               
    targetlist = [ 5, 6; ... 
                 10, 10; ...
                 16,  8; ...
                 16, 12];
    SandT = 3;                          
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [ 120, 120];               
    Nobj         = length(CoordsTMP(:,1));                 
    ObjEncThresh = 4.0;                    
end

if simflag == 241   %%% Novelty detection without HPC lesion
    Xag              = 10; 
    Yag              = 4;
    HDag             = 0;
    Vlin             = 5;            % a little faster becasue its a long sim.         
    percep_flag      = 1;               
    imag_flag        = 0;               
    navi_flag        = 0;               
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;               
    targetlist = [10  ,   5; ...
                  10  ,  13; ...
                  16  ,  13; ...
                  16  ,   4; ...
                  10  ,   4; ...
                  10  ,   5; ...
                  10   ,  13; ...    % this one gets overidden with the selection from the novelty signal
                  10  ,   14];
    SandT            = 6;              
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [7, 10; ...       % dummy
        13, 10];                 
    Nobj         = 2;                   
    ObjEncThresh = 4.0;
    novelty_sig  = 0;
    H2OVCphi   = 1.0;
end

if simflag == 242   %%% Novelty detection with HPC lesion
    Xag              = 10;
    Yag              = 4; 
    HDag             = 0;
    Vlin             = 5;               % ...
    percep_flag      = 1;              
    imag_flag        = 0;               
    navi_flag        = 0;              
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;               
    targetlist = [10  ,   5; ...
                  10  ,   6; ...
                  10  ,   9; ...
                  10  ,   10];
    SandT            = 1;               
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [7, 14; ...         % dummy
                   13, 10];                  
    Nobj         = 2;             
    ObjEncThresh = 4.0;
    novelty_sig  = 0;
    H2OVCphi   = 1.0;
end

if simflag == 21   %%% BVC traces (memory) of removed barrier
    Xag              = 5; 
    Yag              = 10; 
    HDag             = 0;
    Vlin             = 2.5;
    percep_flag      = 1;         
    imag_flag        = 0;              
    navi_flag        = 0;            
    move2nextTarget  = 0;      
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;             
    targetlist = [5  ,16; ...
                  8  ,16; ...
                 18  ,16; ...
                  6  ,17];
    SandT            = 2;             
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [];                  
    Nobj         = 0;                   
    ObjEncThresh = 4.0;
end

if simflag == 26   %%% same as simflag 21 but longer simulation without plotting to get nice rate maps
    Xag              = 5; 
    Yag              = 10;
    HDag             = 0;
    Vlin             = 2.5;
    percep_flag      = 1;              
    imag_flag        = 0;              
    navi_flag        = 0;              
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;            
    target_no        = 0;            
    tl = 1 + 20*rand(30,2);
    tl = round(tl*10)/10;
    targetlist   = tl;
    SandT        = 0;
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [];              
    Nobj         = 0;                   
    ObjEncThresh = 4.0;
end

if simflag == 27   %%% same as simflag 21 but even longer 
    Xag              = 5; 
    Yag              = 10; 
    HDag             = 0;
    Vlin             = 2.5;
    percep_flag      = 1;              
    imag_flag        = 0;              
    navi_flag        = 0;              
    move2nextTarget  = 0;             
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;       
    tl = 8.5 + 7*rand(100,2);
    tl = round(tl*10)/10;
    targetlist = tl;    
    SandT = 0; 
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [120, 120];                 
    Nobj         = length(CoordsTMP(:,1));               
    ObjEncThresh = 4.0;
end

if simflag == 22   %%% OVC traces (memory) of removed object
    Xag              = 5; 
    Yag              = 8; 
    HDag             = 2*pi/2;
    Vlin             = 2.5;
    percep_flag      = 1;               
    imag_flag        = 0;               
    navi_flag        = 0;           
    move2nextTarget  = 0;             
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;               
    targetlist = [6   ,  17; ...
                  5   ,  16; ...
                  9   , 9.5; ...
                  16  ,   5; ...
                  17  ,  17];
    SandT            = 0;              
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [100, 100];                
    Nobj         = 1;                 
    ObjEncThresh = 4.0;
end

if simflag == 28   %%% same as simflag 22 but longer simulation without plotting to get nice rate maps
    Xag              = 5; 
    Yag              = 10; 
    HDag             = 0;
    Vlin             = 2.5;
    percep_flag      = 1;               
    imag_flag        = 0;             
    navi_flag        = 0;               
    move2nextTarget  = 0;              
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;               
    tl = 5.0 + 6*rand(30,2);   
    tl = round(tl*10)/10;
    targetlist = tl;
    SandT = 0;
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [120, 120];                  % dummy
    CoordsTMPmem = [ 12, 12];
    Nobj         = length(CoordsTMP(:,1));;                   
    ObjEncThresh = 4.0;
end

if simflag == 31   %%% encode first, move on, then retrieve and shift att
    Xag              = 8;
    Yag              = 10;
    HDag             = 3*pi/2;
    Vlin             = 2.5;            
    percep_flag      = 1;              
    imag_flag        = 0;               
    navi_flag        = 0;               
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;             
    SandT            = 1;
    targetlist = [ 18, 10; ...          
                   18.5, 10];
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP  = [13, 14; ...         
                  14, 6];
    Nobj         = length(CoordsTMP(:,1));
    ObjEncThresh = 8.0;                 
end

if simflag == 40   %%% mental navigation
    Xag              = 5;
    Yag              = 10;
    HDag             = 3*pi/2;
    Vlin             = 2.5;             
    percep_flag      = 1;               
    imag_flag        = 0;              
    navi_flag        = 0;               
    move2nextTarget  = 0;               
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;             
    SandT            = 0;
    SandTnav         = 4;
    targetlist = [  7, 10; ...          
                   13, 17; ...
                   13,  5; ...
                   16,  5];
    targetlist_imag_nav = [18, 10];
    target_no_imag_nav  = 0;
    CoordsTMP  = [ 7.5,  7; ...           
                    15, 16; ...
                    17, 7.0];
    Nobj         = length(CoordsTMP(:,1));
    ObjEncThresh = 5.0;                
end

if simflag == 50   %%% planning/preplay
    Xag              = 16.1;   
    Yag              = 5;
    HDag             = pi/2;
    Vlin             = 2.5;            
    percep_flag      = 1;              
    imag_flag        = 0;               
    navi_flag        = 0;               
    move2nextTarget  = 0;          
    locateNextTarget = 0;
    rotResolved      = 1;              
    target_no        = 0;          
    targetlist = [16,  5; ...
                  16,  6; ...
                  16, 18; ...
                  10, 18; ...  
                   5, 18; ...
                   4, 18];      
    SandT            = 0;
    SandTnav         = 3;
    targetlist_imag_nav = [16, 18];
    target_no_imag_nav  = 0;
    CoordsTMP = [16, 18; ...        
                 16, 3];
    Nobj             = length(CoordsTMP(:,1));
    ObjEncThresh     = 0.0;                 
    GC2PCbuffer      = zeros(1936,700);
end

if simflag == 60   %%% empty env for TR video
    Xag              = 3; 
    Yag              = 3; 
    HDag             = 3*pi/2;
    Vlin             = 5.0;             
    percep_flag      = 1;               
    imag_flag        = 0;               
    navi_flag        = 0;              
    move2nextTarget  = 0;              
    locateNextTarget = 0;
    rotResolved      = 1;               
    target_no        = 0;               
    tl = 2 + 17*rand(10,2);
    tl = round(tl*10)/10;
    targetlist = tl;
    SandT = 0;                         
    targetlist_imag_nav = [];
    target_no_imag_nav  = 0;
    CoordsTMP    = [];                 
    Nobj         = 0;                  
    ObjEncThresh = 0.000000001;         
end


