%% Test the BVC firing function
close all
set(0,'DefaultFigureWindowStyle','docked')

visualise_plotting = 1;
save_option = 1;

bvc_model = 'bvc_full'; % 'simple_exp_decay' or 'bvc_full'
angle_index = 0:1:359;

% Define an environment with boundaries at the edges
walls = [1,1,1,1];

env_mask = zeros(101);
if walls(1); env_mask(:,1) = 1; end
if walls(2); env_mask(:,end) = 1; end
if walls(3); env_mask(1,:) = 1; end
if walls(4); env_mask(end,:) = 1; end

[par.env.x_bound,par.env.y_bound] = ind2sub(size(env_mask),find(env_mask));

% Test the BVC function

par.bvc.tuning_distance = 81e-3; % Tuning distance (set at 81mm)

% Overqrite some parameters if using the dimple model
%if strcmpi(bvc_model,'simple_exp_decay')
%    angle_index = [0,90,180,270];
%end

% Get the firing rates of BVCs tuned to each direction
F = cell(1,length(angle_index));
for i = 1:length(angle_index)
    clc; fprintf('%i deg...\n',angle_index(i));
    par.bvc.phi = angle_index(i)*pi/180;
    F{i} = kf_bvc_func_env(par,'gpu_option',1);
end

%% Also visualise the shape of the tuning curve from the agent's position at the centre of the environment (the 'spotlight' view...)
xvec = 0:0.01:1; [xq,yq] = meshgrid(xvec);

spot = cell(1,length(angle_index));
for i = 1:length(angle_index)
    sigA = 0.2;
    sig0 = 122e-3;
    beta = 1830e-3;
    sigR = (d/beta + 1)*sig0;
    
    if strcmpi(bvc_model,'simple_exp_decay'); d = 0; sigA = 0; end
    
    spot{i} = exp( -(  sqrt((xq-0.5).^2+(yq-0.5).^2)-d  ).^2 / (2*sigR^2) ) .*...%/ sqrt(2*pi*sigR^2).*...
        exp( -(-pi+mod(atan2(0.5-yq,0.5-xq)-angle_index(i)*pi/180+pi,2*pi)).^2 / (2*sigA^2) );% / sqrt(2*pi*sigA^2);
end

%v = VideoWriter('BVC_demo.avi');
%open(v);

%% Plot
close all
if visualise_plotting
    subplot(1,2,1)
    Fim = imagesc(linspace(0,1,size(F{1}-2,2)),linspace(0,1,size(F{1}-2,1)),F{1}(2:(end-1),2:(end-1))); axis square off
    title(sprintf('d=%.2f',d))
    hold on
    line_plot =     plot([0.5,0.5+0.25*cosd((angle_index(1)))],[0.5,0.5+0.25*sind((angle_index(1)))],'r--');
    subplot(1,2,2)
    spot_im = imagesc(linspace(0,1,size(F{1},2)),linspace(0,1,size(F{1},1)),spot{1}); axis square off
    hold off
    for i = 1:length(angle_index)
        set(Fim,'CData',F{i}(2:(end-1),2:(end-1)));
        set(line_plot,'YData',[0.5,0.5+0.25*sind((angle_index(i)-180))],'XData',[0.5,0.5+0.25*cosd((angle_index(i)-180))]);
        title(sprintf('phi=%i',angle_index(i)))
        set(spot_im,'CData',spot{i});
        drawnow
        %pause(0.01);
        pause
        %writeVideo(v,getframe(gcf));
    end
end

%close(v);

set(0,'DefaultFigureWindowStyle','normal')

%% Save the data to file
if save_option
    save(sprintf('../Data_Files/BVC_Firing_d_%.2f_Walls_%i%i%i%i_All_Directions_%s.mat',d,walls(1),walls(2),walls(3),walls(4),bvc_model),'F','-v7.3')
end