%% Plot the current state

function [ax,par] = kf_plot_state(ax,s,par,stats)

%% Shortcuts
names = par.misc.names;

%% Monitor positions
numMons = par.opts.numMons;

%% Initialise if not already initialised
if par.plot.init==0
    
    namestring = names;
    for n=1:length(names); namestring{n} = regexprep(namestring{n},'_',' '); namestring{n}(1) = upper(namestring{n}(1)); end
    
    set(0,'defaulttextinterpreter','latex')
    
    f=0;
    
    FST = 8; % Title font size
    FSA = 8; % Axis label font size
    
    %% Position estimates
    if par.opts.drawfigs.pos_estimates
        figure('Name','Position estimates'); f=f+1;
        %set(gcf,'Position',1e3*[2.3082   -0.1430    0.7664    0.3568])
        if numMons==2; set(gcf,'Position',1e3*[-0.3374    1.9458    1.1504    0.2736]); end
        set(gcf,'color','w')
        M = 1; N = 4; p=1;
        
        % Estimate based purely on PI
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        hold off
        imagesc(par.plot.yg,par.plot.xg,zeros(par.plot.gridsize),[0,1]);           % Plot the measurement distribution
        hold on
        plot(s.xPI(1),s.xPI(2),'om','MarkerSize',6)                              % Overlay the a priori location
        plot(s.X(1),s.X(2),'xw','MarkerSize',10)                               % Overlay the true location
        
        axis equal
        ylim([0,par.Y]); xlim([0,par.X])
        xlabel('X')
        ylabel('Y')
        title(sprintf('Pure PI\nEstimate'),'FontSize',FST)
        legend({'Pure PI','True Location'},'Position',[ 0.1500    0.1100    0.1366    0.150])
        
        % A priori (before measurement) state estimate and true position
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        hold off
        imagesc(par.plot.yg,par.plot.xg,zeros(par.plot.gridsize));             % Plot the measurement distribution
        hold on
        plot(s.xd(1),s.xd(2),'or','MarkerSize',6)                              % Overlay the a priori location
        plot(s.X(1),s.X(2),'xw','MarkerSize',10)                               % Overlay the true location
        
        axis equal
        ylim([0,par.Y]); xlim([0,par.X])
        xlabel('X')
        ylabel('Y')
        title(sprintf('A Priori \nEstimate'),'FontSize',FST)
        legend({'A Priori (PI)','True Location'},'Position',[0.3561    0.1100    0.1366    0.15])
        
        % Measurement and true location
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        hold off
        imagesc(par.plot.yg,par.plot.xg,zeros(par.plot.gridsize));             % Plot the measurement distribution
        hold on
        plot(s.z(1),s.z(2),'oc','MarkerSize',6)                                % Overlay measurement location
        plot(s.X(1),s.X(2),'xw','MarkerSize',10)                               % Overlay the true location
        
        axis equal
        ylim([0,par.Y]); xlim([0,par.X])
        xlabel('X')
        ylabel('Y')
        title(sprintf('Measurement (Sensory)\nEstimate'),'FontSize',FST)
        legend({'Measurement','True Location'},'Position',[0.5622    0.1100    0.1366    0.15])
        
        % A posteriori state estimate and true position
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        imagesc(par.plot.yg,par.plot.xg,zeros(par.plot.gridsize));             % Plot the measurement distribution
        hold on
        plot(s.z(1),s.z(2),'oc','MarkerSize',6)                                % Overlay measurement location
        plot(s.xd(1),s.xd(2),'or','MarkerSize',6)                              % Overlay a priori location
        plot(s.xd(1),s.xd(2),'og','MarkerSize',8)                              % Overlay a posteriori
        plot(s.X(1),s.X(2),'xw','MarkerSize',10)                               % Overlay the true location
        hold off
        
        axis equal
        ylim([0,par.Y]); xlim([0,par.X])
        xlabel('X')
        ylabel('Y')
        title(sprintf('A Posteriori (Integrated)\nEstimate'),'FontSize',FST)
        legend({'Measurement','A Priori','A Posteriori','True Location'},'Position',[0.7684    0.1100    0.1366    0.15])
    end
    
    %% Grid readout
    if par.opts.drawfigs.readout
        
        figure('Name','Grid readouts'); f=f+1;
        if numMons==2; set(gcf,'Position',1e3*[-0.3358    0.9210    1.0920    0.5960]); end
        set(gcf,'color','w')
        M = 2; N = length(names); p=1;
        
        % Rate maps
        for n=1:length(names)
            par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
            if strcmpi(names{n},'aposteriori'); MON_CELL=par.plot.mon_cell; else; MON_CELL = 1; end
            imagesc([0,par.X],[0,par.Y],stats.summary.rate_map.(names{n})(:,:,MON_CELL));
            hold on
            plot(s.X(1),s.X(2),'xr','MarkerSize',10)
            xlim([0,par.X]); ylim([0,par.Y])
            axis equal off
            title([upper(regexprep(names{n},'_',' ')),' Readout'],'FontSize',FST)
            set(gca,'YDir','normal')
        end
        
        % Overlay environment mask
        p=1;
        for n=1:length(names)
            overlayMask(par.plot.xg,par.plot.yg,par.plot.env_mask,par.plot.ax{f}{p}); p=p+1;
            %{
            P = get(par.plot.ax{f}{p},'Position'); p=p+1;
            ax_tmp = axes();
            msk = par.plot.env_mask; msk(msk==0)=nan;
            ax_tmp2 = pcolor(par.plot.xg,par.plot.yg,msk);
            xlim([0,par.X]); ylim([0,par.Y])
            ax_tmp.Position = P;
            colormap(ax_tmp,[0,0,0;1,0,0])
            set(ax_tmp2,'EdgeColor','none')
            set(ax_tmp2,'alphadata',~isnan(msk))
            ax_tmp.Color='none';
            axis equal off
            set(gca,'YDir','reverse')
            title('')
            %}
            set(gca,'YDir','normal')
         end
        
        % Autocorrelograms
        for n=1:length(names)
            par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
            if strcmpi(names{n},'aposteriori'); MON_CELL=par.plot.mon_cell; else; MON_CELL = 1; end
            imagesc(stats.summary. rate_map.([names{n},'_AC'])(:,:,MON_CELL));
            axis equal off
            title([upper(regexprep(names{n},'_',' ')),' AC'],'FontSize',FST)
            set(gca,'YDir','normal')
        end

    end
    
    %% Statistics
    if par.opts.drawfigs.stats
        figure('Name','Statistics'); f=f+1;
        if numMons==2; set(gcf,'Position',1e3*[ -0.3350    1.6058    1.1504    0.6056]); end
        set(gcf,'color','w')
        M = 2; N = 2; p=1;
        
        % Mean estimate errors
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        bar(zeros(1,length(names)));
        axis equal
        title('Mean Estimation Error','FontSize',FST)
        set(gca,'XTickLabel',namestring)
        set(gca,'XTickLabelRotation',45)
        
        % Gridness over time
        subplot(M,N,p);
        par.plot.ax{f}{p} = {}; 

        yyaxis left
        par.plot.ax{f}{p}{1} = plot(ones(2,length(names)),zeros(2,length(names)));
        ylabel('Gridness','FontSize',FSA)
        ylim([-1,1])
        
        yyaxis right
        par.plot.ax{f}{p}{2} = plot(ones(2,1),zeros(2,1));
        ylabel('Mean Error','FontSize',FSA)
        ylim([0,par.grid.scale/2])
        
        xlabel('Iteration')
        xlim([1,par.T])
        axis square
        legend(namestring,'Position',[0.0062    0.0166    0.1032    0.1536])
                
        p=p+1;
        
        % Sensory estiate input error over time
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        plot(1,0)
        ylabel('Error','FontSize',FSA)
        xlabel('Iteration')
        xlim([1,par.T])
        title('Sensory Estimate Error')
        axis equal
        
        % Occupancy of environment
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        imagesc(par.plot.xg,par.plot.yg,stats.occupancy_map); axis equal off
        hold on
        plot(s.X(1),s.X(2),'xr','MarkerSize',10)                              
        title('Occupancy','FontSize',FST)

    end
    
    %% Base sheet activity
    
    % Overall firing
    figure('Name','Base sheet'); f=f+1;
    if numMons==2; set(gcf,'Position',1e3*[0.8234    1.6162    1.1104    0.6016]); end
    set(gcf,'color','w')
    M = 1; 
    N = sum([par.opts.drawfigs.CANsheet && par.opts.update.CANsheet,...
             par.opts.drawfigs.pc_firing && par.opts.update.pc_firing]);
    p=1;
    
    xw = wrap_xy_in_hex(s.X,par.grid.phi,par.grid.scale,[0,0]);
    if par.opts.drawfigs.CANsheet && par.opts.update.CANsheet
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),10000/(par.grid.Ncell)^2,zeros(1,size(par.plot.grid_baseSheet,1)),'filled')
        axis equal off
        
        hold on
        
        plot(xw(1),xw(2),'xr','MarkerSize',20)
        
        for i = 1:7; x(i,:) = 0.5*par.grid.scale*2/sqrt(3)*[cosd(60*i+par.grid.phi+30),sind(60*i+par.grid.phi+30)]; end
        plot(x(:,1),x(:,2),'k')
        
        title('GC Firing')
            
    end
    
    % PC-GC Stim
    if par.opts.drawfigs.pc_firing && par.opts.update.pc_firing
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        scatter(par.plot.grid_baseSheet(:,1),par.plot.grid_baseSheet(:,2),...
            10000/(par.grid.Ncell)^2,...
            zeros(1,size(par.plot.grid_baseSheet,1)),'filled')
        title('PC $\rightarrow$GC Stim.')
        
        hold on
        plot(xw(1),xw(2),'xr','MarkerSize',20)
        
        hold on
        for i = 1:7; x(i,:) = 0.5*par.grid.scale*2/sqrt(3)*[cosd(60*i+par.grid.phi+30),sind(60*i+par.grid.phi+30)]; end
        plot(x(:,1),x(:,2),'k')
        axis equal off
        
        %ylim(par.grid.scale/2*[-2/sqrt(3),2/sqrt(3)])
        %xlim(par.grid.scale/2*[-1,1])
    end
    
    %% PC firing
    if par.opts.drawfigs.pc_firing
        
        figure('Name','Place cells'); f=f+1;
        if numMons==2; set(gcf,'Position',[764.2000  923.4000  597.6000  594.4000]); end
        set(gcf,'color','w')
        M = 2; N = 3; p=1;
                
        % Firing
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        if ~strcmpi(par.pc.distribution,'bvc')
            scatter(gather(par.pc.mu(:,1)),gather(par.pc.mu(:,2)),2e6/par.pc.npc^2,zeros(1,par.pc.npc),'filled')
            hold on
            
            plot(s.X(1),s.X(2),'xr','MarkerSize',10)
            
            %set(gca,'Ydir','Reverse')
            axis equal
            box on
            set(gca,'YTick',[],'XTick',[])
            xlim([0,par.X]); ylim([0,par.Y])
        else
            % PCs don't have predetermined COMs when they're generated from
            % the BVCs
            plot(ones(1,par.pc.npc),'o-')
        end
        
        title('PC Firing')
        
        % Weights
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        imagesc(s.pc.w); axis equal
        title('PC-GC Weights')
        xlabel('GC \#')
        ylabel('PC \#')
        
        % Example PC
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        if ~strcmpi(par.pc.distribution,'bvc')
            imagesc(reshape(bvnpdf(par.plot.grid,[0.5,0.5],par.pc.C(:,:,1)),par.plot.gridsize)); axis equal
        else
            imagesc(par.pc.example_pc); axis equal off
        end
        title('Example PC')

        % History of weights
        if par.opts.drawfigs.weights
            par.plot.logFunc = @(x) x; %sign(x+eps).*real(log(x+eps));
            
            par.plot.ax{f}{p} = subplot(M,N,p);
            plot(zeros(2,par.pc.npc),par.plot.logFunc(repmat(gather(sum(s.pc.w,2))',2,1)))
            title('PC-GC Weights (Each PC))')
            ylabel('Weight (sgn(x)*real(log(x)))')
            xlabel('Time')
            axis equal
        end
        p=p+1;
        
        % Change in weights
        if par.opts.drawfigs.changeweights
            par.plot.ax{f}{p} = subplot(M,N,p);
            plot(zeros(2,par.pc.npc),par.plot.logFunc(zeros(2,par.pc.npc)))
            title({'Change in abs(sum(w))','PC-GC Weights (Each PC))'})
            ylabel('dw/dt (sgn(x)*real(log(x)))','Interpreter','Tex')
            xlabel('Time')
            axis equal
        end
        p=p+1;
        
        % Threshold
        if par.opts.drawfigs.threshold
        par.plot.ax{f}{p} = subplot(M,N,p);
        plot(zeros(2,par.pc.npc),zeros(2,par.pc.npc))
        xlabel('Time')
        ylabel('Threshold')
        title('BCM Modification Threshold')
        end
        p=p+1;
        
    end
    
    %% PC->GC sheet connections
    if par.opts.drawfigs.pc_firing && par.opts.update.pc_firing && par.opts.drawfigs.CANsheet && par.opts.update.CANsheet
        figure('Name','PC_to_GC_sheet_connections'); f=f+1;
        if numMons==2; set(gcf,'Position',1e3*[1.3730    0.9274    0.5600    0.5920]); end
        set(gcf,'color','w')
        M=1; N=1; p=1;
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        
        kf_plot_pc_gc_weights(par,ones(par.pc.npc*par.grid.NcellSheet,1));
        
        if strcmpi(par.pc.distribution,'bvc'); title('Position does not correspond to PC COMs'); end
        
    end
    
    %% Replay statistics
    if par.opts.replay.option
        figure('Name','Replay statistics'); f=f+1;
        
        M = 1; N = 3; p = 1;
        
        % Prediction error over time
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        plot(0,0);
        xlabel('Iterations')
        ylabel('Prediction error')
        axis square
        
        % Grid sheet pairwise encoding error
        subplot(M,N,p); 
        par.plot.ax{f}{p}{1} = plot(0,0);
        par.plot.ax{f}{p}{2} = [];
        p=p+1;
        
        xlabel('Iterations')
        ylabel('Grid sheet pairwise error')
        axis square
        
        % Place cell pairwise error
        par.plot.ax{f}{p} = subplot(M,N,p); p=p+1;
        plot(0,0);
        xlabel('Iterations')
        ylabel('Place cell pairwise error')
        axis square
        
        set(gcf,'color','w','Position',1e3*[0.5650    1.0378    0.6136    0.4800])
    
    end
  
    %% Formatting
    for i = 1:length(ax); for j = 1:length(ax{i}); ax{i}{j} = flipud(ax{i}{j}); end; end
    
    par.plot.init = 1; % Signal that plots have now been initially drawn
    
    set(0,'defaulttextinterpreter','none')
    
    if ~isempty(par.opts.layout); layout('load',par.opts.layout); end
    
end

%% Update
if 1
    
    f=0;
    
    %% Position plot
    if par.opts.drawfigs.pos_estimates
        
        f=f+1; p=1;
        
        % Update the pure PI state plot
        par.plot.ax{f}{p}.Children(2).YData = s.xPI(2); par.plot.ax{f}{p}.Children(2).XData = s.xPI(1);
        par.plot.ax{f}{p}.Children(1).YData = s.X(2); par.plot.ax{f}{p}.Children(1).XData = s.X(1); p=p+1;
        
        % Update a priori state plot
        par.plot.ax{f}{p}.Children(3).CData = reshape(mvnpdf(par.plot.grid,s.xd',s.Pd),par.plot.gridsize);
        par.plot.ax{f}{p}.Children(2).YData = s.xd(2); par.plot.ax{f}{p}.Children(2).XData = s.xd(1);
        par.plot.ax{f}{p}.Children(1).YData = s.X(2); par.plot.ax{f}{p}.Children(1).XData = s.X(1); p=p+1;
        
        % Update measurement plot
        par.plot.ax{f}{p}.Children(3).CData = reshape(mvnpdf(par.plot.grid,s.z',s.R),par.plot.gridsize);
        par.plot.ax{f}{p}.Children(2).YData = s.z(2); par.plot.ax{f}{p}.Children(2).XData = s.z(1);
        par.plot.ax{f}{p}.Children(1).YData = s.X(2); par.plot.ax{f}{p}.Children(1).XData = s.X(1); p=p+1;
        
        % Update a posteriori state plot
        par.plot.ax{f}{p}.Children(5).CData = reshape(mvnpdf(par.plot.grid,s.x',s.P),par.plot.gridsize);
        par.plot.ax{f}{p}.Children(4).YData = s.z(2); par.plot.ax{f}{p}.Children(4).XData = s.z(1);
        par.plot.ax{f}{p}.Children(3).YData = s.xd(2); par.plot.ax{f}{p}.Children(3).XData = s.xd(1);
        par.plot.ax{f}{p}.Children(2).YData = s.x(2); par.plot.ax{f}{p}.Children(2).XData = s.x(1);
        par.plot.ax{f}{p}.Children(1).YData = s.X(2); par.plot.ax{f}{p}.Children(1).XData = s.X(1); p=p+1;
    end
    
    %% Grid readout plots
    if par.opts.drawfigs.readout
        f=f+1; p=1;
        
        % Firing rate maps
        for n=1:length(names)
            if strcmpi(names{n},'aposteriori'); MON_CELL=par.plot.mon_cell; else; MON_CELL = 1; end
            par.plot.ax{f}{p}.Children(2).CData = (stats.summary.rate_map.(names{n})(:,:,MON_CELL)); 
            par.plot.ax{f}{p}.Children(1).XData=s.X(1); par.plot.ax{f}{p}.Children(1).YData=s.X(2); 
            p=p+1;
        end
        
        % Autocorrelograms
        for n=1:length(names)
            if strcmpi(names{n},'aposteriori'); MON_CELL=par.plot.mon_cell; else; MON_CELL = 1; end
            par.plot.ax{f}{p}.Children(1).CData = stats.summary.rate_map.([names{n},'_AC'])(:,:,MON_CELL); 
            p=p+1; 
        end
    end
    
    %% Stats plot
    if par.opts.drawfigs.stats
        f=f+1; p=1;
        
        % Update error bar plot
        par.plot.ax{f}{p}.Children.YData = struct2array(stats.err); p=p+1;
        
        % Skip a plot
        %p=p+1;
        
        % Gridness over time
        gr=[]; for n=1:length(names); gr(n)=stats.summary.(names{n}){1}.gridness; end; gr=fliplr(gr);
        for ch=1:length(par.plot.ax{f}{p}{1})
            par.plot.ax{f}{p}{1}(ch).YData = horzcat(par.plot.ax{f}{p}{1}(ch).YData,gr(ch));
            par.plot.ax{f}{p}{1}(ch).XData = horzcat(par.plot.ax{f}{p}{1}(ch).XData,par.t);
        end
        
        % Update measurement error
        err=[]; for n=1:length(names); err(n)=stats.summary.err.(names{n})(end); end; gr=fliplr(err);
        for ch=1:length(par.plot.ax{f}{p}{2})
            par.plot.ax{f}{p}{2}(ch).YData = horzcat(par.plot.ax{f}{p}{2}(ch).YData,err(ch));
            par.plot.ax{f}{p}{2}(ch).XData = horzcat(par.plot.ax{f}{p}{2}(ch).XData,par.t);
        end
        
        p=p+1;
        
        % Measurement estimate input over time
        par.plot.ax{f}{p}.Children(1).YData = horzcat(par.plot.ax{f}{p}.Children(1).YData,stats.measurement_stim_err);
        par.plot.ax{f}{p}.Children(1).XData = horzcat(par.plot.ax{f}{p}.Children(1).XData,par.t); p=p+1;
        
        % Occupancy map
        par.plot.ax{f}{p}.Children(1).XData = s.X(1); par.plot.ax{f}{p}.Children(1).YData = s.X(2);
        par.plot.ax{f}{p}.Children(2).CData = stats.occupancy_map; p=p+1;
    
    end
        
    %% Base sheet
    f=f+1; p=1;
    
    % GC firing
    xw = wrap_xy_in_hex(s.X(:)',par.grid.phi,par.grid.scale,par.grid.offset);
    if par.opts.drawfigs.CANsheet && par.opts.update.CANsheet
        par.plot.ax{f}{p}.Children(3).CData = gather(stats.baseSheetActivity);
        par.plot.ax{f}{p}.Children(2).XData = xw(1); par.plot.ax{f}{p}.Children(2).YData = xw(2); p=p+1;
    end
    
    % PC->GC stimulation
    if par.opts.update.pc_firing
        par.plot.ax{f}{p}.Children(3).CData = gather(s.pc.gc_u);
        par.plot.ax{f}{p}.Children(2).XData = xw(1); par.plot.ax{f}{p}.Children(2).YData = xw(2); p=p+1;
    end
        
    %% Place cell firing
    if par.opts.drawfigs.pc_firing && par.opts.update.pc_firing
        f=f+1; p=1;
        
        % PC Firing
        if ~strcmpi(par.pc.distribution,'bvc')
            par.plot.ax{f}{p}.Children(2).CData = gather(s.pc.F);
            par.plot.ax{f}{p}.Children(1).YData = s.X(2); par.plot.ax{f}{p}.Children(1).XData = s.X(1); p=p+1;
        else
            par.plot.ax{f}{p}.Children(1).YData = gather(s.pc.F(par.pc.sort_inds)); p=p+1;
        end
        
        % PC weights image
        par.plot.ax{f}{p}.Children.CData = gather(s.pc.w(par.pc.sort_inds,:)); p=p+1;
        
        % Skip plot
        p=p+1;
        
        % Weights
        if par.opts.drawfigs.weights
            W = par.plot.logFunc(gather(sum(s.pc.w,2)));
            for ch=1:length(par.plot.ax{f}{p}.Children)
            par.plot.ax{f}{p}.Children(ch).YData = horzcat(par.plot.ax{f}{p}.Children(ch).YData,W(ch));
            par.plot.ax{f}{p}.Children(ch).XData = horzcat(par.plot.ax{f}{p}.Children(ch).XData,par.t);
            end
        end
        p=p+1;
        
        % Change in weights
        if par.opts.drawfigs.changeweights
            dw = gather(sum(s.pc.dw,2));
            for ch=1:length(par.plot.ax{f}{p}.Children)
                par.plot.ax{f}{p}.Children(ch).YData = horzcat(par.plot.ax{f}{p}.Children(ch).YData,dw(ch));
                par.plot.ax{f}{p}.Children(ch).XData = horzcat(par.plot.ax{f}{p}.Children(ch).XData,par.t);
            end
        end
        p=p+1;
        
        % Threshold
        if par.opts.drawfigs.threshold
            for ch=1:length(par.plot.ax{f}{p}.Children)
                par.plot.ax{f}{p}.Children(ch).YData = horzcat(par.plot.ax{f}{p}.Children(ch).YData,gather(s.pc.Th_v(ch)));
                par.plot.ax{f}{p}.Children(ch).XData = horzcat(par.plot.ax{f}{p}.Children(ch).XData,par.t);
            end
        end
        p=p+1;
        
    end
    
    %% PC->GC sheet connections
    if par.opts.drawfigs.pc_firing && par.opts.update.pc_firing && par.opts.drawfigs.CANsheet && par.opts.update.CANsheet
        f=f+1; p=1;
        W = s.pc.w; %bsxfun(@times,s.pc.w,1./max(eps,max(s.pc.w,[],2))); % Normalise the range of each cell
        %W = bsxfun(@times,s.pc.w,s.pc.F);                      % Visualise the currently active cell
        %W = s.pc.w;
        %if ~strcmpi(par.pc.distribution,'bvc')
            par.plot.ax{f}{p}.Children.CData = gather(W(:));
        %else
            %par.plot.ax{f}{p}.Children.CData = gather(W);
        %end
        %al = gather(repmat(s.pc.F,[par.grid.NcellSheet,1])); al = al/max(al(:)); par.plot.ax{f}{p}.Children.MarkerFaceAlpha = al;
    end
    p=p+1;
    
    %% Replay
    if par.opts.replay.option
        f=f+1; p=1;
        
        par.plot.ax{f}{p}.Children.YData = stats.replay.pred_err(1:par.t);                % Prediction error
        par.plot.ax{f}{p}.Children.XData = 1:par.t;
        xlim([0,par.t])
        p=p+1;
        
        par.plot.ax{f}{p}{1}.YData = stats.p2g_pw_err.mean;                             % Grid pairwise error    
        par.plot.ax{f}{p}{1}.XData = stats.pw_err_t;
        xlim([0,par.t])
        axes(par.plot.ax{f}{p}{1}.Parent)
        SP = find(stats.replay.spwr); 
        hold on
        for pp=1:length(par.plot.ax{f}{p}{2}); delete(par.plot.ax{f}{p}{2}(pp)); end
        for sp=1:length(SP)
            par.plot.ax{f}{p}{2}(sp) = plot([(SP(sp)),SP(sp)],[min(par.plot.ax{f}{p}{1}.YData),max(par.plot.ax{f}{p}{1}.YData)],'r');
        end
        p=p+1;
        
        par.plot.ax{f}{p}.Children.YData = stats.p2p_pw_err.mean;                            % Place cell pairwise error
        par.plot.ax{f}{p}.Children.XData = stats.pw_err_t;
        xlim([0,par.t])
        p=p+1;
    end
        
end

%% Formatting
drawnow

end