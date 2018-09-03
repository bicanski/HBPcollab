
% Plotting subroutine for the BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%

function [frame]  = BB_subr_PLT_fancy(step, modstep, plt_ttl_flag, percep_flag, simflag, ...
    H_rate, NHx, NHy, HX, HY, ...
    GC2PCwts, GC_rate, ...
    BVCX, BVCY, BVC_rate, NBVCR, NBVCTheta, ...
    HDX, HDY, HD_rate, ...
    PW_rate, ...
    LVC_rate, ...
    oPW_rate, ...
    PR_rate, nobarrier, ...
    oPR_rate, ObjEncoded, ...
    TX, TY, BX, BY, ObjCenX, ObjCenY, ...
    old_HD, HDag, Xag, Yag, ...
    imag_flag, navi_flag_imag, Xag_imag_nav, Yag_imag_nav, ...
    oldCoordsTMP, AllImagX, AllImagY, VBX_rt, VBY_rt, L_r, ...
    Hadapt, I_comp, ...
    Hmem1, Hmem2, Hmem3, oPRCue, BndryPtX_insB, BndryPtY_insB,CoordsTMPmem)



IMsizeX = 1350; 
IMsizeY = 825;
if strcmp(computer,'MACI64')
    IMsizeX = 1620/2; 
    IMsizeY = 990/2; 
end

if step == modstep
    fig = figure('position',[100 100 IMsizeX IMsizeY]);%, 'visible', 'off');
else
    fig = gca;
end

colormap hot

Olive = [6/255, 177/255, 82/255];
Gblue = [44/255, 102/255, 212/255];
Goran = [232/255, 81/255, 0/255];
Rpurp = [128/255, 47/255, 202/255];
myGrey = [120/255, 120/255, 120/255];
Ocolor = Olive;
if simflag == 50
    Ocolor = Rpurp;
end



pltx = 22;
plty = 11;

ego_range  = [1*pltx+(2:4) 2*pltx+(2:4) 3*pltx+(2:4)]-1;
PWb_range  = [1*pltx+(5:8) 2*pltx+(5:8) 3*pltx+(5:8)];
PWo_range  = [4*pltx+(5:8) 5*pltx+(5:8) 6*pltx+(5:8)];
HD_range   = [3*pltx+(9:11) 4*pltx+(9:11)];
BVC_range  = [1*pltx+(12:15) 2*pltx+(12:15) 3*pltx+(12:15)];
OVC_range  = [4*pltx+(12:15) 5*pltx+(12:15) 6*pltx+(12:15)];
PRb_range  = [2*pltx+(17:20) 3*pltx+(17:20)]+1;
PRo_range  = [5*pltx+(17:18) 6*pltx+(17:18)]+1;
OBJ_range  = [5*pltx+(20) 6*pltx+(20)]+1;
allo_range = [4*pltx+(2:4) 5*pltx+(2:4) 6*pltx+(2:4)]-1;
PC_range   = [8*pltx+(12:15) 9*pltx+(12:15) 10*pltx+(12:15)]+2;
GC_range   = [8*pltx+(17:20) 9*pltx+(17:20) 10*pltx+(17:20)]+2;
PP1_range  = [8*pltx+(1:3) 9*pltx+(1:3) 10*pltx+(1:3)];
PP2_range  = [8*pltx+(5:7) 9*pltx+(5:7) 10*pltx+(5:7)];
PP3_range  = [8*pltx+(9:11) 9*pltx+(9:11) 10*pltx+(9:11)];



subplot(plty,pltx,ego_range,'replace');
plot(VBX_rt,VBY_rt,'k.'); hold on;
plot(0,0,'^','MarkerFaceColor','k','MarkerEdgeColor','k','MarkerSize',3.5);hold on
plot(VBX_rt(VBY_rt>0),VBY_rt(VBY_rt>0),'.','color',Rpurp);hold on
xlabel('egoc. agent view');
if imag_flag || navi_flag_imag
    R = [cos(old_HD) -sin(old_HD); sin(old_HD)  cos(old_HD)];
else
    R = [cos(HDag) -sin(HDag); sin(HDag)  cos(HDag)];
end
oL_r = R'*[ObjCenX'-Xag ; ObjCenY'-Yag];
ag_r = [0 ; 0];
plot(oL_r(1,:)-ag_r(1),oL_r(2,:)-ag_r(2),'o','MarkerFaceColor',Ocolor,'MarkerEdgeColor',Ocolor,'MarkerSize',4);hold on
set(gca,'ylim', [-22 22]);
set(gca,'xlim', [-22 22]);
axis off
axis square;
box off
text(-40,18,['egocentric'; 'agent view']);
%text(-18,-26,['egoc. agent view']);



subplot(plty,pltx,PWb_range,'replace');
%surface(BVCX,BVCY,reshape(PW_rate,NBVCR,NBVCTheta)','EdgeColor','none');%colorbar
surface(BVCX,BVCY,reshape(PW_rate,NBVCR,NBVCTheta)');%colorbar
caxis([0 1]);
shading interp;
axis square;
box off
axis off
text(-4,-18.5,'Behind');   text(-4,18,'Ahead');   text(-19,0,'L');   text(17,0,'R');
text(-20,15,['PWb  '; 'rates']);

subplot(plty,pltx,PWo_range,'replace');
surface(BVCX,BVCY,reshape(oPW_rate,NBVCR,NBVCTheta)');%colorbar
%caxis([0 1]);%
caxis([min(oPW_rate) max(0.15,max(oPW_rate))]);
if sum(oPRCue)==0
   caxis([min(oPW_rate) 1]);
end
shading interp;
axis square;
box off
axis off
text(-4,-18.5,'Behind');   text(-4,18,'Ahead');   text(-19,0,'L');   text(17,0,'R');
text(-20,15,['PWo  '; 'rates']);



if simflag==11 || simflag==12 || simflag==111 
    HD_rate=HD_rate*0;
end
subplot(plty,pltx,HD_range,'replace');
ax = gca;
surface(HDX,HDY,[HD_rate HD_rate]);%colorbar
shading interp;
xlabel('HD Rates')
set(ax,'ylim',[-1.6 1.6]);
set(ax,'xlim',[-1.6 1.6]);
axis square;
text(-0.05,-1.8-0.1,'S');   text(-0.05,1.8-0.1,'N');   text(-1.8-0.1,0,'W');   text(1.8-0.1,0,'E');
ax.XTick = [];   ax.YTick = [];
box off
axis off
text(-2.2,1.4,['HDC  '; 'rates']);
caxis([0 1]);



subplot(plty,pltx,BVC_range,'replace');
%surface(BVCX,BVCY,reshape(BVC_rate,NBVCR,NBVCTheta)','EdgeColor','none'); hold on;%colorbar
surface(BVCX,BVCY,reshape(BVC_rate,NBVCR,NBVCTheta)'); hold on;%colorbar
caxis([0 1]);%
shading interp;
axis square;
box off
axis off
text(-1,-18.5,'S');   text(-1,18,'N');   text(-19.5,0,'W');   text(17,0,'E');
text(-20,15,['BVC  '; 'rates']);



subplot(plty,pltx,OVC_range,'replace');
surface(BVCX,BVCY,reshape(LVC_rate,NBVCR,NBVCTheta)'); hold on; %colorbar
%caxis([0 1]);%
caxis([min(LVC_rate) max(0.15,max(LVC_rate))]);
if sum(oPRCue)==0
   caxis([min(LVC_rate) 1]);
end
shading interp;
axis square;
box off
axis off
text(-1,-18.5,'S');   text(-1,18,'N');   text(-19.5,0,'W');   text(17,0,'E');
text(-20,15,['OVC  '; 'rates']);
curax = gca;
secAxesPos = curax.Position;
axes('position',secAxesPos)
plot([10 20],[-20 -20],'k')
ylim([-20 20]);
xlim([-20 20]);
box off
axis off
text(22,-20,'1 m');%,'FontWeight','bold');



if percep_flag==1  
     delete(findall(gcf,'type','annotation'))
     ColStr = 'black';
     ColStr2 = Rpurp;
     annotation('textbox',[.437 .79 .055 .03],'String','\bf TR/RSC');
     ha = annotation('arrow','Color',ColStr2);  
     ha.X = [0.395 0.434];       
     ha.Y = [0.77 0.805];   
     ha.LineWidth  = 3;          
     hb1 = annotation('arrow','Color',ColStr2);
     hb1.X = [0.23 0.265];       
     hb1.Y = [0.75 0.75];   
     hb1.LineWidth  = 3;   
     hb2 = annotation('arrow','Color',Olive);
     hb2.X = [0.231 0.265];       
     hb2.Y = [0.74 0.63];   
     hb2.LineWidth  = 3;   
     hc = annotation('arrow','Color',ColStr);  
     hc.X = [0.465 0.465];         
     hc.Y = [0.72 0.78];   
     hc.LineWidth  = 3;
     hd = annotation('arrow','Color',ColStr2); 
     hd.X = [0.495 0.53];          
     hd.Y = [0.805 0.775];   
     hd.LineWidth  = 3;   
     annotation('textbox',[.437 .44 .055 .03],'String','\bf TR/RSC');
     he = annotation('arrow','Color',Olive);
     he.X = [0.395 0.434];
     he.Y = [0.49 0.455];
     he.LineWidth  = 3;
     hf = annotation('arrow','Color',ColStr); 
     hf.X = [0.465 0.465];          
     hf.Y = [0.535 0.48];   
     hf.LineWidth  = 3; 
     hg = annotation('arrow','Color',Olive); 
     hg.X = [0.495 0.53];          
     hg.Y = [0.455 0.49];   
     hg.LineWidth  = 3;    
end
if percep_flag==0  
     delete(findall(gcf,'type','annotation'))
     ColStr = Goran;
     annotation('textbox',[.437 .79 .055 .03],'String','\bf TR/RSC');
     ha = annotation('arrow','Color',ColStr);  
     ha.X = flip([0.395 0.434]);       
     ha.Y = flip([0.77 0.805]);   
     ha.LineWidth  = 3;    
     hc = annotation('arrow','Color','black');  
     hc.X = [0.465 0.465];         
     hc.Y = [0.72 0.78];   
     hc.LineWidth  = 3; 
     hd = annotation('arrow','Color',ColStr); 
     hd.X = flip([0.495 0.53]);          
     hd.Y = flip([0.805 0.775]);   
     hd.LineWidth  = 3;   
     annotation('textbox',[.437 .44 .055 .03],'String','\bf TR/RSC');
     he = annotation('arrow','Color',ColStr);
     he.X = flip([0.395 0.434]);
     he.Y = flip([0.49 0.455]);
     he.LineWidth  = 3;
     hf = annotation('arrow','Color','black'); 
     hf.X = [0.465 0.465];          
     hf.Y = [0.535 0.48];   
     hf.LineWidth  = 3;
     hg = annotation('arrow','Color',ColStr); 
     hg.X = flip([0.495 0.53]);          
     hg.Y = flip([0.455 0.49]);   
     hg.LineWidth  = 3;    
end



if nobarrier == 0 && length(PR_rate)==6
    subplot(plty,pltx,PRb_range,'replace');
    ax = gca;
    bar(PR_rate,'k');
    title('PRb rates','FontWeight','Normal');
    ax.XTick = [1 2 3 4 5 6]; ax.XTickLabel = {'N','W','S','E', 'bN', 'bS'};
    ax.YTick = [0 0.5 1];
    set(ax,'ylim',[0 1]);
end
if nobarrier == 0 && length(PR_rate)==4
    subplot(plty,pltx,PRb_range,'replace');
    ax = gca;
    bar(PR_rate,'k');
    title('PRb rates','FontWeight','Normal');
    ax.XTick = [1 2 3 4]; ax.XTickLabel = {'N','W','S','E'};
    ax.YTick = [0 0.5 1];
    set(ax,'ylim',[0 1]);
end
if nobarrier == 0 && length(PR_rate)==8
    subplot(plty,pltx,PRb_range,'replace');
    ax = gca;
    bar(PR_rate,'k');
    title('PRb rates','FontWeight','Normal');
    ax.XTick = [1 2 3 4 5 6 7 8]; ax.XTickLabel = {'N','W','S','E1','E2','E3','E4','E5'};
    ax.YTick = [0 0.5 1];
    set(ax,'ylim',[0 1]);
end
if nobarrier == 1 && length(PR_rate)==8
    subplot(plty,pltx,PRb_range,'replace');
    ax = gca;
    bar(PR_rate,'k');
    title('PRb rates','FontWeight','Normal');
    ax.XTick = [1 2 3 4 5 6 7 8]; ax.XTickLabel = {'N','W','S','Bw','','Be','','E'};
    ax.YTick = [0 0.5 1];
    set(ax,'ylim',[0 1]);
end



subplot(plty,pltx,PRo_range,'replace');
ax = gca;
bar(oPR_rate,'k');
title('PRo rates','FontWeight','Normal');
%ax.Tick = [1 2];
set(ax,'xlim',[0 3]);
if simflag == 40
    set(ax,'xlim',[0 4]);
end
set(ax,'ylim',[0 1]);



subplot(plty,pltx,OBJ_range,'replace');
ax = gca;
bar(ObjEncoded,'k');
title('object encoded','FontWeight','Normal');
%ax.XTick = [1 2];
set(ax,'xlim',[0 3]);
if simflag == 40
    set(ax,'xlim',[0 4]);
end
set(ax,'ylim',[0 1]);
ax.YTick = [];



Ocolor = Olive;
if simflag == 26
    insBX = BndryPtX_insB(logical( double(BndryPtX_insB>4) .* double(BndryPtX_insB<18) .* double(BndryPtY_insB>4) .* double(BndryPtY_insB<18) ));   % all pts inside the main walls
    insBY = BndryPtY_insB(logical( double(BndryPtX_insB>4) .* double(BndryPtX_insB<18) .* double(BndryPtY_insB>4) .* double(BndryPtY_insB<18) ));
end
subplot(plty,pltx,allo_range,'replace');
plot(BX,BY,'k.','LineWidth',1.5); hold on
if simflag == 26
    plot(insBX,insBY,'r.','LineWidth',1.5); hold on
end
reallocVertices = [TX TY+1.2; TX-0.6 TY-0.4; TX+0.6 TY-0.4; TX TY+1.2];
reallocVertices(:,1) = reallocVertices(:,1)-TX;
reallocVertices(:,2) = reallocVertices(:,2)-TY;
if imag_flag
    HDarr = old_HD;
    HDarr2 = HDag;
else
    HDarr = HDag;
end
R = [cos(HDarr) -sin(HDarr); sin(HDarr)  cos(HDarr)];
pVrl = R*reallocVertices';
plot(TX+pVrl(1,:),TY+pVrl(2,:),'k'); hold on
plot(TX,TY,'k.'); hold on
xlabel('alloc. agent position');
set(gca,'ylim', [-1 23]);
set(gca,'xlim', [-1 23]);
axis square;
axis off
box off
text(-18,20,['allocentric   '; 'agent position']);
%text(0,-3,['alloc. agent position']);
if imag_flag && Xag_imag_nav<0 && simflag~=242
    Htmp = reshape(H_rate,NHx,NHy)';
    Ytmp = sum(Htmp,2);
    Xtmp = sum(Htmp,1);
    Xag_imag = find(Xtmp==max(Xtmp))/2;
    Yag_imag = find(Ytmp==max(Ytmp))/2;
    imaglocVertices = [Xag_imag Yag_imag+1.2; Xag_imag-0.6 Yag_imag-0.4; Xag_imag+0.6 Yag_imag-0.4; Xag_imag Yag_imag+1.2];
    imaglocVertices(:,1) = imaglocVertices(:,1)-Xag_imag;
    imaglocVertices(:,2) = imaglocVertices(:,2)-Yag_imag;
    R = [cos(HDarr2) -sin(HDarr2); sin(HDarr2)  cos(HDarr2)];
    pVim = R*imaglocVertices';
    plot(Xag_imag+pVim(1,:),Yag_imag+pVim(2,:),'r'); hold on
end
if imag_flag && Xag_imag_nav>0 && simflag~=242
    imaglocVertices = [Xag_imag_nav Yag_imag_nav+1.2; Xag_imag_nav-0.6 Yag_imag_nav-0.4; Xag_imag_nav+0.6 Yag_imag_nav-0.4; Xag_imag_nav Yag_imag_nav+1.2];
    R = [cos(HDarr2) -sin(HDarr2); sin(HDarr2)  cos(HDarr2)];
    imaglocVertices(:,1) = imaglocVertices(:,1)-Xag_imag_nav;
    imaglocVertices(:,2) = imaglocVertices(:,2)-Yag_imag_nav;
    pVim = R*imaglocVertices';
    plot(Xag_imag_nav+pVim(1,:),Yag_imag_nav+pVim(2,:),'r'); hold on
end
plot(ObjCenX,ObjCenY,'o','MarkerFaceColor',Ocolor,'MarkerEdgeColor',Ocolor,'MarkerSize',5);hold on
if simflag == 28
    plot(CoordsTMPmem(1,1),CoordsTMPmem(1,2),'o','MarkerEdgeColor','r','MarkerSize',5);hold on
end
k=1:length(ObjCenX);
for kk = k
    hh = text(ObjCenX(kk)-1.2,ObjCenY(kk)-1.2,num2str(k(kk)),'Color',Ocolor);   hh.FontSize = 7;
end

if simflag == 22 && imag_flag
    plot(oldCoordsTMP(1,1),oldCoordsTMP(1,2),'ro');hold on
    k=1:length(ObjCenX);
    for kk = k
        hh = text(oldCoordsTMP(1,1)-1.2,oldCoordsTMP(1,2)-1.2,num2str(k(kk)),'Color','r');   hh.FontSize = 5;
    end
end

if Xag_imag_nav>0
    plot(AllImagX,AllImagY,':')
end



if simflag == 40 || simflag == 50     
    GCplotrate = GC2PCwts*GC_rate;
    if percep_flag == 1
        GCplotrate = GCplotrate*0.1;
    end
    subplot(plty,pltx,GC_range,'replace');
    %surface(HX,HY,reshape(GCplotrate,NHx,NHy)');view(30,45);set(gca,'BoxStyle','full','Box','on');
    surface(HX*2,HY*2,reshape(GCplotrate,NHx,NHy)');colormap(gca,gray);
    shading interp;
    %ylabel('cell#');
    xlabel('cell#');axis tight;%zlim([0 50]);%set(gca,'XDir','reverse');
    title('W_{GC2PC} * r_{GC} (x,y)','FontWeight','Normal');
    %colorbar;
end



subplot(plty,pltx,PC_range,'replace');
%surface(HX*2,HY*2,reshape(H_rate,NHx,NHy)','EdgeColor','none');%colorbar
surface(HX*2,HY*2,reshape(H_rate,NHx,NHy)');%colorbar
axis tight
shading interp;
xlabel('cell#');
ylabel('cell#')
%ax = gca; ax.XTick = [0.5 43.5]; ax.XTickLabel = {'0','44'};
%ax.YTick = [0.5 43.5]; ax.YTickLabel = {'0','44'};
caxis([0 1]);
%axis square;
text(16,46,['PC rates']);



if simflag==50 
    x_Gauss = -5:5;
    y_Gauss = exp(-(x_Gauss.^2)/3);
    thresh  = 0.1;
    if ~isempty(Hmem1)
        T           = length(Hmem1(1,:));
        Q           = length(Hmem1(:,1));
        tmp1        = Hmem1;
        for j = 1:Q
            for i = 1:T
                if i<=5
                    lr = i-1;
                else
                    lr = 5;
                end
                if i>=T-5
                    rr = T-i;
                else
                    rr = 5;
                end
                tmp1(j,i)  = sum( y_Gauss(-lr+6:rr+6) .* Hmem1(j,i-lr:i+rr) );
            end
        end
        tmp1 = round(tmp1*100)/100;
        tmp1 = tmp1/max(max(tmp1));
        NEWorder = zeros(length(tmp1(:,1)),1);
        MAXloc   = zeros(length(tmp1(:,1)),1);
        for i = 1:length(tmp1(:,1))
            onerow    = tmp1(i,:);
            colmaxind = find(onerow==max(onerow));
            if length(colmaxind)>1
                colmaxind = colmaxind(floor(length(colmaxind)/2));
            end
            MAXloc(i)   = colmaxind;
            NEWorder(i) = i;
        end
        [SortedByMAXloc,NEWorder] = sort(MAXloc);
        tmp11   = tmp1(NEWorder,:);
        indvec2 = max(tmp11,[],2)>thresh;
        subplot(plty,pltx,PP1_range,'replace');
        imagesc(tmp11(indvec2,:));colorbar; colormap(gca,jet); axis xy
        ax = gca;
        xUnit = round(200/44); 
        maxX  = round(T*200/44);
        ax.XTick = [1:T-1:T]; ax.XTickLabel = {'1',num2str(maxX)};
        xlabel('y_{new} [cm]')                
        ylabel('cell#')
        title('r_{resPCs} planning','FontWeight','Normal');
    end
    if ~isempty(Hmem2)      
        T           = length(Hmem2(1,:));
        Q           = length(Hmem2(:,1));
        tmp2        = Hmem2;
        for j = 1:Q
            for i = 1:T
                if i<=5
                    lr = i-1;
                else
                    lr = 5;
                end
                if i>=T-5
                    rr = T-i;
                else
                    rr = 5;
                end
                tmp2(j,i)  = sum( y_Gauss(-lr+6:rr+6) .* Hmem2(j,i-lr:i+rr) );
            end
        end
        tmp2 = round(tmp2*100)/100;
        tmp2 = tmp2/max(max(tmp2));
        if step>=5780
           test=1; 
        end
        tmp22   = tmp2(NEWorder,:);
        indvec2 = max(tmp22,[],2)>thresh;        
        subplot(plty,pltx,PP2_range,'replace');
        imagesc(tmp22(indvec2,:));colorbar; colormap(gca,jet); axis xy
        ax = gca;
        xUnit = round(200/44); 
        maxX  = round(T*200/44);
        ax.XTick = [1:T-1:T]; ax.XTickLabel = {'1',num2str(maxX)};
        xlabel('y_{new} [cm]')                
        %ylabel('cell#')
        title('r_{resPCs} percep.','FontWeight','Normal');
    end
    if ~isempty(Hmem3)
        T           = length(Hmem3(1,:));
        Q           = length(Hmem3(:,1));
        tmp3        = Hmem3;
        for j = 1:Q
            for i = 1:T
                if i<=5
                    lr = i-1;
                else
                    lr = 5;
                end
                if i>=T-5
                    rr = T-i;
                else
                    rr = 5;
                end
                tmp3(j,i)  = sum( y_Gauss(-lr+6:rr+6) .* Hmem3(j,i-lr:i+rr) );
            end
        end
        tmp3 = round(tmp3*100)/100;
        tmp3 = tmp3/max(max(tmp3));
        tmp33   = tmp3(NEWorder,:);
        indvec2 = max(tmp33,[],2)>thresh;
        subplot(plty,pltx,PP3_range,'replace');
        imagesc(tmp33(indvec2,:));colorbar; colormap(gca,jet); axis xy
        ax = gca;
        xUnit = round(200/44); 
        maxX  = round(T*200/44);
        ax.XTick = [1:T-1:T]; ax.XTickLabel = {'1',num2str(maxX)};
        xlabel('y_{new} [cm]')                
        %ylabel('cell#')
        title('r_{resPCs} recall','FontWeight','Normal');
    end
end



set(findall(gcf,'-property','FontSize'),'FontSize',12)



PLTposX = 0.5; 
PLTposY = 0.98;
Tsize   = 16;  



if plt_ttl_flag==1  % change plot title for video frames
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Perceptually driven MTL (bottom-up mode)','fontsize',Tsize,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');
end
if plt_ttl_flag==2
    if simflag==21
        ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
        text(PLTposX, PLTposY,'\bf Imagery (top-down mode)','fontsize',Tsize,'color',Goran,'HorizontalAlignment','center','VerticalAlignment', 'top');        
    else
        ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
        text(PLTposX, PLTposY,'\bf Imagery (top-down mode) - "Where did I leave my keys?"','fontsize',Tsize,'color',Goran,'HorizontalAlignment','center','VerticalAlignment', 'top');
    end
end
if plt_ttl_flag==241 || plt_ttl_flag==242
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode) - "Recall for comparison with perceived object positions"','fontsize',Tsize,'color',Goran,'HorizontalAlignment','center','VerticalAlignment', 'top');
end
if plt_ttl_flag==7
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Sampling scene elements in imagery - "What object was to the right?"','fontsize',Tsize,'color',Gblue,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==77
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Sampling scene elements in imagery - "What object was to the right?"','fontsize',Tsize,'color',Gblue,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==3
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Re-estabish perceptually driven representation','fontsize',Tsize,'color',myGrey,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==5
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf recall test','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==6
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Mental navigation (top-down mode)','fontsize',Tsize,'color',Goran,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==8
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode) - where was I last time I saw object 1?','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==9
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode) - where was object 1 in peripersonal space?','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if simflag == 241 
    simflag = 13;
end
if simflag == 242 
    simflag = 14;
end
if simflag == 28 
    simflag = 22;
end
if simflag == 26 
    simflag = 21;
end

timestring = ['time: ' num2str(step*0.001,'%#5.2f') ' s'];
text(0.94, 0.97, timestring,'fontsize',12,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');

simstring = ['Sim. ' num2str(simflag/10,'%#5.1f') ''];
text(0.047, 0.97,simstring,'fontsize',12,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');

 frame = getframe(gca);

 
 
