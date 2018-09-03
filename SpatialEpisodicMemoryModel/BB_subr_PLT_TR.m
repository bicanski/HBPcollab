
% Plotting subroutine for the BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752
%


function [frame]  = BB_subr_PLT_TR(step, modstep, plt_ttl_flag, percep_flag, simflag, ...
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
    imag_flag, Xag_imag_nav, Yag_imag_nav, ...
    oldCoordsTMP, AllImagX, AllImagY, VBX_rt, VBY_rt, L_r, ...
    Hadapt, I_comp, ...
    Hmem1, Hmem2, Hmem3, oPRCue, ...
    TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13, TR14, TR15, TR16, TR17, TR18, TR19, TR20, ...
    oTR1, oTR2, oTR3, oTR4, oTR5, oTR6, oTR7, oTR8, oTR9, oTR10, oTR11, oTR12, oTR13, oTR14, oTR15, oTR16, oTR17, oTR18, oTR19, oTR20);



IMsizeX = 1350;%1620
IMsizeY = 825;%990
if strcmp(computer,'MACI64')
    IMsizeX = 1620/2; 
    IMsizeY = 990/2; 
end

if step == modstep
    fig = figure('position',[100 100 IMsizeX IMsizeY]);
else
    fig = gca;
end

colormap hot

Olive = [6/255, 177/255, 82/255];
Gblue = [44/255, 102/255, 212/255];
Rpurp = [128/255, 47/255, 202/255];
Ocolor = Olive;
if simflag == 50;
    Ocolor = Rpurp;
end

pltx = 70;
plty = 50;

HD_range = [20*pltx+30+(1:10) 21*pltx+30+(1:10) 22*pltx+30+(1:10) 23*pltx+30+(1:10) 24*pltx+30+(1:10) 25*pltx+30+(1:10) 26*pltx+30+(1:10) 27*pltx+30+(1:10) 28*pltx+30+(1:10) 29*pltx+30+(1:10)];

PWb_range = [6*pltx+1+(1:10) 7*pltx+1+(1:10) 8*pltx+1+(1:10) 9*pltx+1+(1:10) 10*pltx+1+(1:10) 11*pltx+1+(1:10) 12*pltx+1+(1:10) 13*pltx+1+(1:10) 14*pltx+1+(1:10) 15*pltx+1+(1:10)];
ego_range = [31*pltx+1+(1:10) 32*pltx+1+(1:10) 33*pltx+1+(1:10) 34*pltx+1+(1:10) 35*pltx+1+(1:10) 36*pltx+1+(1:10) 37*pltx+1+(1:10) 38*pltx+1+(1:10) 39*pltx+1+(1:10) 40*pltx+1+(1:10)];

BVC_range = [6*pltx+58+(1:10) 7*pltx+58+(1:10) 8*pltx+58+(1:10) 9*pltx+58+(1:10) 10*pltx+58+(1:10) 11*pltx+58+(1:10) 12*pltx+58+(1:10) 13*pltx+58+(1:10) 14*pltx+58+(1:10) 15*pltx+58+(1:10)];
allo_range = [31*pltx+58+(1:10) 32*pltx+58+(1:10) 33*pltx+58+(1:10) 34*pltx+58+(1:10) 35*pltx+58+(1:10) 36*pltx+58+(1:10) 37*pltx+58+(1:10) 38*pltx+58+(1:10) 39*pltx+58+(1:10) 40*pltx+58+(1:10)];

for i = 1:20
    ang_step = 2*pi/20;    
    x_plt    = round(20*cos((i)*ang_step));
    y_plt    = round(20*sin((i)*ang_step));
    x_cen    = pltx/2 + x_plt;
    y_cen    = plty/2 - y_plt;
    TRind = i+16;
    if TRind>20   TRind = TRind-20;   end
    subplot(plty,pltx, [ y_cen*pltx+((x_cen-2):(x_cen+2)) (y_cen-1)*pltx+((x_cen-2):(x_cen+2)) (y_cen+1)*pltx+((x_cen-2):(x_cen+2)) (y_cen+2)*pltx+((x_cen-2):(x_cen+2)) ],'replace');
    eval(['surface(BVCX,BVCY,reshape(TR' int2str(TRind) ',NBVCR,NBVCTheta)'');']);%colorbar
    caxis([0 1]);
    shading interp;
    axis square;
    box off
    axis off
    text(-34,17,['TR' int2str(TRind) '']);
end



subplot(plty,pltx,HD_range,'replace');
ax = gca;
surface(HDX,HDY,[HD_rate HD_rate]);%colorbar
shading interp;
xlabel('HD Rates')
set(ax,'ylim',[-1.6 1.6]);
set(ax,'xlim',[-1.6 1.6]);
axis square;
text(0,-1.8-0.1,'S');   text(0,1.8-0.1,'N');   text(-1.8-0.1,0,'W');   text(1.8-0.1,0,'E');
ax.XTick = [];   ax.YTick = [];
box off
axis off
text(-2.2,1.4,['HDC  '; 'rates']);%,'FontWeight','bold');%, 'Interpreter', 'latex');



subplot(plty,pltx,PWb_range,'replace');
surface(BVCX,BVCY,reshape(PW_rate,NBVCR,NBVCTheta)');%colorbar
shading interp;
axis square;
box off
axis off
text(-4,-18.5,'Behind');   text(-4,18,'Ahead');   text(-19,0,'L');   text(17,0,'R');
text(-25,15,['PWb  '; 'rates']);



subplot(plty,pltx,ego_range,'replace');
plot(VBX_rt,VBY_rt,'k.'); hold on;
plot(0,0,'^','MarkerFaceColor','k','MarkerEdgeColor','k','MarkerSize',3.5);hold on
plot(VBX_rt(VBY_rt>0),VBY_rt(VBY_rt>0),'.','color',Rpurp);hold on
xlabel('egoc. agent view');
if imag_flag
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
box off
text(-18,-26,['egoc. agent view']);



subplot(plty,pltx,BVC_range,'replace');
surface(BVCX,BVCY,reshape(BVC_rate,NBVCR,NBVCTheta)'); hold on;%colorbar
shading interp;
axis square;
box off
axis off
text(-1,-18.5,'S');   text(-1,18,'N');   text(-19.5,0,'W');   text(17,0,'E');
text(-25,15,['BVC  '; 'rates']);



subplot(plty,pltx,allo_range,'replace');
plot(BX,BY,'k.','LineWidth',1.5); hold on
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
%         if simflag == 50
%             [Xb,Yb] = meshgrid(Xblack,Yblack);
%             plot(Xb,Yb,'k.')
%         end
xlabel('alloc. agent position');
set(gca,'ylim', [-1 23]);
set(gca,'xlim', [-1 23]);
axis square;
axis off
box off
text(0,-3,['alloc. agent position']);
if imag_flag && Xag_imag_nav<0
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
if imag_flag && Xag_imag_nav>0
    imaglocVertices = [Xag_imag_nav Yag_imag_nav+1.2; Xag_imag_nav-0.6 Yag_imag_nav-0.4; Xag_imag_nav+0.6 Yag_imag_nav-0.4; Xag_imag_nav Yag_imag_nav+1.2];
    R = [cos(HDarr2) -sin(HDarr2); sin(HDarr2)  cos(HDarr2)];
    imaglocVertices(:,1) = imaglocVertices(:,1)-Xag_imag_nav;
    imaglocVertices(:,2) = imaglocVertices(:,2)-Yag_imag_nav;
    pVim = R*imaglocVertices';
    plot(Xag_imag_nav+pVim(1,:),Yag_imag_nav+pVim(2,:),'r'); hold on
end
plot(ObjCenX,ObjCenY,'o','MarkerFaceColor',Ocolor,'MarkerEdgeColor',Ocolor,'MarkerSize',5);hold on
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



PLTposX = 0.5; 
PLTposY = 1.0;
Tsize   = 14;  



if plt_ttl_flag==1  % change plot title for video frames
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Perceptually driven MTL (bottom-up mode)','fontsize',Tsize,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');
end
if plt_ttl_flag==2
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode): "Where did I leave my keys?"','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end
if plt_ttl_flag==7
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery sampling: "What object was to the right?"','fontsize',Tsize,'color',Rpurp,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==77
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery sampling: "What is that thing on the other side?"','fontsize',Tsize,'color',Rpurp,'HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==3
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf re-estabish perceptually driven representation','fontsize',Tsize,'color','g','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==5
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf recall test','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==6
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf mental navigation (top-down mode)','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==8
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode): where was I last time I saw object 1?','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

if plt_ttl_flag==9
    ha = axes('Position',[0 0 1 0.98],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
    text(PLTposX, PLTposY,'\bf Imagery (top-down mode): where was object 1 in peripersonal space?','fontsize',Tsize,'color','r','HorizontalAlignment','center','VerticalAlignment', 'top');
end

timestring = ['time: ' num2str(step*0.001,'%#5.2f') ' s'];
text(0.94, 0.97, timestring,'fontsize',12,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');

simstring = ['Sim. ' num2str(0,'%#5.1f') ''];
text(0.047, 0.97,simstring,'fontsize',12,'color','k','HorizontalAlignment','center','VerticalAlignment', 'top');


frame = getframe(gca);



