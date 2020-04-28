

% Subroutine for the BB model of spatial cognition (2014-2018) 
% Code developed by Andrej Bicanski (andrej.bicanski@gmail.com)
% 
% published in eLife
%
% Bicanski A, Burgess N. A neural-level model of spatial memory and imagery. Elife. 2018;7:e 33752. 
% DOI: 10.7554/eLife.33752


% Make connections between PCs and GCs


load('GC_FRmaps_BB.mat')

res      = 0.5;
Xmax     = 22;
Ymax     = 22;
N_GCs    = 100*7;
N_PCs    = 44^2;
GC2PCwts = zeros(N_PCs,N_GCs);
sFRm     = size(GC_FRmaps_SD);
PCtmplte = zeros(sFRm(1),sFRm(2));

for x = (res:res:Xmax)*2
    for y = (res:res:Ymax)*2
        PC = PCtmplte;
        PC(x,y) = 1;
        for i = 1:7
            for j = 1:100
                GC2PCwts((x-1)*44+y,(i-1)*100+j) = max(max(PC.*GC_FRmaps_SD(:,:,j,i)));
            end
        end 
    end
end

save GC2PCwts_BB GC2PCwts



