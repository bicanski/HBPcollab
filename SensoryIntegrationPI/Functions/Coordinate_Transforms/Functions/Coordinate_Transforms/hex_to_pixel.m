function X = hex_to_pixel(IJ,Phi)

T_i = 0;  % Relative orientation of the
T_j = 120; % two new coordinate axes

T_i = T_i + Phi;
T_j = T_j + Phi;
    
Ai =  [  sind(T_j)/(cosd(T_i)*sind(T_j) - sind(T_i)*cosd(T_j)), -sind(T_i)/(cosd(T_i)*sind(T_j) - sind(T_i)*cosd(T_j));
        -cosd(T_j)/(cosd(T_i)*sind(T_j) - sind(T_i)*cosd(T_j)),  cosd(T_i)/(cosd(T_i)*sind(T_j) - sind(T_i)*cosd(T_j))];

% Transform back to XY
X = [Ai(1,1)*IJ(:,1) + Ai(1,2)*IJ(:,2),...
     Ai(2,1)*IJ(:,1) + Ai(2,2)*IJ(:,2)];
    
end