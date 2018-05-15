function UVW = hex_to_cube(IJ)

UVW = [IJ(:,1),...
       IJ(:,2),...
       -(IJ(:,1) + IJ(:,2))];

end