%REFERENCE: CODE GIVEN IN THE LECTURE SLIDES IS USED FOR THIS QUESTION 

clear all
clc
filename = 'TransformationMatrix1.txt';	
T = importdata(filename);    %import T matrix obtained from Q3
imgZ = imread('cameraman.tif');
[r,c] = size(imgZ);

% create array of destination x,y coordinates
[X,Y]=meshgrid(0:c-1,0:r-1);

%Calculate Destination coordinates
Z = ones(c,r);
destCoor =[X(:) Y(:) Z(:)]*(T);
% calculate nearest neighbor interpolation 
sizeOfDestcoor = size(destCoor)
Xd = (destCoor(:,1));
Yd = (destCoor(:,2));

% calculate new image
figure
[Xq,Yq]=meshgrid(-2*c:2*c,-2*r:2*r);  %output meshgrid

%griddata(knownPoints, knownValues, queryPoints)
Vq = griddata(Xd(:),Yd(:),double(imgZ(:)),Xq(:),Yq(:));
V = reshape(Vq,4*c+1,4*r+1);

%%%ROTATED IMAGE AFTER APPLYING TRANSFORMATION (T) 
imshow(V,[])
imwrite(uint8(V),'RotateImage.jpg','Quality',100) % save output image


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PART 2
%LOAD THE INPUT IMAGE 
figure
imgful = imread('RotateImage.jpg');
%imshow(imgful, [])
[r1, c1] = size(imgful)

%crop the rotated image to last quadrant only. 
rotatedImg = imcrop(imgful,[512, 512, r1, c1]);
[r, c] = size(rotatedImg)
figure
imshow(rotatedImg, [])

[X, Y] = meshgrid(0:c-1, 0:r-1);

% calculate query coordinates
Z = ones(c, r);

sourceCoor =[X(:) Y(:) Z(:)]*inv(T);
% get only x and y coordinates
Xs = (sourceCoor(:,1));
Ys = (sourceCoor(:,2));

% calculate new image
figure
[Xq,Yq]=meshgrid(0:c-1,0:r-1);  %input meshgrid  query points

Vq = griddata(Xs(:),Ys(:),double(rotatedImg(:)),Xq(:),Yq(:));
V = reshape(Vq, c, r);

%%%%INVERSION OF THE ROTATED IMAGE => ORIGINAL IMAGE
imshow(V,[])
imwrite(uint8(V),'InvT.jpg','Quality',100) % save output image