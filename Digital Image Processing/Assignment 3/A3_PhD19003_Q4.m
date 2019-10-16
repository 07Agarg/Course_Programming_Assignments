%%Name: Ashima 
%%Roll No: PhD19003
%unsharp masking via DFT.
clear all;
clc;
close all;
w = ones(3, 3)/9;      %Box filter 3*3
f = imread('Chandrayaan2_img.png');  
%crop image
f = f(1:512, 1:512);


%Zero Pad image and filter
w_pad=padarray(w,[511, 511],0,'post');
f_pad=padarray(f,[2, 2],0,'post');

%Perform FFT2 on both
Fp=fft2(double(f_pad));  %, P, Q); % FFT of image fp
Wp=fft2(double(w_pad));  %, P, Q); % FFT of kernel hp

% Hadamard product
H1=Fp .* Wp;

%Subtract the product from FFT2 of zero padded image
Hp = Fp - H1;

% Add the resultant to FFT2 of zero padded image
Sp = Hp + Fp;

%Take IFFT2 of Subtracted Image, followed by real operation
gp=ifft2(Hp); % Inverse FFT
gp=real(gp); % Take real part
%Crop 512x512 from top left.
g = gp(1:512, 1:512);
imshow(g,[])  %%% Laplacian image
title('Laplacian image')
%imwrite(uint8(g),'Laplacian.jpg','Quality',100); % save output image

%Take IFFT2 of the addition resultant, followed by real operation
sp = real(ifft2(Sp));
Sharpened = sp(1:512, 1:512);
figure,imshow(uint8(Sharpened))   %%%Sharpened image
title('sharpened image')
imwrite(uint8(Sharpened),'UnsharpMasking.jpg','Quality',100); % save output image

% show original image
figure
imshow(f,[])   %%  Original image
title('original image')