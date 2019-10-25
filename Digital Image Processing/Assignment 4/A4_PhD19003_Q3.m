%%Name: Ashima 
%%Roll No: PhD19003

close all
clear all
clc
%% Read and Display Original Image
ffull = imread('cameraman.tif');
f = (ffull(1:256,1:256)); % image to be degraded
imshow(f,[])
title('Original Image')

%% AWGN
noisevar = 50;
g = f + uint8(sqrt(noisevar).*randn(256));
%g = f + sqrt(noisevar).*uint8(30*randn(256));                        
gim = g;
figure, imshow(gim,[]) % Show observed image
title('Observed Image')  %Gaussian Noise Added to Original Image

%% Restored Image
f1 = double(g);   % Initialize output image = observed image
m = 3;            % local window size
padlength = floor(m/2);
f_hat = padarray(f1, [padlength, padlength], 0, 'both')

s1 = size(f_hat,1); s2 = size(f_hat,2);
for i = 1:s1-m+1
    for j = 1:s2-m+1
        windows = f_hat(i:i+m-1, j:j+m-1);
        localmean = mean(mean(windows));
        localvar = var(windows(:));
        var_ratio = noisevar/localvar;
        if var_ratio > 1
            var_ratio = 1;
        end
        x = f_hat(i+padlength, j+padlength) - ((var_ratio)*(f_hat(i+padlength, j+padlength)-localmean));     
        f1(i,j) = x;
    end
end
% 
fim = uint8(f1);
figure, imshow(fim), title('filtered image')

%% SNR Filtered Image
mse = mean(mean((f1-double(f)).^2));
snrfiltered = 20*log10(255/(sqrt(mse)))        % SNR restored inverse filtering

%% SNR Observed Image
mse = mean(mean((double(g)-double(f)).^2));
snrobserved = 20*log10(255/(sqrt(mse)))        % SNR Observed Image