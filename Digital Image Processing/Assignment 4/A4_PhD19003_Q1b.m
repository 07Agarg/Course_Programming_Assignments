%%Name: Ashima 
%%Roll No: PhD19003
close all
clear all
clc
%% Read and Display Original Image
ffull = imread('cameraman.tif');
f = (ffull(1:256,1:256)); % image to be degraded
imshow(f,[])
title('orig')

%% zero padding
M = 280;N=280;
fz=zeros(M,N);hp=fz;
fz(1:256,1:256) = f;

%% Laplacian for smothness
hp(1,1)=-8; hp(2,1)=1; hp(1,2)=1; % Center is at (1,1)
hp(M,1)=1; hp(1,N)=1; % Indices modulo P or Q
hp(M,2) = 1; hp(2,N) = 1;hp(2,2)=1;hp(M,N) = 1;
hp = hp;

%% degradation function
hz = zeros(3);
h=[1.6 2.9 0;1.3 1 0; 0 0 0;];
h = h./sum(sum(h));
hz(1:3,1:3) = h;

%% using fft to obtain conv
G = fft2(h,M,N).*fft2(f,M,N) ; % extension is taken care by FFT = H.*F
gspace = real(ifft2(G,M,N)) ;
g = gspace(1:256,1:256);   

%% AWGN
g = uint8(g) + uint8(30*randn(256)); % get h*f + n = HF + N
G = fft2(g,M,N); %% FFT to get Fourier of observed image
gim = g;
figure, imshow(gim,[]) % Show observed image
title('observed')

%%
H = fft2(h,M,N); % FFt of impulse response/PSF
Hp = fft2(hp,M,N); %%% for constrained filtering - laplace filter 

%% Declare C and Lambda 
C =0.001:.04:4;err=zeros(1,length(C));
lambda = 0.01;
%%
for i = 1:length(C)
    F = conj(H).*G./(abs(H).^2 + C(i) + lambda .* (abs(Hp).^2) .* ((abs(H).^2) + C(i)));
    f1 = real(ifft2(F));
    fim = f1(1:256,1:256); %5 best restored image
    mserestore = mean(mean((fim-double(f)).^2));
    errconst(i) = mserestore;
end

%% Show best restored constrained
[val,ind] = min(errconst);
F = conj(H).*G./(abs(H).^2 + C(ind) + lambda * (abs(Hp).^2) * ((abs(H).^2) + C(ind)));
f1 = real(ifft2(F));
fim = f1(1:256,1:256); %5 best restored image
figure,imshow(fim,[])
title('best filteirng')
snr_filtered = 20*log10(255/(sqrt(min(errconst)))); % SNR for restored

%% Degraded image
mse = mean(mean((double(gim)-double(f)).^2));
snr_degraded = 20*log10(255/(sqrt(mse))); % SNR degraded