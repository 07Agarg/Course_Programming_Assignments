%%Name: Ashima 
%%Roll No: PhD19003
%2-D convolution between f and w using DFT.
close all
clear all
clc
f = [1, 3, 4; 2, 5, 3; 6, 8, 9];
%given W
w = [-1, -2, -3; -4, 0, 1; -6, -5, -1];

%Arrange W
W_new = [0, 1, 0, 0, -4; -5, -1, 0, 0, -6; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0; -2, -3, 0, 0, -1];
%f_pad
f_pad = [1, 3, 4, 0, 0; 2, 5, 3, 0, 0; 6, 8, 9, 0, 0; 0, 0, 0, 0, 0; 0, 0, 0, 0, 0];

%conv with dft
out_withdft = real(ifft2(fft2(f_pad).*fft2(W_new)))

%conv without dft
out_withoutdft = conv2(f, w)
