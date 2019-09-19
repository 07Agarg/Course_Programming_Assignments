%3c
clear all;
clc;
f = imread('Chandrayaan2 - Q3a-inputimage.png');
f = double(f);
I = zeros(7, 7) ;  %Impulse function
I(4, 4) = 1;
w1 = ones(7, 7)/49;  %Blur(box) filter
%size(f)
W = 2*I - w1;
Output3c = convn(f, W, 'same');
save('Out3c.mat', 'Output3c')
output_size = size(Output3c);
figure, imshow(uint8(f)), title('orig');
figure, imshow(uint8(Output3c)), title('UnsharpmaskedImg-3c.png');
imwrite(uint8(Output3c),'UnsharpmaskedImg-3c.jpg','Quality',100); % save output image
