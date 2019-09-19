%%%UnsharpMasking
f = imread('Chandrayaan2 - Q3a-inputimage.png');
input_size = size(f)
w = ones(7, 7)/49;  %Blur(box) filter
f = double(f);
Blur = convn(f, w, 'same');
x = f - Blur;
Output3a = f + x;
save('Out3a.mat', 'Output3a');
output_size = size(Output3a);
figure, imshow(uint8(f)), title('orig');
figure, imshow(uint8(Output3a)), title('UnsharpmaskedImg.png');
imwrite(uint8(Output3a),'UnsharpmaskedImg.jpg','Quality',100); % save output image
