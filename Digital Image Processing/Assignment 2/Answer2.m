%%%%%%%%%%%%%%%%%%compute histogram and T(r)
x=double(imread('Q2-input image.tif')); 			% inpute image
[M,N]=size(x);
for i=0:255
	h(i+1)=sum(sum(x==i)); % histogram of input image
end
% compute hist equalization
y=x; % initialize output image y
s=sum(h);
figure; subplot(1,2,1); imshow(imread('cameraman.tif')); %show input image
title(' input image');
subplot(1,2,2); imhist(imread('cameraman.tif')) % show input hist
title('Input image hist ');

%transfer function
for r = 0:255
	T(r+1)=uint8(sum(h(1:r))/s*255); %T(r)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


x = double(imread('UnsharpmaskedImg_Q2-input image.jpg'));  	% specified image, UnsharpmaskedImg_Q2-input enhanced

[M,N]=size(x);
for i=0:255
	h1(i+1)=sum(sum(x==i));
end
% compute hist equalization
s=sum(h1);
%figure, imshow(imread('x5.bmp')) %show specified image
figure, subplot(1,2,1), imshow(uint8(x))
title('Specified image')
subplot(1,2,2), imhist(uint8(x)) % show specified hist
title('Specified image hist')

for z = 0:255
	G(z+1)=uint8(sum(h1(1:z))/s*255); %G(z)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[pix] = unique(y); % find all indices of unique pixel values
matchedY = y; %initialize matched output as matchedY
for i =1:numel(pix)
	diffpix = abs(double(T(pix(i)+1))-double(G)); % for every unique pixel value in 'y' find difference of T(r) and G(z)
	[~,ind]=min(diffpix); % find index of nearest value of a unique pixel value in 'y' in G(z)
	val = ind-1; %take the pixel value corresponding to index obtained in previous line
	I=find(y==pix(i)); % find all indices of pixels with value 'pix(i)' in 'y'
	matchedY(I)=val; % substitute pixel value 'val' in the indices obtained in previous line
end
figure, subplot(1,2,2), imshow(uint8(matchedY)) %show matched image
title('Hist Matched image')
subplot(1,2,1), imhist(uint8(matchedY)) % show matched hist
title('Matched image hist')