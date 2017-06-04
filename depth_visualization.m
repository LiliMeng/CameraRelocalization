clear all;
close all;
depthImage=imread('frame-000353.depth.png'); 
[filledImage,zeroPixels] = Kinect_DepthNormalization(depthImage); 
figure; 
 
normImage = single( depthImage ) ./1000;
normImage(normImage > 6.0) = 6.0;
depth=imagesc(normImage);
colorbar
%figure; 
%imagesc(filledImage); 
%figure; 
%imshowpair(depthImage,filledImage,'montage');
