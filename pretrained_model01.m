clc; 
clear;

% import googlenet
deepnet = googlenet;

% show the structure of network
analyzeNetwork(deepnet)
%%
% read image
img1 = imread("1.jpg");
imshow(img1);

% resize image
img1 = imresize(img1,[224 224]);

% input image to googlenet
pred1 = classify(deepnet, img1);

fprintf('%s\n', pred1);