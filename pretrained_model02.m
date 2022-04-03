clc; 
clear;

% import googlenet
deepnet =  resnet101;

% show the structure of network
analyzeNetwork(deepnet)

%% display layer names
deepnet.Layers

%% 
% read image
img1 = imread("1.jpg");
img1 = imresize(img1,[224 224]);

% input image to resnet101
pred1 = classify(deepnet, img1);

%% Show the image and the classification results figure

imshow(img1)
text(10, 20, char(pred1), 'Color', 'white')

