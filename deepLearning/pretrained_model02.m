clc; 
clear;

% import googlenet
deepnet =  resnet101;

%% show the structure of network
analyzeNetwork(deepnet)

%% show the class name
categorynames = deepnet.Layers(end).ClassNames;

%% check the input layer
inlayer = deepnet.Layers(1);
insz = inlayer.InputSize;

%% 
% read image
img1 = imread("dog.jpg");
img1 = imresize(img1,[224 224]);

%% input image to googlenet
[pred, scores] = classify(deepnet, img1);

%% show result
bar(scores)

%%
highscores = scores > 0.01;
bar(scores(highscores))
xticklabels(categorynames(highscores))

%%
imshow(img1)
text(10, 20, char(pred), 'Color', 'white')
