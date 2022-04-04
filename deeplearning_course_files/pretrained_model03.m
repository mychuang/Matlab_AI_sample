clc; 
clear;
%% import data
% imageDatastore: 讀取指定路徑下的所有文件, 輸出ImageDatastore變數
flowerds = imageDatastore("Flowers","IncludeSubfolders",true,"LabelSource","foldernames");

% Split into training and testing sets
% 輸入 ImageDatastore，拆分資料
[trainImgs,testImgs] = splitEachLabel(flowerds, 0.6);

numClasses = numel(categories(flowerds.Labels));

%% Create a network by modifying GoogLeNet
net = googlenet;
lgraph = layerGraph(net);

% Modify the classification and output layers
newFc = fullyConnectedLayer(12,"Name","new_fc");
lgraph = replaceLayer(lgraph, "loss3-classifier", newFc);
newOut = classificationLayer("Name", "new_out");
lgraph = replaceLayer(lgraph, "output", newOut);

%% train model
options = trainingOptions("sgdm", "InitialLearnRate", 0.001);

[flowernet, info] = trainNetwork(trainImgs, lgraph, options);

%%

