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
lgraph = layerGraph(net); % 獲取神經網路資訊

% 創建/替換 fullyConnectedLayer，輸出為12個神經元
newFc = fullyConnectedLayer(numClasses, "Name", "new_fc"); 
lgraph = replaceLayer(lgraph, "loss3-classifier", newFc);

% 創建/替換 分類層(輸出)
newOut = classificationLayer("Name", "new_out");
lgraph = replaceLayer(lgraph, "output", newOut);

%% train model
options = trainingOptions("adam","Plots","training-progress", ...
    "MiniBatchSize",64,"InitialLearnRate",0.001);

[flowernet, info] = trainNetwork(trainImgs, lgraph, options);

%% 預測
testpreds = classify(flowernet, testImgs);

% 評估
accuracy = nnz(testpreds == testImgs.Labels) / numel(testpreds);
fprintf('accuracy %f\n', accuracy);

confusionchart(testImgs.Labels, testpreds);
