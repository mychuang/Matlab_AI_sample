clc; 
clear;
% import data
% imageDatastore: 讀取指定路徑下的所有文件, 輸出ImageDatastore變數
flowerds = imageDatastore("Flowers","IncludeSubfolders",true,"LabelSource","foldernames");

% Split into training and testing sets
% 輸入 ImageDatastore，拆分資料
[trainImgs, testImgs] = splitEachLabel(flowerds, 0.6);

numClasses = numel(categories(flowerds.Labels));

%% Create a network by modifying darknet19
net = darknet19;
lgraph = layerGraph(net); % 獲取神經網路資訊
%analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;

%% augmentedImageSource()用於數據增強
%第一個參數表示輸出的大小，第二個参数輸入數據
trainds = augmentedImageDatastore(inputSize, trainImgs);
testds = augmentedImageDatastore(inputSize, testImgs);

%% 重構網路進行遷移學習
% Find the names of the two layers to replace.
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses,'Name','new_fc', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);

elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1, numClasses, 'Name','new_conv',...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
end

lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);

% 創建/替換 分類層(輸出)
newOut = classificationLayer("Name", "new_out");
lgraph = replaceLayer(lgraph, classLayer.Name, newOut);

%%  plot the new layer graph and zoom in on the last layers of the network
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% train model
options = trainingOptions("adam","Plots","training-progress", ...
    "MiniBatchSize",64,"InitialLearnRate",0.001, "MaxEpochs", 100, ...
    'Shuffle','every-epoch', "Verbose",true, ...
    "ValidationData",testds, "ValidationFrequency", 30);

%%
[flowernet, info] = trainNetwork(trainds, lgraph, options);

%% 預測
testpreds = classify(flowernet, testds);

%%
% 評估
accuracy = nnz(testpreds == testImgs.Labels) / numel(testpreds);
fprintf('accuracy %f\n', accuracy);

confusionchart(testImgs.Labels, testpreds);

%%
% findLayersToReplace(lgraph) finds the single classification layer and the
% preceding learnable (fully connected or convolutional) layer of the layer
% graph lgraph.
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end

% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);


% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end

end


