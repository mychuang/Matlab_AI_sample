rng(0,'twister');
% create data: feature reflectivety
dbz_min = 45;
dbz_max = 60;
dbz_bright = (dbz_max-dbz_min).*rand(1, 500) + dbz_min;

dbz_min = 10;
dbz_max = 30;
dbz_ice = (dbz_max-dbz_min).*rand(1, 500) + dbz_min;

dbz_min = 10;
dbz_max = 50;
dbz_warm = (dbz_max-dbz_min).*rand(1, 500) + dbz_min;

raw_dbz = [dbz_bright, dbz_ice, dbz_warm];
raw_dbz = reshape(raw_dbz, [1500,1]);

% create data: feature temperature
temp_min = -5;
temp_max = 5;
temp_bright = (temp_max-temp_min).*rand(1, 500) + temp_min;

temp_min = -20;
temp_max = 2;
temp_ice = (temp_max-temp_min).*rand(1, 500) + temp_min;

temp_min = -2;
temp_max = 20;
temp_warm = (temp_max-temp_min).*rand(1, 500) + temp_min;

raw_temp = [temp_bright, temp_ice, temp_warm];
raw_temp = reshape(raw_temp, [1500,1]);

% merge features
x_train = [raw_temp, raw_dbz];

%%
% create data: labels
y1 = zeros(1, 500);       %bright band
y2 = ones(1, 500);        %ice
y3 = ones(1, 500) + 1;    %warm
y_train = [y1, y2, y3];
y_train = reshape(y_train, [1500, 1]);

%categorical
y_train2 = categorical(y_train);

%%
% plot method
gscatter(x_train(:,1), x_train(:,2), y_train2)

%%
%KNN model
knnmodel = fitcknn(x_train, y_train2, "NumNeighbors", 5);

%evaluate
predicted = predict(knnmodel, [-3, 50] );
fprintf('%c\n', predicted);

%% Create the test data
dbz_min = 45;
dbz_max = 60;
dbz_bright = (dbz_max-dbz_min).*rand(1, 3) + dbz_min;

dbz_min = 10;
dbz_max = 30;
dbz_ice = (dbz_max-dbz_min).*rand(1, 3) + dbz_min;

dbz_min = 10;
dbz_max = 50;
dbz_warm = (dbz_max-dbz_min).*rand(1, 3) + dbz_min;

raw_dbz = [dbz_bright, dbz_ice, dbz_warm];
raw_dbz = reshape(raw_dbz, [9,1]);

temp_min = -5;
temp_max = 5;
temp_bright = (temp_max-temp_min).*rand(1, 3) + temp_min;

temp_min = -20;
temp_max = 2;
temp_ice = (temp_max-temp_min).*rand(1, 3) + temp_min;

temp_min = -2;
temp_max = 20;
temp_warm = (temp_max-temp_min).*rand(1, 3) + temp_min;

raw_temp = [temp_bright, temp_ice, temp_warm];
raw_temp = reshape(raw_temp, [9,1]);

x_test = [raw_temp, raw_dbz];

y1 = zeros(1, 3);       %bright band
y2 = ones(1, 3);        %ice
y3 = ones(1, 3) + 1;    %warm
y_test = [y1, y2, y3];
y_test = reshape(y_test, [9, 1]);

%categorical
y_test2 = categorical(y_test);

% evaluate
predictions = predict(knnmodel, x_test );

iscorrect = predictions == y_test2;
accuracy = sum(iscorrect)/numel(predictions);
fprintf('accuracy %f\n', accuracy);

iswrong = predictions ~= y_test2;
misclassrate = sum(iswrong)/numel(predictions);
fprintf('missing %f\n', misclassrate);

confusionchart(y_test2, predictions);


