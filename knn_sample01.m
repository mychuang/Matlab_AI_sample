% import data
x1 = rand(1, 500);
x2 = rand(1, 500) + 1;
x_train = [x1, x2];
x_train = reshape(x_train,[1000,1]);

% plot method
%plot(x_train, 'ro','MarkerFaceColor','r')

y1 = zeros(1, 500);
y2 = ones(1, 500);
y_train = [y1, y2];
y_train = reshape(y_train, [1000, 1]);

%KNN model
knnmodel = fitcknn(x_train, y_train, "NumNeighbors", 5);

%evaluate
predicted = predict(knnmodel, 1.5 );
fprintf('%i\n', predicted);

%% Create the test data
x1 = rand(1, 3);
x2 = rand(1, 3) + 1;
x_test = [x1, x2];
x_test = reshape(x_test,[6,1]);

y1 = zeros(1, 3);
y2 = ones(1, 3);
y_test = [y1, y2];
y_test = reshape(y_test, [6, 1]);

% evaluate
predictions = predict(knnmodel, x_test );

iscorrect = predictions == y_test;
accuracy = sum(iscorrect)/numel(predictions);
fprintf('accuracy %f\n', accuracy);

iswrong = predictions ~= y_test;
misclassrate = sum(iswrong)/numel(predictions);
fprintf('missing %f\n', misclassrate);

confusionchart(y_test, predictions);

