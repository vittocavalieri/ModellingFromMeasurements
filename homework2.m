clear variables;
close all;
clc;



trainNN = false;     % true (train), false (load)  

%% Data for training

if trainNN

    % Simulate KS equation
    N = 2^8;
    %dt = 0.01;
    tfinal = 2;
    
    n_train = 50;
    dataTrain = [];
    for j = 1:n_train  % training trajectories
        [tsave, xsave, usave] = ks_solve(tfinal,N);
        dataTrain = [dataTrain; usave];
    end
    
    mu = mean(dataTrain);
    sig = std(dataTrain);
    
    dataTrainStandardized = (dataTrain - mu) ./ sig;
    
    XTrain = dataTrainStandardized(1:end-1,:);
    YTrain = dataTrainStandardized(2:end,:);

end



%% NN

mmry = false;        % true (lstm), false (otherwise)

% Training

if trainNN
    numFeatures = N;
    numResponses = N;
    if mmry
        layers = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(2*N)
            tanhLayer
            fullyConnectedLayer(4*N)
            tanhLayer
            dropoutLayer
            fullyConnectedLayer(2*N)
            reluLayer
            fullyConnectedLayer(numResponses)
            regressionLayer];
    else
        layers = [ ...
            sequenceInputLayer(numFeatures)
            tanhLayer
            fullyConnectedLayer(4*N)
            tanhLayer
            fullyConnectedLayer(4*N)
            reluLayer
            fullyConnectedLayer(2*N)
            tanhLayer
            dropoutLayer
            fullyConnectedLayer(4*N)
            tanhLayer
            fullyConnectedLayer(2*N)
            reluLayer
            fullyConnectedLayer(4*N)
            tanhLayer
            fullyConnectedLayer(numResponses)
            regressionLayer];
    end
    options = trainingOptions('adam', ...
        'MaxEpochs',1000, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.001, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');
    net = trainNetwork(XTrain',YTrain',layers,options);
    save('netKS_v2.mat','net','tfinal','N','mu','sig');
else
    load('netKS.mat');
end


% Test

[tsave, xsave, usave] = ks_solve(tfinal,N);
dataTest = usave;
dataTestStandardized = (dataTest - mu) ./ sig;
x0 = dataTestStandardized(1,:);

if mmry
    [net,YPred] = predictAndUpdateState(net,x0');
    YPred = YPred';
    for i = 2:length(tsave)
        [net,YPred(i,:)] = predictAndUpdateState(net,YPred(i-1,:)','ExecutionEnvironment','cpu');
    end
else
    YPred = predict(net,x0');
    YPred = YPred';
    for i = 2:length(tsave)
        YPred(i,:) = predict(net,YPred(i-1,:)','ExecutionEnvironment','cpu');
    end
end

YPred = sig.*YPred + mu;

figure();
hold on;
subplot(1,2,1);
pcolor(xsave,tsave,usave), shading interp, colormap(hot);  colorbar;
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('x');
ylabel('t (s)');
title('KS');
subplot(1,2,2);
pcolor(xsave,tsave,YPred), shading interp, colormap(hot);  colorbar;
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('x');
ylabel('t (s)');
title('NN');
