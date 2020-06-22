clear variables;
close all;
clc;



%% Data

load BZ.mat
[m,n,k] = size(BZ_tensor); % x vs y vs time data

% for j = 1:10:k
%     A = BZ_tensor(:,:,j);
%     pcolor(A), shading interp, pause(0.2)
% end

X = reshape(BZ_tensor,[],k);



%% Low-dimensional subspace projection

[U,S,V] = svd(X,'econ');

sig = diag(S);
energy = sig/sum(sig);
figure(2);
hold on;
subplot(2,1,1);
plot(energy,'o');
xlim([0,400]);
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('k');
ylabel('Singular values energy');
subplot(2,1,2);
semilogy(energy,'o');
xlim([0,400]);
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('k');
ylabel('Singular values energy');


% Low rank approximation
r = 100;
U = U(:,1:r);
V = V(:,1:r);
S = S(1:r,1:r);

U_tensor = reshape(U,m,n,[]);
% figure();
% for j=1:r
%     A = U_tensor(:,:,j);
%     pcolor(A), shading interp, pause(0.2)
% end

% Encoding
z = U'*X;

% Decoding
x_tilde = U*z;

x_tilde_mat = reshape(x_tilde,m,n,k);

% figure();
% hold on;
% for j=1:10:k
%     subplot(1,2,1)
%     A = BZ_tensor(:,:,j);
%     pcolor(A), shading interp
%     subplot(1,2,2)
%     A = x_tilde_mat(:,:,j);
%     pcolor(A), shading interp; drawnow; pause(0.2)
% end



%% Normalized Data for NN

for i = 1:size(z,1)
    d(i) = norm(z(i,:));
    z(i,:) = z(i,:)/d(i);
end

i_max = 1000;
% input = z(:,1:end-1);
% output = z(:,2:end);
input = z(:,1:i_max-1);
output = z(:,2:i_max);


%% NN

numFeatures = r;
numResponses = r;
numHiddenUnits = 5*r;

layers = [ ...
 sequenceInputLayer(numFeatures)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numResponses)
 regressionLayer];
 

options = trainingOptions('adam', ...
 'MaxEpochs',500, ...
 'GradientThreshold',1, ...
 'InitialLearnRate',0.005, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',125, ...
 'LearnRateDropFactor',0.2, ...
 'Verbose',0, ...
 'Plots','training-progress');

net = trainNetwork(input,output,layers,options);

%save('netBZ.mat','net');
%load netBZ.mat

% Forecast
x0 = z(:,1);
ynn(1,:) = x0;
for jj = 2:k
    y0 = predict(net,x0);
    ynn(jj,:) = y0.';
    x0 = y0;
end



%% Comparison Plots

% Verify forecast (subspace)
figure();
hold on;
subplot(2,1,1);
plot(z(1:10,:)');
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('time');
ylabel('z');
title('First 10 low-rank variables');
subplot(2,1,2);
plot(ynn(:,1:10));
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('time');
ylabel('y_{nn}');

% Decoding
for i = 1:size(ynn,2)
    ynn(:,i) = ynn(:,i) * d(i);
end
BZ_nn = U*ynn';
BZ_nn_tensor = reshape(BZ_nn,m,n,k);


% Verify forecast (full space)
jj = [200 600 1000 1200];
l = length(jj);
figure();
set(gcf,'position',[100 100 1000 500],'DefaultLineLineWidth',1.5);
hold on;
for i = 1:l
    j = jj(i);
    subplot(2,l,i);
    pcolor(BZ_tensor(:,:,j)); shading interp; colorbar;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('BZ','Fontsize',16);
    end
    title(['time ',num2str(j)]);
    subplot(2,l,l+i);
    pcolor(BZ_nn_tensor(:,:,j)); shading interp; colorbar;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('NN','Fontsize',16);
    end
end