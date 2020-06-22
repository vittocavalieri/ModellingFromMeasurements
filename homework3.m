clear variables;
close all;
clc;



%% Data

% Run reaction_diffusion.m to create this MAT file
%reaction_diffusion;
load reaction_diffusion_data.mat;

u_data = reshape(u,[],length(t));
v_data = reshape(v,[],length(t));
X = [u_data; v_data];

% % Plot simulated solution
% jj = [150 160 170 180 190 200];
% l = length(jj);
% figure(1);
% hold on;
% for i = 1:l
%     j = jj(i);
%     subplot(2,l,i);
%     pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
%     subplot(2,l,l+i);
%     pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
% end



%% Low-dimensional subspace projection

% SVD
[U,S,V] = svd(X,'econ');
sig = diag(S);
energy = sig/sum(sig);
figure(2);
hold on;
subplot(2,1,1);
plot(energy,'o');
xlim([0,70]);
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('k');
ylabel('Singular values energy');
subplot(2,1,2);
semilogy(energy,'o');
xlim([0,70]);
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('k');
ylabel('Singular values energy');

% Low rank approximation
r = 10;
U = U(:,1:r);
V = V(:,1:r);
S = S(1:r,1:r);

% Encoding
z = U'*[u_data; v_data];

% Decoding
data_tilde = U*z;
u_tilde = data_tilde(1:end/2,:);
v_tilde = data_tilde(end/2+1:end,:);

u_tilde_mat = reshape(u_tilde,size(u));
v_tilde_mat = reshape(v_tilde,size(v));

% jj = [150 160 170 180 190 200];
% l = length(jj);
% figure(3);
% hold on;
% figure(4);
% hold on;
% for i = 1:l
%     j = jj(i);
%     figure(3);
%     subplot(2,l,i);
%     pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
%     subplot(2,l,l+i);
%     pcolor(x,y,u_tilde_mat(:,:,j)); shading interp; colormap(hot); colorbar;
%     figure(4);
%     subplot(2,l,i);
%     pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
%     subplot(2,l,l+i);
%     pcolor(x,y,v_tilde_mat(:,:,j)); shading interp; colormap(hot); colorbar;
% end



%% Normalized Data for NN

for i = 1:size(z,1)
    d(i) = norm(z(i,:));
    z(i,:) = z(i,:)/d(i);
end

i_max = 180;
% input = z(:,1:end-1);
% output = z(:,2:end);
input = z(:,1:i_max-1);
output = z(:,2:i_max);



%% NN

numFeatures = 10;
numResponses = numFeatures;
numHiddenUnits = 20;

layers = [ ...
 sequenceInputLayer(numFeatures)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numHiddenUnits)
 fullyConnectedLayer(numResponses)
 regressionLayer];
 

options = trainingOptions('adam', ...
 'MaxEpochs',1000, ...
 'GradientThreshold',1, ...
 'InitialLearnRate',0.005, ...
 'LearnRateSchedule','piecewise', ...
 'LearnRateDropPeriod',125, ...
 'LearnRateDropFactor',0.2, ...
 'Verbose',0, ...
 'Plots','training-progress');

net = trainNetwork(input,output,layers,options);

%save('netRD.mat','net');
%load netRD.mat

% Forecast
x0 = z(:,1);
ynn(1,:) = x0;
for jj = 2:length(t)
    y0 = predict(net,x0);
    ynn(jj,:) = y0.';
    x0 = y0;
end



%% Comparison Plots

% Verify forecast (subspace)
figure();
hold on;
subplot(2,1,1);
plot(z');
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('time');
ylabel('z');
title('Low-rank variables');
subplot(2,1,2);
plot(ynn);
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('time');
ylabel('y_{nn}');

% Decoding
for i = 1:size(ynn,2)
    ynn(:,i) = ynn(:,i) * d(i);
end
data_nn = U*ynn';
u_nn = data_nn(1:end/2,:);
v_nn = data_nn(end/2+1:end,:);

u_nn_mat = reshape(u_nn,size(u));
v_nn_mat = reshape(v_nn,size(v));

rmse = sqrt(mean(([u_nn; v_nn]-[u_data; v_data]).^2,'all'));


% Verify forecast (full space)
jj = [120 160 180 200];
l = length(jj);
figure(5);
set(gcf,'position',[100 100 1000 500],'DefaultLineLineWidth',1.5);
hold on;
figure(6);
set(gcf,'position',[100 100 1000 500],'DefaultLineLineWidth',1.5);
hold on;
for i = 1:l
    j = jj(i);
    figure(5);
    subplot(2,l,i);
    pcolor(x,y,u(:,:,j)); shading interp; colormap(hot); colorbar;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('u - RD','Fontsize',16);
    end
    title(['time ',num2str(j)]);
    subplot(2,l,l+i);
    pcolor(x,y,u_nn_mat(:,:,j)); shading interp; colormap(hot); colorbar; drawnow;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('u - NN','Fontsize',16);
    end
    figure(6);
    subplot(2,l,i);
    pcolor(x,y,v(:,:,j)); shading interp; colormap(hot); colorbar;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('v - RD','Fontsize',16);
    end
    title(['time ',num2str(j)]);
    subplot(2,l,l+i);
    pcolor(x,y,v_nn_mat(:,:,j)); shading interp; colormap(hot); colorbar; drawnow;
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        ylabel('v - NN','Fontsize',16);
    end
end
