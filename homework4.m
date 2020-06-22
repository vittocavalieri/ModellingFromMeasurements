clear variables;
close all;
clc;



%% Data

% Simulate Lorenz system
dt = 0.01;
T = 10;
t = 0:dt:T;
nt = length(t);
b = 8/3;
sig = 10;
r_vect = [10, 28, 40];

Lorenz = @(t,x,r)([ sig * (x(2) - x(1))       ; ...
                  x(1) * (r - x(3)) - x(2)  ; ...
                  x(1) * x(2) - b * x(3)    ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);


% Training Data
input=[];
output=[];
for i = 1:3
    r = r_vect(i);
    for j = 1:100  % training trajectories
        x0 = 30*(rand(3,1)-0.5);
        [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
        input = [input; [r*ones(nt-1,1), y(1:end-1,:)]];
        output = [output; y(2:end,:)];
        plot3(y(:,1),y(:,2),y(:,3)), hold on
        plot3(x0(1),x0(2),x0(3),'ro')
    end
end
grid on, view(-23,18)



%% NN

net = feedforwardnet([20 10 20]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'tansig';
net = train(net,input.',output.');

%load netLorenz.mat


r_vect = [10, 28, 40];
figure(2), set(gcf,'position',[200,10,600,600],'DefaultLineLineWidth',1.5);
for i = 1:3
    r = r_vect(i);
    x0 = 30*(rand(3,1)-0.5);
    [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
    figure(2);
    subplot(3,2,2*i-1); view(-75,15)
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
    grid on, xlabel('x(t)'), ylabel('y(t)'), zlabel('z(t)')
    title(['\rho = ',num2str(r_vect(i))]);
    set(gca,'Fontsize',12,'LineWidth',1);

    ynn(1,:) = x0;
    for jj = 2:length(t)
        y0 = net([r; x0]);
        ynn(jj,:) = y0.';
        x0 = y0;
    end
    plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])
    subplot(3,2,2*i);
    plot(t,y(:,1),t,ynn(:,1)), grid on, xlabel('t (s)'), ylabel('x(t)')
    title(['\rho = ',num2str(r_vect(i))]);
    set(gca,'Fontsize',12,'LineWidth',1);
    if i == 1
        legend('Lorenz','NN');
    end

    figure(2+i);
    set(gcf,'DefaultLineLineWidth',1.5);
    subplot(3,1,1), plot(t,y(:,1),t,ynn(:,1)), grid on, xlabel('t (s)'), ylabel('x(t)')
    title(['\rho = ',num2str(r_vect(i))]);
    legend('Lorenz','NN','Location','Best');
    set(gca,'Fontsize',12,'LineWidth',1);
    subplot(3,1,2), plot(t,y(:,2),t,ynn(:,2)), grid on, xlabel('t (s)'), ylabel('y(t)')
    set(gca,'Fontsize',12,'LineWidth',1);
    subplot(3,1,3), plot(t,y(:,3),t,ynn(:,3)), grid on, xlabel('t (s)'), ylabel('z(t)')
    set(gca,'Fontsize',12,'LineWidth',1);

end


r_vect = [17, 35];
figure(7), set(gcf,'position',[360 198 700 420],'DefaultLineLineWidth',1.5);
figure(8), set(gcf,'position',[360 198 560 420],'DefaultLineLineWidth',1.5);
for i = 1:2
    r = r_vect(i);
    x0 = 30*(rand(3,1)-0.5);
    [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
    figure(7);
    subplot(1,2,i); view(-75,15)
    plot3(y(:,1),y(:,2),y(:,3)), hold on
    plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
    grid on, xlabel('x(t)'), ylabel('y(t)'), zlabel('z(t)')
    title(['\rho = ',num2str(r_vect(i))]);
    set(gca,'Fontsize',12,'LineWidth',1);

    ynn(1,:) = x0;
    for jj = 2:length(t)
        y0 = net([r; x0]);
        ynn(jj,:) = y0.';
        x0 = y0;
    end
    plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

    figure(8);
    subplot(1,2,i), plot(t,y(:,1),t,ynn(:,1)), grid on, xlabel('t (s)'), ylabel('x(t)')
    title(['\rho = ',num2str(r_vect(i))]);
    if i == 2
        legend('Lorenz','NN','Location','Best');
    end
    set(gca,'Fontsize',12,'LineWidth',1);

end



%% Transition identification

r = 28;
x00 = [-8; 8; 27];

input = [];
output = [];
for j = 1:100  % training trajectories
    x0 = x00 + rand(3,1);
    [t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x0);
    x = y(:,1);
    label = setTransitionLabel(x);
    input = [input; y(1:end-1,:)];
    output = [output; [y(2:end,:), label(2:end)]];
end

% Training
net = feedforwardnet([20 10 20]);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'tansig';
net = train(net,input.',output.');

%load netLorenz_Transition.mat

% Forecast
x0 = x00;
ynn_tr(1,:) = [x0', -1];
for jj = 2:length(t)
    y0 = net(x0);
    ynn_tr(jj,:) = y0.';
    x0 = y0(1:3);
    yL(jj) = y0(4);
end

% Comparison
[t,y] = ode45(@(t,x) Lorenz(t,x,r),t,x00);
figure(); set(gcf,'position',[260 100 700 400],'DefaultLineLineWidth',1.5);
plot(t,y(:,1),t,ynn_tr(:,1));
hold on; grid on;
plot(t,yL,'rx')
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('t (s)');
ylabel('x(t)');
legend('Lorenz','NN','Transition Label','Location','Best');
title(['\rho = ',num2str(r)]);