clear variables;
close all;
clc;



%% Population data

hare = [20 20 52 83 64 68 83 12 36 150 110 60 7 10 70 100 92 70 10 11 137 137 18 22 52 83 18 10 9 65];
lynx = [32 50 12 10 13 36 15 12 6 6 65 70 40 9 20 34 45 40 15 15 60 80 26 18 37 50 35 12 12 25];

t_0 = 1845; t_end = 1903; dt = 2;
time = t_0:dt:t_end;
n = length(time);



%% DMD

X = [hare(1:end-1); lynx(1:end-1)];
X1 = [hare(2:end); lynx(2:end)];

[U,S,V] = svd(X,'econ');

A_tilde = U'*X1*V/S;
[W,D] = eig(A_tilde);
Phi = X1*V/S*W;

lambda = diag(D);
omega = log(lambda)/dt;

x0 = X(:,1);
y0 = Phi\x0;

t = time-t_0;
u_modes = zeros(2,n);
for i = 1:n
    u_modes(:,i) = y0.*exp(omega*t(i));
end
u_dmd = Phi*u_modes;


figure();
set(gcf,'position',[200,100,1000,400],'DefaultLineLineWidth',1.5);
subplot(1,2,1);
hold on; grid on;
plot(time,hare,'k');
plot(time,lynx,'r');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[0 160],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
title('Data');
legend('hare','lynx','Location','NorthWest');
subplot(1,2,2);
hold on; grid on;
plot(time,u_dmd(1,:),'k');
plot(time,u_dmd(2,:),'r');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[0 160],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
title('DMD model');
legend('hare','lynx');



%% Time-delay DMD 

td_emb = 8;
q = td_emb + 1;
p = n-td_emb;
H = zeros(2*q,p);
for i = 1:q
    H(2*i-1:2*i,:) = [hare(i:i+p-1); lynx(i:i+p-1)];
end

H1 = H(:,1:end-1);
H2 = H(:,2:end);

[Uh,Sh,Vh] = svd(H1);

figure();
set(gcf,'position',[360,198,560,250],'DefaultLineLineWidth',1.5);
sig = diag(Sh);
energy = sig/sum(sig);
plot(energy,'ko','MarkerSize',8,'LineWidth',1.5);
grid on;
set(gca,'Fontsize',12,'LineWidth',1);
xlabel('k');
ylabel('Singular values energy');

r = 16;
u = Uh(:,1:r);
s = Sh(1:r,1:r);
v = Vh(:,1:r);

A_tilde = u'*H2*v/s;
[W,D] = eig(A_tilde);
Phi = H2*v/s*W;

lambda = diag(D);
omega = log(lambda)/dt;

y0 = Phi\H1(:,1);

u_modes = zeros(r,n);
for i = 1:n
    u_modes(:,i) = y0.*exp(omega*t(i));
end
u_dmd_td = Phi*u_modes;

figure();
set(gcf,'position',[200,100,1000,400],'DefaultLineLineWidth',1.5);
subplot(1,2,1);
hold on; grid on;
plot(time,hare,'k');
plot(time,lynx,'r');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[0 160],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
title('Data');
legend('hare','lynx','Location','NorthWest');
subplot(1,2,2);
hold on; grid on;
plot(time,abs(u_dmd_td(1,:)),'k');
plot(time,abs(u_dmd_td(2,:)),'r');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[0 160],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
title('Time-delay DMD model');
legend('hare','lynx');



%% Lotka-Volterra model

% Regression

x = hare';
y = lynx';
xdot = zeros(n-2,1);
ydot = zeros(n-2,1);
% center difference scheme
for j=2:n-1
  xdot(j-1) = (x(j+1)-x(j-1))/(2*dt);
  ydot(j-1) = (y(j+1)-y(j-1))/(2*dt);
end

xs = x(2:n-1);
ys = y(2:n-1);

A1 = [xs -xs.*ys];
A2 = [xs.*ys -ys];

figure();
hold on;
xi1 = pinv(A1)*xdot;
xi2 = pinv(A2)*ydot;
subplot(1,2,1), bar(xi1)
set(gca,'Fontsize',12,'Ylim',[0 0.25],'LineWidth',1);
title('\xi_1');
subplot(1,2,2), bar(xi2)
set(gca,'Fontsize',12,'Ylim',[0 0.25],'LineWidth',1);
title('\xi_2');
% xi1 = A1\xdot;
% xi2 = A2\ydot;
% subplot(3,2,3), bar(xi1)
% subplot(3,2,4), bar(xi2)
% xi1 = robustfit(A1,xdot);
% xi2 = robustfit(A2,ydot);
% subplot(3,2,5), bar(xi1)
% subplot(3,2,6), bar(xi2)

b = xi1(1);
p = xi1(2);
r = xi2(1);
d = xi2(2);

tt = linspace(0,t(end),1000);
[tt,z] = ode45('rhs_LV',tt,x0,[],b,p,r,d);
figure();
set(gcf,'position',[200,100,500,350],'DefaultLineLineWidth',1.5);
hold on; grid on;
plot(time,hare,'k');
plot(time,lynx,'r');
plot(t_0+tt,z(:,1),'ko');
plot(t_0+tt,z(:,2),'ro');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[0 250],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
legend('hare (data)','lynx (data)','hare (LV)','lynx (LV)','Location','NorthEast');



%% SINDy

A = [xs ys xs.*ys xs.^2 ys.^2 xs.^2.*ys xs.*ys.^2 xs.^3 ys.^3 cos(xs) cos(ys) sin(xs) sin(ys)];

theta = {'x_1'; 'x_2'; 'x_1*x_2'; 'x_1^2'; 'x_2^2'; 'x_1^2*x_2'; 'x_1*x_2^2'; 'x_1^3'; 'x_2^3'; 'cos(x_1)'; 'cos(x_2)'; 'sin(x_1)'; 'sin(x_2)'};

figure();
set(gcf,'position',[150,50,1000,500],'DefaultLineLineWidth',1.5);
hold on;
xi1 = pinv(A)*xdot;
xi2 = pinv(A)*ydot;
subplot(3,2,1), bar(xi1)
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);
ylabel('loadings');
title('\xi_1 (hare)');
subplot(3,2,2), bar(xi2), text(4,-2,'least-square','FontSize',12);
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);
title('\xi_2 (lynx)');
xi1 = lasso(A,xdot,'Lambda',0.1);
xi2 = lasso(A,ydot,'Lambda',0.1);
subplot(3,2,3), bar(xi1)
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);
ylabel('loadings');
subplot(3,2,4), bar(xi2), text(4,-2,'lasso (\lambda=0.1)','FontSize',12);
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);
xi1 = lasso(A,xdot,'Lambda',0.01);
xi2 = lasso(A,ydot,'Lambda',0.01);
subplot(3,2,5), bar(xi1)
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);
ylabel('loadings');
subplot(3,2,6), bar(xi2), text(4,-2,'lasso (\lambda=0.01)','FontSize',12);
set(gca,'Fontsize',10,'Ylim',[-5 2],'LineWidth',1,'xticklabel',theta,'XTickLabelRotation',60);


% Selected solution
xi1 = lasso(A,xdot,'Lambda',0.1);
xi2 = lasso(A,ydot,'Lambda',0.1);


[tt,z] = ode45('rhs_syndy',tt,x0,[],xi1,xi2);
figure();
set(gcf,'position',[200,100,500,350],'DefaultLineLineWidth',1.5);
hold on; grid on;
plot(time,hare,'k');
plot(time,lynx,'r');
plot(t_0+tt,z(:,1),'k-.');
plot(t_0+tt,z(:,2),'r-.');
set(gca,'Fontsize',12,'Xlim',[t_0 t_end],'Ylim',[-10 160],'LineWidth',1);
xlabel('time (years)');
ylabel('population');
legend('hare (data)','lynx (data)','hare (SINDy)','lynx (SINDy)','Location','NorthEast');
