function [tsave, xsave, usave] = ks_solve(tfinal,N)

% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - nu*u_xxxx,  periodic BCs

%N = 2048;
if N ~= 2^nextpow2(N)
    error('N must be a power of 2');
end
x = 2*pi*(1:N)'/N;
%u = -sin(x)+2*cos(2*x)+3*cos(3*x)-4*sin(4*x);
%u = 0.01*(-sin(x)+2*cos(2*x)+3*cos(3*x)-4*sin(4*x));
u=randn(N,1);
v = fft(u);
%alpha = 87;
%nu = 4/alpha;
nu=0.005;

% % % % % %
%Spatial grid and initial condition:
h = 0.01; %10^(-4);
k = [0:N/2-1 0 -N/2+1:-1]';
L = k.^2 - nu*k.^4; %% added the 4* for my specif
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2));
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 , 2) );
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
tmax = 10; nmax = round(tmax/h); nplt = floor((tmax/1000)/h); g = -0.5i*k;
tt = zeros(1,nmax);
uu = zeros(N,nmax);

for n = 1:nmax
    t = n*h;
    Nv = g.*fft(real(ifft(v)).^2);
    a = E2.*v + Q.*Nv;
    Na = g.*fft(real(ifft(a)).^2); 
    b = E2.*v + Q.*Na;
    Nb = g.*fft(real(ifft(b)).^2);
    c = E2.*a + Q.*(2*Nb-Nv);
    Nc = g.*fft(real(ifft(c)).^2);
    v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;     
    if mod(n,nplt)==0
        n;
        u = real(ifft(v));
        uu(:,n) = u; 
        tt(n) = t;
    end
end


%%
cutoff = tt > 0;
cutoff = cutoff & tt<=tfinal;
%  cutoff = tt > 0; cutoff = cutoff & tt<10;

%contour(x/(2*pi),tt(cutoff),uu(:,cutoff).',[-10 -5 0 5 10]),shading interp, colormap(gray)
% contour(x,tt,uu.'),shading interp, colormap(gray)

%surfl(x,tt(cutoff),uu(:,cutoff).'),shading interp, colormap(gray), view(15,50)
%pcolor(x,tt(cutoff),uu(:,cutoff).'),shading interp, colormap(hot), %view(15,50)
tsave = tt(cutoff);
xsave = x/(2*pi);
dt = h;
dx = 1/N;
usave = uu(:,cutoff).';

%%
% uu2=uu(:,cutoff); [mm,nn]=size(uu2);
% for j=1:nn
%     uut(:,j)=abs(fftshift(fft(uu2(:,j))));
% end
% k=[0:N/2-1 -N/2:-1].';
% ks=fftshift(k);
% figure(4)
% surfl(ks,tt(cutoff),(uut(:,cutoff).')),shading interp, colormap(hot), %view(15,50)
% figure(5)
% waterfall(ks,tt(cutoff),(uut(:,cutoff).')), colormap([0 0 0])
% set(gca,'Xlim',[-50 50])
