clear;
clc;


%% Matlab Settings
rng(2024);  % For reproducibility


%% Parameters
n = 4;      % Number of states
m = 1;      % Number of inputs
p = 1;      % Number of outputs
l = n;      % SISO case

% LTI system matrices definition
A = [ 0.5780,  0.8492,  0.4220,  0.1508;
     -0.6985,  0.5780,  0.6985,  0.4220;
      0.4220,  0.1508,  0.5780,  0.8492;
      0.6985,  0.4220, -0.6985,  0.5780];

B = [ 0.4610;
      0.8492;
      0.0390;
      0.1508];

C = [0, 0, 1, 0];

% System dynamics and output function
f = @(x, u, d) A*x + B*u + d;
h = @(x) C*x;

% System initial state
x0_mag = 0.1;
x0 = x0_mag * 2 .* (rand([n, 1]) - 0.5);


%% Minimality tests
if(rank(ctrb(A,B)) < n)
    error("system not controllable");
end
if(rank(obsv(A,C)) < n)
    error("system not observable");
end


%% Sampling Simulation
T = 100;    % Number of samples

% Generate random input sequence
U_mag = 1.0;
U = U_mag .* 2 .* (rand([m, T+1]) - 0.5);

% Generate random noise sequence
gamma = 0.1;
D = 2 .* (rand([n, T+1]) - 0.5);
D = gamma * D ./ max(vecnorm(D), [], 2);

% Compute noise bounds
% delta = max(eig(D(:,1:T)*D(:,1:T).')) .* eye(n);
delta = gamma^2 .* T .* eye(n);     % Point-wise worst-case bounds

% Simulate the system with and without noise
X = zeros([n, T+1]);
Y = zeros([p, T+1]);

Xd = zeros([n, T+1]);
Yd = zeros([p, T+1]);

X(:,1) = x0;
Xd(:,1) = x0;
for k = 1:T
    Y(:,k) = h(X(:,k));
    X(:,k+1) = f(X(:,k), U(:,k), zeros([n,1]));
    Yd(:,k) = h(Xd(:,k));
    Xd(:,k+1) = f(Xd(:,k), U(:,k), D(:,k));
end
Y(:,end) = h(X(:,end));
Yd(:,end) = h(Xd(:,end));

% Data vectors construction
U0 = U(:,1:end-1);
U1 = U(:,2:end-0);

X0 = X(:,1:end-1);
X1 = X(:,2:end-0);

Y0 = Y(:,1:end-1);
Y1 = Y(:,2:end-0);

Xd0 = Xd(:,1:end-1);
Xd1 = Xd(:,2:end-0);

Yd0 = Yd(:,1:end-1);
Yd1 = Yd(:,2:end-0);

Phi0 = [hankelmat(Y0, l); hankelmat(U0, l)];
Phi1 = [hankelmat(Y1, l); hankelmat(U1, l)];


%% Problem 0
% Just to test the cvx toolbox
cvx_begin sdp
    variable Y_0(T,n)
    variable P_0(n,n) symmetric
    X0*Y_0 == P_0;              %#ok
    [P_0-eye(n), X1*Y_0; 
       Y_0'*X1',    P_0] >= 0;  %#ok
cvx_end

% Closed-loop gain computation
K_0 = U0 * Y_0 / P_0;


%% Problem 1
% State-feedback controller (P1)
cvx_begin sdp
    variable P_1(n,n) symmetric
    variable Y_1(T,n)
    variable a
    a >= 0                                          %#ok
    Xd0*Y_1 == P_1;                                 %#ok
    [P_1-eye(n),     Y_1'*Xd1',         Y_1';
        Xd1*Y_1, P_1 - a*delta, zeros([n,T]);
            Y_1,  zeros([T,n]),     a*eye(T)] >= 0; %#ok
cvx_end

% Closed-loop gain computation
K_1 = U0 * Y_1 / P_1;


%% Problem 3
% Output-feedback control (P3)
cvx_begin sdp
    variable P_3((p+m)*l, (p+m)*l) symmetric
    variable Y_3(T-(l-1), (p+m)*l)
    Phi0*Y_3 == P_3;                    %#ok
    [P_3-eye((p+m)*l), Phi1*Y_3; 
           Y_3'*Phi1',      P_3]  >= 0; %#ok
cvx_end

% Closed-loop gain computation
K_3 = U(:, l+1:end) * Y_3 / P_3;


%% Closed-Loop Simulation Data
T_cl = 100;     % Number of steps simulated

% Initial conditions and noise generation
x0_cl_mag = 1.0;
x0_cl = x0_cl_mag * 2 .* (rand([n, 1]) - 0.5);

z0_cl_mag = 1.0;
z0_cl = z0_cl_mag * 2 .* (rand([(p+m)*l, 1]) - 0.5);

D_cl = 2 .* (rand([n, T_cl+1]) - 0.5);
D_cl = gamma * D_cl ./ max(vecnorm(D_cl), [], 2);


%% Closed-Loop 0
% Control law
u_cl_0 = @(x) K_0 * x;

% Closed-loop simulation
X_cl_0 = zeros([n, T_cl+1]);

X_cl_0(:,1) = x0_cl;
for k = 1:T_cl
    X_cl_0(:,k+1) = f(X_cl_0(:,k), u_cl_0(X_cl_0(:,k)), zeros([n,1]));
end

% Closed-loop eigen-values check
A_cl_0 = A + B*K_0;
eig_cl_0 = eig(A + B*K_0);


%% Closed-Loop 1
% Control law
u_cl_1 = @(x) K_1 * x;

% Closed-loop simulation
X_cl_1 = zeros([n, T_cl+1]);

X_cl_1(:,1) = x0_cl;
for k = 1:T_cl
    X_cl_1(:,k+1) = f(X_cl_1(:,k), u_cl_1(X_cl_1(:,k)), D_cl(:,k));
end

% Closed-loop eigen-values check
A_cl_1 = A + B*K_1;
eig_cl_1 = eig(A_cl_1);


%% Closed-Loop 3
% Control law
u_cl_3 = @(z) K_3 * z;

% Build matrices for controller state evolution
F = [kron(diag(ones([1,l-1]),1), eye(p)),        kron(zeros(l), zeros([p,m]));
            kron(zeros(l), zeros([m,p])), kron(diag(ones([1,l-1]),1), eye(m))];

L = [ kron((1:l).' == l, ones(p));
     kron(zeros([l,1]), zeros(p))];

G = [kron(zeros([l,1]), zeros(m));
      kron((1:l).' == l, ones(m))];

% Controller dynamic
g = @(z, y, u) F*z + L*y + G*u;

% Closed-loop simulation
X_cl_3 = zeros([n, T_cl+1]);
Z_cl_3 = zeros([(p+m)*l, T_cl+1]);

X_cl_3(:,1) = x0_cl;
Z_cl_3(:,1) = z0_cl;
for k = 1:T_cl
    X_cl_3(:,k+1) = f(X_cl_3(:,k), u_cl_3(Z_cl_3(:,k)), zeros([n,1]));
    Z_cl_3(:,k+1) = g(Z_cl_3(:,k), h(X_cl_3(:,k)), u_cl_3(Z_cl_3(:,k)));
end

% Closed-loop eigen-values check
A_cl_3 = [  A,   B*K_3; 
          L*C, F+G*K_3];
eig_cl_3 = eig(A_cl_3);


%% Plotting
% Visualize closed-loop simulations

% Set plot options
set(groot, "defaulttextinterpreter", "latex");
set(groot, "defaultAxesTickLabelInterpreter", "latex");
set(groot, "defaultLegendInterpreter", "latex");

fig = 0;
figs = cell([1, 3]);

% Problem 0
fig = fig + 1;
figs{fig} = figure(fig);
plot((0:T_cl).', X_cl_0');
grid on;
title("\bf{Problem 0}");
legend("$" + "x_{" + string(1:n) + "}" + "$");
xlabel("Step");
ylabel("Amplitude");

% Problem 1
fig = fig + 1;
figs{fig} = figure(fig);
plot((0:T_cl).', X_cl_1');
grid on;
title("\bf{Problem 1}");
legend("$" + "x_{" + string(1:n) + "}" + "$");
xlabel("Step");
ylabel("Amplitude");

% Problem 3
fig = fig + 1;
figs{fig} = figure(fig);
plot((0:T_cl).', X_cl_3');
grid on;
title("\bf{Problem 3}");
legend("$" + "x_{" + string(1:n) + "}" + "$");
xlabel("Step");
ylabel("Amplitude");

% Reset plot options
set(groot, "defaulttextinterpreter", "none");
set(groot, "defaultAxesTickLabelInterpreter", "none");
set(groot, "defaultLegendInterpreter", "none");


%% Export Figures
PlotSize = [520, 210];
LineWidth = 1.0;
FontSize = 14;
FileDir = "Figures/";
FileName = ["Assignment_1_0", "Assignment_1_1", "Assignment_1_3"];

for k = 1:length(figs)
    figs{k}.Units = "points";
    figs{k}.Position(3:4) = PlotSize;
    set(findobj(figs{k}, 'Type', 'Line'), 'LineWidth', LineWidth);
    set(findobj(figs{k}, 'Type', 'Axes'), 'LineWidth', LineWidth);
    set(findobj(figs{k}, 'Type', 'Axes'), 'GridLineWidth', LineWidth);
    set(findobj(figs{k}, 'Type', 'Axes'), 'MinorGridLineWidth', LineWidth);
    fontsize(figs{k}, FontSize, "points");
    saveas(figs{k}, FileDir + FileName(k), "epsc");
end