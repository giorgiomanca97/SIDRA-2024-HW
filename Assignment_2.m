clear;
clc;


%% Matlab Settings
rng(2024);    % For reproducibility


%% Parameters
n = 4;      % Number of states
m = 1;      % Number of inputs
p = 1;      % Number of outputs
q = 1;      % Number of non-linearities
s = n+q;    % Number of Z(x) entries

% Non-linear system definition
F_1 = 0.10;
J_1 = 0.15;
F_2 = 0.15;
J_2 = 0.20;
K_c = 0.40;
N_c = 2.00;
M = 0.4;
G = 9.8;
L = 0.1;

Q = @(x) [cos(x(1))];
Z = @(x) [x; Q(x)];

A = [           0.0,      1.0,             0.0,      0.0,          0.0;
           -K_c/J_2, -F_2/J_2,   K_c/(J_2*N_c),      0.0, -(M*G*L)/J_2;
                0.0,      0.0,             0.0,      1.0,          0.0;
     -K_c/(J_1*N_c),      0.0, K_c/(J_1*N_c^2), -F_1/J_1,          0.0];

B = [ 0.0;
      0.0;
      0.0;
     1/J_1];

% System dynamics and output function
f = @(x, u) [...
    x(2); ...
    (-K_c*x(1) -F_2*x(2) +K_c/N_c*x(3) -M*G*L*cos(x(1)))/J_2; ...
    x(4); ...
    (-K_c/N_c*x(1) +K_c/N_c^2*x(3) -F_1*x(4) + u)/J_1 ...
];
    
h = @(x) x(1);

% Augmented system dynamics
g = @(x, u, d, r) [...
    f(x, u) + d;
    r - h(x) ...
];

% System initial state (with augmented state)
x0_mag = 0.1;
x0 = x0_mag .* 2 .* (rand([n+p, 1]) - 0.5);

% Symbolic variables
x = sym('x_', [n,1]);
xi = sym('xi_', [p,1]);
xxi = [x; xi];


%% Sampling
T = 10;             % Number of samples
ts = 0.1;           % Sampling-time
tf = T*ts;          % Final time
times = 0:ts:tf;    % Sample times vector

% Generate random input sequence
U_mag = 0.1;
U = U_mag .* 2 .* (rand([m, T+1]) - 0.5);
u = @(t) interp1(times, U.', t, "previous", "extrap").';

% Noise and reference definition
D = [0.1; 0.2; 0.3; 0.4];
d = sym(D);
R = pi/3;
r = sym(R);

% Simulation with and without state augmentation and noise
odeopt = odeset("RelTol", 1e-6, "AbsTol", 1e-6);
sol_1 = ode45(@(t,x) f(x, u(t)), [0 tf], x0(1:n));
sol_3 = ode45(@(t,x) g(x, u(t), D, R), [0 tf], x0);

% Data vectors construction
U0 = zeros([m, T]);

X0_1 = zeros([n, T]);
X1_1 = zeros([n, T]);

X0_3 = zeros([n+p, T]);
X1_3 = zeros([n+p, T]);

for k = 1:T
    U0(:,k) = U(:,k);

    X0_1(:,k) = deval(sol_1, times(k));
    X1_1(:,k) = f(X0_1(:,k), u(times(k)));

    X0_3(:,k) = deval(sol_3, times(k));
    X1_3(:,k) = g(X0_3(:,k), u(times(k)), D, R);
end


%% Example 1a
% Ex1 with 1st choice of Q(x)
q_1a = 1;       % Number of non-linearities
s_1a = n+q_1a;  % Number of Z(x) entries
r_1a = n;       % R matrix size

% Non-linearities definition and gradient check
Q_1a = @(x) [cos(x(1))];
DQ_1a = jacobian(Q_1a(x), x);
DDQ_1a = DQ_1a.' * DQ_1a;

% Problem matrices construction
RQ_1a = [1, 0, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0];

M_1a = ones(1,T);

% Z function and vectors definition
Z_1a = @(x) [x; Q_1a(x)];
Z0_1a = zeros([s_1a, T]);
for k = 1:T 
    Z0_1a(:,k) = Z_1a(X0_1(:,k));
end

% CVX problem
cvx_begin sdp
    variable P_1a(n,n) symmetric
    variable Y1_1a(T,n)
    variable G2_1a(T,s_1a-n)
    variable a_1a
    a_1a >= 0;                                                                              %#ok
    P_1a >= zeros(n);                                                                       %#ok
    Z0_1a*Y1_1a == [P_1a; zeros(s_1a-n,n)];                                                 %#ok
    [X1_1*Y1_1a + Y1_1a'*X1_1' + a_1a*eye(n),         X1_1*G2_1a,         P_1a*RQ_1a;
                                G2_1a'*X1_1',       -eye(s_1a-n), zeros(s_1a-n,r_1a);
                                RQ_1a'*P_1a', zeros(r_1a,s_1a-n),         -eye(r_1a)] <= 0; %#ok
    Z0_1a*G2_1a == [zeros(n,s_1a-n); eye(s_1a-n)];                                          %#ok
    M_1a * [Y1_1a, G2_1a] == zeros(1,s_1a);                                                 %#ok
cvx_end

% Closed-loop gain computation
G1_1a = Y1_1a/P_1a;
G_1a  = [G1_1a, G2_1a];
K_1a  = U0*G_1a;
CL_1a = X1_1*G_1a;


%% Example 1b
% Ex1 with 2nd choice of Q(x)
q_1b = 3;       % Number of non-linearities
s_1b = n+q_1b;  % Number of Z(x) entries
r_1b = n;       % R matrix size

% Non-linearities definition and gradient check
Q_1b = @(x) [cos(x(1)); x(1)^2; sin(x(2))];
DQ_1b = jacobian(Q_1b(x), x);
DDQ_1b = DQ_1b.' * DQ_1b;

% Problem matrices construction
w_1b = 1;   % Parameter for convergence of x1 component
RQ_1b = [sqrt(4*w_1b^2 + 1), 0, 0, 0;
                          0, 1, 0, 0;
                          0, 0, 0, 0;
                          0, 0, 0, 0];

M_1b = ones(1,T);

% Z function and vectors definition
Z_1b = @(x) [x; Q_1b(x)];
Z0_1b = zeros([s_1b, T]);
for k = 1:T 
    Z0_1b(:,k) = Z_1b(X0_1(:,k));
end

% CVX problem
cvx_begin sdp
    variable P_1b(n,n) symmetric
    variable Y1_1b(T,n)
    variable G2_1b(T,s_1b-n)
    variable a_1b
    a_1b >= 0;                                                                              %#ok
    P_1b >= zeros(n);                                                                       %#ok
    Z0_1b*Y1_1b == [P_1b; zeros(s_1b-n,n)];                                                 %#ok
    [X1_1*Y1_1b + Y1_1b'*X1_1' + a_1b*eye(n),         X1_1*G2_1b,         P_1b*RQ_1b;
                                G2_1b'*X1_1',       -eye(s_1b-n), zeros(s_1b-n,r_1b);
                                RQ_1b'*P_1b', zeros(r_1b,s_1b-n),         -eye(r_1b)] <= 0; %#ok
    Z0_1b*G2_1b == [zeros(n,s_1b-n); eye(s_1b-n)];                                          %#ok
    M_1b * [Y1_1b, G2_1b] == zeros(1,s_1b);                                                 %#ok
cvx_end

% Closed-loop gain computation
G1_1b = Y1_1b/P_1b;
G_1b  = [G1_1b, G2_1b];
K_1b  = U0*G_1b;
CL_1b = X1_1*G_1b;


%% Example 3a
% Ex3 with 1st choice of Q(x)
q_3a = 1;       % Number of non-linearities
s_3a = n+q_3a;  % Number of Z(x) entries
r_3a = n;       % R matrix size

% Non-linearities definition and gradient check
Q_3a = @(x) [cos(x(1))];
DQ_3a = jacobian(Q_3a(x), x);
DDQ_3a = DQ_3a.' * DQ_3a;   % Condition on Q gradient wrt x

% Problem matrices construction
RQ_3a = [1, 0, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0];
RQe_3a = [RQ_3a; zeros(p,r_3a)];

M_3a = ones(1,T);

% Z function and vectors definition
Z_3a = @(x) [x; Q_3a(x)];   % note: this x contains also the state augmentation
Z0_3a = zeros([n+p+q_3a, T]);
Z1_3a = zeros([n+p, T]);
for k = 1:T 
    Z0_3a(:,k) = Z_3a(X0_3(:,k));
    Z1_3a(:,k) = X1_3(:,k);
end

% CVX problem
cvx_begin sdp
    variable P_3a(n+p,n+p) symmetric
    variable Y1_3a(T,n+p)
    variable G2_3a(T,s_3a-n)
    variable a_3a
    a_3a >= 0;                                                                                  %#ok
    P_3a >= zeros(n+p);                                                                         %#ok
    Z0_3a*Y1_3a == [P_3a; zeros(s_3a-n,n+p)];                                                   %#ok
    [Z1_3a*Y1_3a + Y1_3a'*Z1_3a' + a_3a*eye(n+p),        Z1_3a*G2_3a,        P_3a*RQe_3a;
                                   G2_3a'*Z1_3a',       -eye(s_3a-n), zeros(s_3a-n,r_3a);
                                   RQe_3a'*P_3a', zeros(r_3a,s_3a-n),         -eye(r_3a)] <= 0; %#ok
    Z0_3a*G2_3a == [zeros(n+p,s_3a-n); eye(s_3a-n)];                                            %#ok
    M_3a * [Y1_3a, G2_3a] == zeros(1,s_3a+p);                                                   %#ok
cvx_end

% Closed-loop gain computation
G1_3a = Y1_3a/P_3a;
G_3a  = [G1_3a, G2_3a];
K_3a  = U0*G_3a;
CL_3a = Z1_3a*G_3a;


%% Example 3b
% Ex3 with 2nd choice of Q(x)
q_3b = 2;       % Number of non-linearities
s_3b = n+q_3b;  % Number of Z(x) entries
r_3b = n;       % R matrix size

% Non-linearities definition and gradient check
Q_3b = @(x) [cos(x(1)); sin(x(2))];
DQ_3b = jacobian(Q_3b(x), x);
DDQ_3b = DQ_3b.' * DQ_3b;   % Condition on Q gradient wrt x

% Problem matrices construction
RQ_3b = [1, 0, 0, 0;
         0, 1, 0, 0;
         0, 0, 0, 0;
         0, 0, 0, 0];
RQe_3b = [RQ_3b; zeros(p,r_3b)];

M_3b = ones(1,T);

% Z function and vectors definition
Z_3b = @(x) [x; Q_3b(x)];   % note: this x contains also the state augmentation
Z0_3b = zeros([n+p+q_3b, T]);
Z1_3b = zeros([n+p, T]);
for k = 1:T 
    Z0_3b(:,k) = Z_3b(X0_3(:,k));
    Z1_3b(:,k) = X1_3(:,k);
end

% CVX problem
cvx_begin sdp
    variable P_3b(n+p,n+p) symmetric
    variable Y1_3b(T,n+p)
    variable G2_3b(T,s_3b-n)
    variable a_3b
    a_3b >= 0;                                                                                  %#ok
    P_3b >= zeros(n+p);                                                                         %#ok
    Z0_3b*Y1_3b == [P_3b; zeros(s_3b-n,n+p)];                                                   %#ok
    [Z1_3b*Y1_3b + Y1_3b'*Z1_3b' + a_3b*eye(n+p),        Z1_3b*G2_3b,        P_3b*RQe_3b;
                                   G2_3b'*Z1_3b',       -eye(s_3b-n), zeros(s_3b-n,r_3b);
                                   RQe_3b'*P_3b', zeros(r_3b,s_3b-n),         -eye(r_3b)] <= 0; %#ok
    Z0_3b*G2_3b == [zeros(n+p,s_3b-n); eye(s_3b-n)];                                            %#ok
    M_3b * [Y1_3b, G2_3b] == zeros(1,s_3b+p);                                                   %#ok
cvx_end

% Closed-loop gain computation
G1_3b = Y1_3b/P_3b;
G_3b  = [G1_3b, G2_3b];
K_3b  = U0*G_3b;
CL_3b = Z1_3b*G_3b;


%% Closed-Loop Simulation Data
T_cl = 250;                 % Closed-loop number of samples
ts_cl = ts;                 % Closed-loop sampling-time
tf_cl = T_cl*ts_cl;         % Closed-loop final time
times_cl = 0:ts_cl:tf_cl;   % Closed-loop sample times vector

% Initial conditions, noise and reference definition
x0_cl_mag = 1.0;
x0_cl = x0_cl_mag .* 2 .* (rand([n+p, 1]) - 0.5);

D_cl = [0.1; 0.2; 0.3; 0.4];
R_cl = pi/3;


%% Closed-Loop 1a
% Control law
u_cl_1a = @(x) K_1a * Z_1a(x);

% Closed-loop simulation
sol_cl_1a = ode45(@(t,x) f(x, u_cl_1a(x)), [0 tf_cl], x0_cl(1:n));

X_cl_1a = zeros([n, T_cl+1]);
for k = 1:T_cl+1
    X_cl_1a(:,k) = deval(sol_cl_1a, times_cl(k));
end

% Equilibrium point
eq_cl_1a = vpasolve(A*Z(x) + B*K_1a*Z_1a(x));
x_cl_eq_1a = double([eq_cl_1a.x_1; eq_cl_1a.x_2; eq_cl_1a.x_3; eq_cl_1a.x_4]);


%% Closed-Loop 1b
% Control law
u_cl_1b = @(x) K_1b * Z_1b(x);

% Closed-loop simulation
sol_cl_1b = ode45(@(t,x) f(x, u_cl_1b(x)), [0 tf_cl], x0_cl(1:n));

X_cl_1b = zeros([n, T_cl+1]);
for k = 1:T_cl+1
    X_cl_1b(:,k) = deval(sol_cl_1b, times_cl(k));
end

% Equilibrium point
eq_cl_1b = vpasolve(A*Z(x) + B*K_1b*Z_1b(x));
x_cl_eq_1b = double([eq_cl_1b.x_1; eq_cl_1b.x_2; eq_cl_1b.x_3; eq_cl_1b.x_4]);


%% Closed-Loop 3a
% Control law (x contains also the state augmentation)
u_cl_3a = @(x) K_3a * Z_3a(x);

% Closed-loop simulation
sol_cl_3a = ode45(@(t,x) g(x, u_cl_3a(x), D, R), [0 tf_cl], x0_cl);

X_cl_3a = zeros([n+p, T_cl+1]);
for k = 1:T_cl+1
    X_cl_3a(:,k) = deval(sol_cl_3a, times_cl(k));
end

% Equilibrium point
eq_cl_3a = vpasolve([A*Z(x) + B*K_3a*Z_3a(xxi) + D; r - h(x)]);
x_cl_eq_3a = double([eq_cl_3a.x_1; eq_cl_3a.x_2; eq_cl_3a.x_3; eq_cl_3a.x_4; eq_cl_3a.xi_1]);


%% Closed-Loop 3b
% Control law (x contains also the state augmentation)
u_cl_3b = @(x) K_3b * Z_3b(x);

% Closed-loop simulation
sol_cl_3b = ode45(@(t,x) g(x, u_cl_3b(x), D, R), [0 tf_cl], x0_cl);

X_cl_3b = zeros([n+p, T_cl+1]);
for k = 1:T_cl+1
    X_cl_3b(:,k) = deval(sol_cl_3b, times_cl(k));
end

% Equilibrium point
eq_cl_3b = vpasolve([A*Z(x) + B*K_3b*Z_3b(xxi) + D; r - h(x)]);
x_cl_eq_3b = double([eq_cl_3b.x_1; eq_cl_3b.x_2; eq_cl_3b.x_3; eq_cl_3b.x_4; eq_cl_3b.xi_1]);


%% Plotting
% Visualize closed-loop simulations

% Set plot options
set(groot, "defaulttextinterpreter", "latex");
set(groot, "defaultAxesTickLabelInterpreter", "latex");
set(groot, "defaultLegendInterpreter", "latex");

% Plots Colors
colors = [0.0000 0.4470 0.7410;
          0.8500 0.3250 0.0980;
          0.9290 0.6940 0.1250;
          0.4940 0.1840 0.5560;
          0.4660 0.6740 0.1880];

fig = 0;
figs = cell([1, 3]);
axs = cell([1, n+p]);

% Example 1a
fig = fig + 1;
figs{fig} = figure(fig);
clf;
hold on;
for k = 1:n
  plot(times_cl.', x_cl_eq_1a(k)*ones([1, length(times_cl)]), ":", "Color", colors(k,:));  
end
for k = 1:n
  axs{k} = plot(times_cl.', X_cl_1a(k,:).', "Color", colors(k,:));  
end
hold off;
grid on;
title("\bf{Example 1a}");
legend([axs{1:n}], "$" + string(x.') + "$");
xlabel("Time (Seconds)");
ylabel("Amplitude");

% Example 1b
fig = fig + 1;
figs{fig} = figure(fig);
clf;
hold on;
for k = 1:n
  plot(times_cl.', x_cl_eq_1b(k)*ones([1, length(times_cl)]), ":", "Color", colors(k,:));  
end
for k = 1:n
  axs{k} = plot(times_cl.', X_cl_1b(k,:).', "Color", colors(k,:));  
end
hold off;
grid on;
title("\bf{Example 1b}");
legend([axs{1:n}], "$" + string(x.') + "$");
xlabel("Time (Seconds)");
ylabel("Amplitude");

% Example 3a
fig = fig + 1;
figs{fig} = figure(fig);
clf;
hold on;
for k = 1:n+p
  plot(times_cl.', x_cl_eq_3a(k)*ones([1, length(times_cl)]), ":", "Color", colors(k,:));  
end
for k = 1:n+p
  axs{k} = plot(times_cl.', X_cl_3a(k,:).', "Color", colors(k,:));  
end
hold off;
grid on;
title("\bf{Example 3a}");
legend([axs{1:n+p}], "$" + [string(x.'), "\" + string(xi.')] + "$");
xlabel("Time (Seconds)");
ylabel("Amplitude");

% Example 3b
fig = fig + 1;
figs{fig} = figure(fig);
clf;
hold on;
for k = 1:n+p
  plot(times_cl.', x_cl_eq_3b(k)*ones([1, length(times_cl)]), ":", "Color", colors(k,:));  
end
for k = 1:n+p
  axs{k} = plot(times_cl.', X_cl_3b(k,:).', "Color", colors(k,:));  
end
hold off;
grid on;
title("\bf{Example 3b}");
legend([axs{1:n+p}], "$" + [string(x.'), "\" + string(xi.')] + "$");
xlabel("Time (Seconds)");
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
FileName = ["Assignment_2_1a", "Assignment_2_1b", "Assignment_2_3a", "Assignment_2_3b"];

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