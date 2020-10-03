%% initialization for data generating
rng default  % For reproducibility
n=200;
p=200;
beta0 = 1;
gamma0 = zeros((p-1), 1);
gamma0(1:2) = [1,-1];
theta0 = [beta0; gamma0];
sd_eps = 0.1;
sd_u = 0.1;
rho_x = 0.2;

% initialization for NearestPSDMatrix
mu=1;
eps=0.0001;
tol1=1e-05;
B0 = eye(p);
Lambda0= eye(p);

% initialization for CDL1
lambda = 0.05;
theta_init = zeros(p,1); %column vector
tol2 = 1e-05;
max_itr = 1000;

% generate the data set
[W, Z, Y, X] = DataGenerator(beta0, gamma0, sd_eps, sd_u, rho_x, n, p);



%% calculate the sparse omega (Dantzig type of estimator)

Q = [X, Z];

lambda1 = 3;

f = [ones(p,1);zeros(p,1)];
A = [zeros(p), Q.'*Q; zeros(p), -Q.'*Q; -eye(p), -eye(p); -eye(p), eye(p)];
b = [Q.'*Y + lambda1*ones(p,1); - Q.'*Y + lambda1*ones(p,1); zeros(2*p,1)];
linprog(f,A,b)
