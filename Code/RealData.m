global n p sd_u mu eps tol1 B0 Lambda0 theta_init tol2 max_itr;

rng(1)
mydata = readtable("DATA_final1.csv");
Data = table2array(mydata);

n = 273;
p = 679;
sd_u =  0.917012;
m4_u = 15.35303;

%testing
beta_star = 0;

% initialization for NearestPSDMatrix
mu=1;
eps=0.0001;
tol1=1e-05;
B0 = eye(p);
Lambda0= eye(p);

% initialization for CDL1
theta_init = zeros(p,1); %column vector
tol2 = 1e-05;
max_itr = 10000;

% Read in the data
Y = Data(:,1);
W = Data(:,2);
Z = Data(:,3:680);
Q = [W, Z];
Data1 = [W, Z, Y];  
% calculate Sigma_hat
Sigma_hat = Q.'*Q/n;
Sigma_hat(1,1) = Sigma_hat(1,1) - sd_u^2;
% Nearly sparse: a lot of elements are very close to 0 but not exact 0.

%%CV for lambda1
rng(1)
lambda1_seq = 0.1:0.01:0.25;
K=4;
err1 = [];
flag = [];
for l = 1:length(lambda1_seq)
    [err1(l), flag(l)]= CVDantzig(Q, lambda1_seq(l), K);
end
[val,idx1] = min(err1);
lambda1 = lambda1_seq(idx1);
disp(lambda1);

q = p-1;
Sigma21 = Sigma_hat(2:p, 1);
Sigma22 = Sigma_hat(2:p, 2:p);
[omega, flag_new]= Dantzig(Sigma21, Sigma22, q, lambda1);
omega_hat = omega(p:length(omega));
E1 = 1 - omega_hat'*Sigma21;

%% CoCoLasso initial
rho_hat = Q.'*Y/n;  
% CV for lambda
rng(1)
lambda_seq = 0.13:0.005:0.165;
k=4;
err2 = [];
for j = 1:length(lambda_seq)
    err2(j) = CVCoCoLasso(Data1, lambda_seq(j), K);
end
[val,idx2] = min(err2);
lambda = lambda_seq(idx2);
disp(lambda);
% Initial estimator
theta_til = CoCoLasso_RD(Sigma_hat, rho_hat, lambda);
s0 = sum(theta_til~=0);

%% Estimated decorrelated score 
gamma_til = theta_til(2:p);
beta_til = theta_til(1);
S_decor = estDecor(beta_star, gamma_til, omega_hat, Sigma_hat, rho_hat);
%% variance        
sigma_eps_H0 = sum((Y - beta_star * W - Z*gamma_til ).^2)/n - beta_star^2*sd_u^2;
sigma_betagamma_H0 = (sigma_eps_H0 +  beta_star^2 * sd_u^2)*E1... 
                     + beta_star^2*m4_u + sigma_eps_H0*sd_u^2 - beta_star^2*sd_u^4;
%% Test Statistic 
Tn_hat = S_decor*sigma_betagamma_H0^(-1/2)*sqrt(n);
%% one-step estimator
S_decor_til = estDecor(beta_til, gamma_til, omega_hat, Sigma_hat, rho_hat);
beta_hat = beta_til - S_decor_til/E1;
%% asymptotic variance
% using beta_til
%sigma_eps_til = sum((Y - beta_til * W - Z*gamma_til ).^2)/n - beta_til^2*sd_u^2;
%sigma_betagamma_til = (sigma_eps_til +  beta_til^2 * sd_u^2)*E1... 
%                     + 2*beta_til^2*sd_u^4 + sigma_eps_til*sd_u^2;
%var_til = E1^(-2)*sigma_betagamma_til;
%using beta_hat
sigma_eps_hat = sum((Y - beta_hat * W - Z*gamma_til ).^2)/n - beta_hat^2*sd_u^2;
sigma_betagamma_hat = (sigma_eps_hat +  beta_hat^2 * sd_u^2)*E1... 
                     + beta_hat^2*m4_u + sigma_eps_hat*sd_u^2 - beta_hat^2*sd_u^4;
var_hat = E1^(-2)*sigma_betagamma_hat/n;

P_value = (1 - normcdf(abs(beta_hat/sqrt(var_hat)),0,1))*2;

CI_1 = beta_hat - 1.96*sqrt(var_hat);
CI_2 = beta_hat + 1.96*sqrt(var_hat);
        
result = [lambda1; lambda; beta_til; s0; Tn_hat; beta_hat; sigma_eps_hat; sigma_betagamma_hat; var_hat; P_value; CI_1; CI_2];

csvwrite('output_RD1.txt', result);
csvwrite('theta_til_RD1.txt', theta_til);
csvwrite('omega_hat_RD1.txt', omega_hat);