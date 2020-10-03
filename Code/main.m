%% Initialization for data generating
rng(1)  % For reproducibility
global n p theta0 sd_u sd_eps rho_x mu eps tol1 B0 Lambda0 theta_init tol2 max_itr;
n=100;
p=250;
beta0 = 1;
beta_star = 1;
gamma0 = zeros((p-1), 1);
gamma0(1) = 1;
theta0 = [beta0; gamma0];
sd_u = 0.1;
m4_u = 3*sd_u^4;
sd_eps = 0.2;
rho_x = 0.25;

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

for sim = 1:100
    disp(sim)
    % generate the data set
    [W, Z, Y, X, Sigma] = DataGenerator(beta0, gamma0, sd_eps, sd_u, rho_x, n, p);
    Data = [W, Z, Y];  
    % calculate Sigma_hat and rho_hat
    Q = [W, Z];
    Sigma_hat = Q.'*Q/n;
    Sigma_hat(1,1) = Sigma_hat(1,1) - sd_u^2;
    rho_hat = Q.'*Y/n;    
    %% Fit the model using the selected lambda
    lambda = sqrt(log(p)/n)*2;
    [theta_til0(:,sim), SE] = CoCoLasso(Sigma_hat, rho_hat, lambda);
    %% second step
    id = find(theta_til0(:,sim)~=0);
    Q_new = Q(:,id);
    Sigma_hat_new = Q_new.'*Q_new/n;
    if (theta_til0(1,sim)~=0)
        Sigma_hat_new(1,1) = Sigma_hat_new(1,1) - sd_u^2;
    end
    rho_hat_new = Q_new.'*Y/n;
    theta_init_new = theta_til0(id);
    [theta_til1, n_itr] = CDL1(Sigma_hat_new, rho_hat_new, 0, theta_init_new, tol1, max_itr);
    %% reformulate the new initial estimator
    theta_til(:,sim) = zeros(p,1);
    theta_til(id,sim) = theta_til1;
    %% Cross validation for lambda1
    lambda1 = 0.1:0.01:0.3;
    K=4;
    err1 = [];
    flag = [];
    for l = 1:length(lambda1)
        [err1(l), flag(l)]= CVDantzig_inf(Data, lambda1(l), K);
        disp(flag(l))
    end
    [err1_value,idx1] = min(err1);
    lambda1 = lambda1(idx1);
    disp(lambda1);
    %% omega_hat
    q = p-1;
    Sigma21 = Sigma_hat(2:p, 1);
    Sigma22 = Sigma_hat(2:p, 2:p);
    omega = Dantzig(Sigma21, Sigma22, q, lambda1);
    omega_hat(:,sim) = omega(p:length(omega));
    %omega0 = (Sigma(1,2:p)*inv(Sigma(2:p, 2:p)))';
    %% Estimated decorrelated score 
    gamma_til = theta_til(2:p,sim);
    S_decor(sim) = estDecor(beta_star, gamma_til, omega_hat(:,sim), Sigma_hat, rho_hat);
    %% Variance 
    sigma_eps_H0(sim) = sum((Y - beta_star * W - Z*gamma_til ).^2)/n - beta_star^2*sd_u^2;
    disp(sigma_eps_H0(sim))
    sigma_betagamma_H0(sim) = (sigma_eps_H0(sim) +  beta_star^2 * sd_u^2)*(1 - omega_hat(:,sim)'*Sigma21)... 
                     + 2*beta_star^2*sd_u^4 + sigma_eps_H0(sim)*sd_u^2;
    disp(sigma_betagamma_H0(sim))
    %sigma_betagamma = (sd_eps^2 +  beta_star^2 * sd_u^2)*(1 - omega0'*Sigma(2:p, 1))... 
                     %+ beta_star^2*m4_u + sd_eps^2 *sd_u^2 - beta_star^2*sd_u^4;
    %% Test Statistic 
    Tn_hat(sim) = S_decor(sim)*sigma_betagamma_H0(sim)^(-1/2)*sqrt(n);
    %% one-step estimator
    beta_til = theta_til(1,sim);
    S_decor_til(sim) = estDecor(beta_til, gamma_til, omega_hat(:,sim), Sigma_hat, rho_hat);
    beta_hat(sim) = beta_til - S_decor_til(sim)/(1 - omega_hat(:,sim)'*Sigma21);
end

result = [S_decor; sigma_eps_H0; sigma_betagamma_H0; Tn_hat; beta_hat];

csvwrite('theta11.txt', theta_til0);

csvwrite('theta_new11.txt', theta_til);

csvwrite('omega11.txt', omega_hat);

csvwrite('output11.txt', result);




