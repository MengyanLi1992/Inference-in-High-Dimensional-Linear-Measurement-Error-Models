global n p theta0 sd_u sd_eps rho_x mu eps tol1 B0 Lambda0 theta_init tol2 max_itr;

rng(1)  % For reproducibility 
beta0_vec = [1, 1.05, 1.1, 1.15, 1.2];

n=100;
p=250;
beta_star = 1;
gamma0 = zeros((p-1), 1);
gamma0(1) = 1;
sd_u = 0.1;
m4_u = 3*sd_u^4;
sd_eps = 0.2;
rho_x = 0.5;

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

N_sim = 100;
result_mul = double.empty([7,length(beta0_vec),0]);

for sim = 1:N_sim
    % generate the data set
    [W, Z, Y_1, X, Sigma] = DataGenerator(beta_star, gamma0, sd_eps, sd_u, rho_x, n, p);
    %Data = [W, Z, Y];  
    % calculate Sigma_hat
    Q = [W, Z];
    Sigma_hat = Q.'*Q/n;
    Sigma_hat(1,1) = Sigma_hat(1,1) - sd_u^2;
    %CV for lambda1
    lambda1_seq = 0.1:0.01:0.25;
    K=4;
    err1 = [];
    flag = [];
    for l = 1:length(lambda1_seq)
        [err1(l), flag(l)]= CVDantzig(Q, lambda1_seq(l), K);
    end
    [err1_srt,idx1] = sort(err1);
    lambda1 = lambda1_seq(idx1(1));
    disp(lambda1);
    %% omega_hat
    q = p-1;
    Sigma21 = Sigma_hat(2:p, 1);
    Sigma22 = Sigma_hat(2:p, 2:p);
    [omega, flag_new]= Dantzig(Sigma21, Sigma22, q, lambda1);
    s = 2;
    while (flag_new ~=1)
        lambda1 = lambda1_seq(idx1(s));
        [omega, flag_new]= Dantzig(Sigma21, Sigma22, q, lambda1);
        s = s+1;
    end
    omega_hat(:,sim) = omega(p:length(omega));
    %omega0 = (Sigma(1,2:p)*inv(Sigma(2:p, 2:p)))'; 
    En(sim) = Sigma_hat(1,1) - omega_hat(:,sim)'*Sigma21;
    E(sim) = 1 - omega_hat(:,sim)'*Sigma21;
    
    for kk = 1:length(beta0_vec)
        beta0 = beta0_vec(kk);
        theta0 = [beta0; gamma0];
        Y = Y_1 + (beta0_vec(kk)- beta_star)*X;
        rho_hat = Q.'*Y/n;  
        %% Fit the model using the selected lambda
        lambda = sqrt(log(p)/n)*2;
        [theta_til0(:,kk), SE] = CoCoLasso(Sigma_hat, rho_hat, lambda);
        %% second step
        id = find(theta_til0(:,kk)~=0);
        Q_new = Q(:,id);
        Sigma_hat_new = Q_new.'*Q_new/n;
        if (theta_til0(1,kk)~=0)
            Sigma_hat_new(1,1) = Sigma_hat_new(1,1) - sd_u^2;
        end
        rho_hat_new = Q_new.'*Y/n;
        theta_init_new = theta_til0(id,kk);
        [theta_til1, n_itr] = CDL1(Sigma_hat_new, rho_hat_new, 0, theta_init_new, tol1, max_itr);
        %% reformulate the new initial estimator
        theta_til(:,kk) = zeros(p,1);
        theta_til(id,kk) = theta_til1;
        %% Estimated decorrelated score 
        gamma_til = theta_til(2:p,kk);
        beta_til = theta_til(1,kk);
        S_decor(kk) = estDecor(beta_star, gamma_til, omega_hat(:,sim), Sigma_hat, rho_hat);
        %% Variance 
        sigma_eps_H0(kk) = sum((Y - beta_star * W - Z*gamma_til ).^2)/n - beta_star^2*sd_u^2;
        %disp(sigma_eps_H0(kk))
        sigma_betagamma_H0(kk) = (sigma_eps_H0(kk) +  beta_star^2 * sd_u^2)*E(sim)... 
                     + 2*beta_star^2*sd_u^4 + sigma_eps_H0(kk)*sd_u^2;
        %disp(sigma_betagamma_H0(kk))
        %% Test Statistic 
        Tn_hat(kk) = S_decor(kk)*sigma_betagamma_H0(kk)^(-1/2)*sqrt(n);
        disp(Tn_hat(kk))
        %% one-step estimator
        S_decor_til(kk) = estDecor(beta_til, gamma_til, omega_hat(:,sim), Sigma_hat, rho_hat);
        beta_hat(kk) = beta_til - S_decor_til(kk)/En(sim);
        %% asymptotic variance
        % using beta_til
        sigma_eps_til(kk) = sum((Y - beta_til * W - Z*gamma_til ).^2)/n - beta_til^2*sd_u^2;
        sigma_betagamma_til(kk) = (sigma_eps_til(kk) +  beta_til^2 * sd_u^2)*E(sim)... 
                     + 2*beta_til^2*sd_u^4 + sigma_eps_til(kk)*sd_u^2;
        var_til(kk) = E(sim)^(-2)*sigma_betagamma_til(kk);
        %using beta_hat
        sigma_eps_hat(kk) = sum((Y - beta_hat(kk) * W - Z*gamma_til ).^2)/n - beta_hat(kk)^2*sd_u^2;
        sigma_betagamma_hat(kk) = (sigma_eps_hat(kk) +  beta_hat(kk)^2 * sd_u^2)*E(sim)... 
                     + 2*beta_hat(kk)^2*sd_u^4 + sigma_eps_hat(kk)*sd_u^2;
        var_hat(kk) = E(sim)^(-2)*sigma_betagamma_hat(kk);
        result = [S_decor; sigma_eps_H0; sigma_betagamma_H0; Tn_hat; beta_hat; var_til; var_hat];
    end
    result_mul = cat(3,result_mul ,result);
end

csvwrite('Alt_output21.txt', result_mul);
