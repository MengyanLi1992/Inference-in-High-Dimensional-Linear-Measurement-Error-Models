function [theta_til, n_itr] = CGD(Sigma_hat, rho_hat, theta_init, lambda, eta, tol)
p = length(theta_init);
dif = 1;
n_itr = 0;
while(dif > tol)
    n_itr = n_itr + 1;
    %disp(n_itr)
    tmp = theta_init - (Sigma_hat*theta_init - rho_hat)/eta;
    tmp1 = max(abs(tmp) - lambda/eta, zeros(p,1));
    theta = sign(tmp).*tmp1;
    dif = norm(theta - theta_init)/norm(theta_init);
    theta_init = theta;
end
theta_til = theta;
end
