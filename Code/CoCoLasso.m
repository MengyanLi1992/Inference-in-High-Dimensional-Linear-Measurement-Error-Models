function [theta_til, SE] = CoCoLasso(Sigma_hat, rho_hat, lambda)
global n mu eps  B0  Lambda0  tol1  p  theta_init  tol2  max_itr  theta0;
% Calculate Sigma_til using NearestPSDMatrix
[Sigma_til, n_itr1, dif_max] = NearestPSDMatrix(mu, eps, B0, Lambda0, Sigma_hat, tol1, p);
% solve the CoCoLasso estimator using CDL1
[theta_til, n_itr2] = CDL1(Sigma_til, rho_hat, lambda, theta_init, tol2, max_itr);
SE = norm(theta_til - theta0)^2;
end







