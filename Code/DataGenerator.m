function [W, Z, Y, X, Sigma] = DataGenerator(beta0, gamma0, sd_eps, sd_u, rho_x, n, p)
% input gamma0 as a column vector
mu = zeros(1,p);
r = [];
for j=1:p
   r(j) = rho_x^(j-1); 
end
Sigma = toeplitz(r);
%Sigma = ones(p)*rho_x + (1 - rho_x)*eye(p);
Q = mvnrnd(mu,Sigma,n); % Q is n times p matrix
%Q = normalize(Q);
%r = vecnorm(Q);
%Q = Q./r*sqrt(n);
X = Q(:,1);
Z = Q(:, 2:p);
U = normrnd(0, sd_u, [n,1]);
W = X + U;
eps = normrnd(0, sd_eps, [n,1]);
Y = beta0*X + Z*gamma0 + eps;
end
