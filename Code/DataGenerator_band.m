function [W, Z, Y, X, Sigma] = DataGenerator_band(beta0, gamma0, sd_eps, sd_u, rho_x, n, p, width)
% input gamma0 as a column vector
mu = zeros(1,p);
d = -width:1:width;
B = rho_x*ones(p, 2*width+1);
A = spdiags(B, d, p, p);
Sigma = A + (1-rho_x)*eye(p); %Sigma might not be semi-positive-definite!
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