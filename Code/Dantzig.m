function [omega_hat, flag] = Dantzig(Sigma21, Sigma22, q, lambda1)
f = [ones(q,1);zeros(q,1)];
A = [zeros(q), Sigma22; zeros(q), -Sigma22; -eye(q), -eye(q); -eye(q), eye(q)];
b = [Sigma21 + lambda1*ones(q,1); -Sigma21 + lambda1*ones(q,1); zeros(2*q,1)];
[omega_hat, fval, flag] = linprog(f,A,b);
end