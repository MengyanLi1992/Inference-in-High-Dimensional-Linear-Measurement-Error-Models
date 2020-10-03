function [A1, n_itr, dif_max] = NearestPSDMatrix(mu, eps, B0, Lambda0, Sigma_hat, tol, p)
n_itr = 0;
dif_max = 0;
if (min(eig(Sigma_hat)) >= 0)
    A1 = Sigma_hat;
    return;
end
dif = 1;
A0 = ones(p,p)*100;
while ((dif > tol) & (n_itr <10000))
    %disp(n_itr)
    % Step A
    A1_0 = B0 + Sigma_hat + mu*Lambda0;
    [V, D] = eig(A1_0);
    d = diag(D);
    d(d < eps) = eps;
    A1 = V*diag(d)*V.'; %nearly symmetric
    A1 = (A1 + A1.')/2;
    % calculate the difference between A1 and A0
    dif = norm(A1-A0, 'fro')/norm(A0, 'fro');
    % Step B
    temp_m = A1 - Sigma_hat - mu*Lambda0;
    mask = tril(true(size(temp_m)));
    temp_vec = temp_m(mask); % column vector
    temp_vec_l1 = ProjectOntoL1Ball(temp_vec, mu); %column vector
    temp_vec_diff = temp_vec - temp_vec_l1;
    B1_l = zeros(p);
    B1_l(tril(ones(p))==1) = temp_vec_diff;
    B1 = B1_l + B1_l.' - diag(diag(B1_l)); 
    % Step Lambda
    Lambda1 = Lambda0 - (A1 - B1 - Sigma_hat)/mu;
    
    %dif = norm(B1-A1+Sigma_hat, 'fro')
    A0 = A1;
    B0 = B1;
    Lambda0 = Lambda1; 
    n_itr = n_itr+1;
end
dif_max = max(max(abs(Sigma_hat - A1)));
%dif_fro = norm(Sigma_hat - A1, 'fro');
%disp(dif_fro)
%disp(norm(B1-A1+Sigma_hat, 'fro'))
end
