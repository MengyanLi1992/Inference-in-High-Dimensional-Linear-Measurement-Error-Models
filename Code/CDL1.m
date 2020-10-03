function [beta, n_itr] = CDL1(Sigma, rho, lambda, beta0, tol, max_itr)
% input beta, rho are column vectors
p = length(beta0);
diff = 1;
n_itr = 0;
% When p is large, not sure whether this kind of stopping rule is
% reasonable
while(diff > tol & n_itr <= max_itr)
    beta = beta0;
    for j = 1:p
        temp = (rho(j) - Sigma(j,:)*beta + Sigma(j,j)*beta(j))/Sigma(j,j);
        beta(j) = wthresh(temp,'s', lambda/Sigma(j,j));        
    end 
    %diff = norm(beta - beta0);
    diff = abs((beta0'*Sigma*beta0/2 - rho'*beta0)/(beta'*Sigma*beta/2 - rho'*beta) - 1);
    %disp(diff)
    beta0 = beta;
    n_itr = n_itr + 1;
end
end