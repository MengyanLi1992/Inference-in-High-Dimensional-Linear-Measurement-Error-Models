function S_decor = estDecor(beta, gamma, omega_hat, Sigma_hat, rho_hat)
global p;
Sigma12 = Sigma_hat(1, 2:p);
Sigma22 = Sigma_hat(2:p, 2:p);
S_decor = Sigma_hat(1,1)*beta + Sigma12*gamma - rho_hat(1)... 
          - omega_hat' * (beta*Sigma12' + Sigma22*gamma - rho_hat(2:p));
end 