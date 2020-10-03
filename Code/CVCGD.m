function err=CVCGD(Data, lambda, K)
global mu eps  B0  Lambda0  tol1  p n theta_init  tol2  max_itr sd_u eta;
idx = randi([1 K],1,n);
err = 0;
for k = 1:K
    % split the data matrix
    test = Data(idx==k,:);
    train = Data(idx~=k,:);
    % calculate the Sigma_hat_tr and rho_hat_tr
    Q_tr = train(:,1:p);
    Y_tr = train(:,p+1);
    n_tr=length(Y_tr);
    Sigma_hat_tr = Q_tr.'*Q_tr/n_tr;
    Sigma_hat_tr(1,1) = Sigma_hat_tr(1,1) - sd_u^2;
    rho_hat_tr = Q_tr.'*Y_tr/n_tr;
    % fit the model using the training part
    [theta_til_tr, n_itr2] = CGD(Sigma_hat_tr, rho_hat_tr, theta_init, lambda, eta, tol1);
    % calculate the Sigma_hat_te and rho_hat_te
    Q_te = test(:,1:p);
    Y_te = test(:,p+1);
    n_te = n - n_tr;
    Sigma_hat_te = Q_te.'*Q_te/n_te;
    Sigma_hat_te(1,1) = Sigma_hat_te(1,1) - sd_u^2;
    rho_hat_te = Q_te.'*Y_te/n_te;
    % calculate the 'corrected' error
    err = err + theta_til_tr'*Sigma_hat_te*theta_til_tr - 2*rho_hat_te'*theta_til_tr;
end
err=err/K;
end
