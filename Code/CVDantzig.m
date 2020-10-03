function [err, flag]=CVDantzig(Data, lambda1, K)
global p n sd_u;
idx = randi([1 K],1,n);
err = 0;
for k = 1:K
    % split the data matrix
    test = Data(idx==k,:);
    train = Data(idx~=k,:);
    % calculate the Sigma_hat_tr and rho_hat_tr
    Q_tr = train(:,1:p);
    n_tr=length(train(:,1));
    Sigma_hat_tr = Q_tr.'*Q_tr/n_tr;
    Sigma_hat_tr(1,1) = Sigma_hat_tr(1,1) - sd_u^2;
    Sigma21_tr = Sigma_hat_tr(2:p, 1);
    Sigma22_tr = Sigma_hat_tr(2:p, 2:p);
    % fit the model using the training part
    [omega, flag1(k)] = Dantzig(Sigma21_tr, Sigma22_tr, p-1, lambda1);
    %disp(flag1(k))
    if (flag1(k)~=1)
        err=100000;
    else
        omega_hat_tr = omega(p: length(omega));
        % calculate the Sigma_hat_te and rho_hat_te
        Q_te = test(:,1:p);
        n_te = n - n_tr;
        Sigma_hat_te = Q_te.'*Q_te/n_te;
        Sigma_hat_te(1,1) = Sigma_hat_te(1,1) - sd_u^2;
        Sigma21_te = Sigma_hat_te(2:p, 1);
        Sigma22_te = Sigma_hat_te(2:p, 2:p);
        % calculate the 'corrected' error
        err = err + norm(Sigma21_te - Sigma22_te*omega_hat_tr);        
    end    
end
err = err/K;
if (sum(flag1 == ones(1,K)) ~= K)
%if (sum(flag1) ~= K)
    flag = 0;
else
    flag = 1;
end
end
