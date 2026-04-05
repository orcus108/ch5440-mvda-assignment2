% Question 5: Logistic Regression — Coupon Redemption (Grouped Binomial Data)
%
% Study: 1000 homes, 5 coupon price levels ($5,$10,$15,$20,$30),
%        n_j = 200 homes per level.
%
% Data:
%   X_j = price reduction (predictor)
%   n_j = number of households
%   Y_j = number who redeemed the coupon
%   p_j = Y_j / n_j  (observed proportion)
%
% Model:
%   pi_j = exp(b0 + b1*X_j) / (1 + exp(b0 + b1*X_j))
%
% Method: Binomial MLE via Newton-Raphson
%   Log-likelihood: L = sum_j [ Y_j*log(pi_j) + (n_j-Y_j)*log(1-pi_j) ]
%   Score:    dL/db = sum_j  x_j * (Y_j - n_j*pi_j)
%   Hessian:  d2L/db^2 = -sum_j  n_j * pi_j*(1-pi_j) * x_j * x_j'
%   Update:   beta_new = beta - H^{-1} * score

clc; clear; format long g;

% DATA (Table 14.2)
X_j = [5; 10; 15; 20; 30];     % price reduction ($)
n_j = [200; 200; 200; 200; 200];
Y_j = [30;  55;  70; 100; 137];
p_j = Y_j ./ n_j;               % observed proportions

m   = length(X_j);               % number of groups
J   = m;

fprintf('=== Grouped Binomial Data ===\n');
fprintf('%-8s %-6s %-6s %-10s\n','X_j','n_j','Y_j','Obs p_j');
for j = 1:m
    fprintf('%-8.0f %-6.0f %-6.0f %-10.4f\n', X_j(j), n_j(j), Y_j(j), p_j(j));
end

% Newton-Raphson for Binomial Logistic Regression
% Design matrix for grouped data: [1, X_j]
X_design = [ones(m,1), X_j];

sigmoid  = @(z) 1 ./ (1 + exp(-z));
max_iter = 200;
tol      = 1e-10;

beta  = zeros(2, 1);   % start at [0; 0]
ll_hist = [];

fprintf('\n=== Newton-Raphson Iterations ===\n');
fprintf('  %-5s %-12s %-12s %-16s\n', 'Iter', 'beta_0', 'beta_1', 'Log-Lik');
fprintf('  %s\n', repmat('-',1,50));

for iter = 1:max_iter
    z      = X_design * beta;
    pi_vec = sigmoid(z);              % predicted probabilities

    % Log-likelihood (binomial)
    ll = sum(Y_j .* log(pi_vec + 1e-15) + (n_j - Y_j) .* log(1 - pi_vec + 1e-15));
    ll_hist(end+1) = ll;              %#ok<AGROW>

    % Score vector (gradient)
    score = X_design' * (Y_j - n_j .* pi_vec);

    % Hessian (negative Fisher information)
    w_diag = n_j .* pi_vec .* (1 - pi_vec);
    W_mat  = diag(w_diag);
    H      = -(X_design' * W_mat * X_design);

    % Newton step: beta_new = beta - H^{-1} * score
    delta = -H \ score;
    beta  = beta + delta;

    if mod(iter, 5) == 0 || iter <= 5
        fprintf('  %-5d %-12.6f %-12.6f %-16.6f\n', iter, beta(1), beta(2), ll);
    end

    if norm(delta) < tol
        fprintf('  Converged at iteration %d  (||delta|| = %.2e)\n', iter, norm(delta));
        break;
    end
end

% Results
z_final  = X_design * beta;
pi_final = sigmoid(z_final);

fprintf('\n=== Estimated Logistic Regression Parameters ===\n');
fprintf('  b0 (intercept) = %12.6f\n', beta(1));
fprintf('  b1 (price red) = %12.6f\n', beta(2));
fprintf('  Final Log-Likelihood = %.6f\n', ll_hist(end));

% Standard errors from Fisher information at MLE
FI = X_design' * diag(n_j .* pi_final .* (1-pi_final)) * X_design;
SE = sqrt(diag(inv(FI)));
fprintf('\n  %-15s  %10s  %10s  %10s\n', 'Parameter','Estimate','Std Error','z-stat');
fprintf('  %-15s  %10.4f  %10.4f  %10.4f\n', 'b0 (intercept)', beta(1), SE(1), beta(1)/SE(1));
fprintf('  %-15s  %10.4f  %10.4f  %10.4f\n', 'b1 (price)',     beta(2), SE(2), beta(2)/SE(2));

% Compare predicted proportions with observed
fprintf('\n=== Comparison: Predicted vs Observed Proportions ===\n');
fprintf('  %-10s %-10s %-10s %-10s %-10s\n', 'X_j','n_j','Y_j','Obs p_j','Pred pi_j');
fprintf('  %s\n', repmat('-',1,55));
for j = 1:m
    fprintf('  %-10.0f %-10.0f %-10.0f %-10.4f %-10.4f\n', ...
        X_j(j), n_j(j), Y_j(j), p_j(j), pi_final(j));
end

% Deviance residuals
dev_res = sign(p_j - pi_final) .* sqrt(2 * (Y_j.*log(p_j./pi_final + 1e-15) + ...
          (n_j-Y_j).*log((1-p_j)./(1-pi_final) + 1e-15)));
fprintf('\n  Deviance residuals: [');
fprintf(' %.4f', dev_res);
fprintf(' ]\n');

% Pearson chi-square goodness of fit
pearson_chi2 = sum((Y_j - n_j.*pi_final).^2 ./ (n_j.*pi_final.*(1-pi_final)));
df_gof       = m - 2;   % groups - parameters
fprintf('\n=== Goodness-of-Fit ===\n');
fprintf('  Pearson chi^2 = %.4f  (df = %d)\n', pearson_chi2, df_gof);
fprintf('  p-value = %.4f\n', 1 - chi2cdf(pearson_chi2, df_gof));

% Convergence plot
figure('Name','Q5: Newton-Raphson Convergence');
plot(ll_hist, 'b-o', 'MarkerSize', 5);
xlabel('Iteration'); ylabel('Log-Likelihood');
title('Q5 Binomial MLE — Newton-Raphson Convergence');
grid on;

% Predicted probability curve
X_range = linspace(0, 35, 200);
pi_curve = sigmoid(beta(1) + beta(2)*X_range);

figure('Name','Q5: Logistic Regression Fit');
plot(X_range, pi_curve, 'b-', 'LineWidth', 2); hold on;
scatter(X_j, p_j, 80, 'r', 'filled');
xlabel('Price Reduction ($)'); ylabel('Probability of Redemption');
title('Q5: Logistic Regression — Coupon Redemption');
legend('Fitted model','Observed proportions');
grid on;

fprintf('\n=== Validation (c) ===\n');
fprintf('  Compare b0 and b1 with JMP/Minitab output.\n');
fprintf('  These estimates will be compared with Q6 (ungrouped Bernoulli MLE).\n');
fprintf('  Q6 beta_0 = %.6f,  beta_1 = %.6f  (should match closely)\n', beta(1), beta(2));
