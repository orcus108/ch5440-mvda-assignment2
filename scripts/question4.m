% Question 4: Logistic Regression for Dark Chocolate Buying Decision
% Using product of Bernoulli outcomes (MLE via Newton-Raphson / IRLS)
%
%  a) Logistic model with income as the only predictor
%  b) Logistic model with both income and gender as predictors
%  c) Validate against JMP/Minitab (optional section at end)
%
% Data columns: [Obs, Income (1000 EUR), Gender (0=F,1=M), Buy (0=no,1=yes)]
%
% Method: Newton-Raphson (Iteratively Reweighted Least Squares)
%   At each iteration:
%     pi  = sigmoid(X * beta)        [predicted probabilities]
%     W   = diag(pi .* (1-pi))       [weight matrix]
%     grad = X' * (y - pi)           [gradient of log-likelihood]
%     H    = X' * W * X              [Hessian (Fisher information)]
%     beta_new = beta + H \ grad
%   Log-likelihood: sum(y.*log(pi) + (1-y).*log(1-pi))

clc; clear; format long g;

% DATA
Data_Expt = [
 1  2.53  0  1
 2  2.37  1  0
 3  2.72  1  1
 4  2.54  0  0
 5  3.20  1  1
 6  2.94  0  1
 7  3.20  0  1
 8  2.72  1  1
 9  2.93  0  1
10  2.37  0  0
11  2.24  1  1
12  1.91  1  1
13  2.12  0  1
14  1.83  1  1
15  1.92  1  1
16  2.01  0  0
17  2.01  0  0
18  2.23  1  0
19  1.82  0  0
20  2.11  0  0
21  1.75  1  1
22  1.46  1  0
23  1.61  0  1
24  1.57  1  0
25  1.37  0  0
26  1.41  1  0
27  1.51  0  0
28  1.75  1  1
29  1.68  1  1
30  1.62  0  0
];

income = Data_Expt(:,2);
gender = Data_Expt(:,3);
y      = Data_Expt(:,4);
n      = length(y);

fprintf('n = %d observations\n', n);
fprintf('y=1 (purchased): %d,   y=0 (not purchased): %d\n\n', sum(y), sum(1-y));

% HELPER: Newton-Raphson for logistic regression
% sigmoid function (numerically stable)
sigmoid = @(z) 1 ./ (1 + exp(-z));

% log-likelihood
loglik  = @(pi, y) sum(y .* log(pi + 1e-15) + (1-y) .* log(1-pi + 1e-15));

function [beta, pi_hat, ll_hist] = newton_raphson_logistic(X_design, y, max_iter, tol)
    n    = length(y);
    p    = size(X_design, 2);
    beta = zeros(p, 1);         % initialise at zero
    ll_hist = [];

    for iter = 1:max_iter
        z      = X_design * beta;
        pi     = 1 ./ (1 + exp(-z));

        ll     = sum(y .* log(pi + 1e-15) + (1-y) .* log(1-pi + 1e-15));
        ll_hist(end+1) = ll;  %#ok<AGROW>

        % Gradient and Hessian
        grad   = X_design' * (y - pi);               % score vector
        W      = diag(pi .* (1 - pi));                % weight matrix
        H      = X_design' * W * X_design;            % Fisher information

        % Newton step
        delta  = H \ grad;
        beta   = beta + delta;

        if norm(delta) < tol
            fprintf('    Converged at iteration %d  (||delta|| = %.2e)\n', iter, norm(delta));
            break;
        end
        if iter == max_iter
            fprintf('    WARNING: did not fully converge in %d iterations.\n', max_iter);
        end
    end
    pi_hat = 1 ./ (1 + exp(-X_design * beta));
end

% (a) Model 1: Income only
%     logit(pi) = b0 + b1 * income
fprintf('=========================================\n');
fprintf(' (a) Model 1: Income Only\n');
fprintf('     logit(pi) = b0 + b1*income\n');
fprintf('=========================================\n');

X1_design = [ones(n,1), income];
[beta1, pi1, ll1_hist] = newton_raphson_logistic(X1_design, y, 200, 1e-10);

fprintf('  b0 (intercept) = %12.6f\n', beta1(1));
fprintf('  b1 (income)    = %12.6f\n', beta1(2));
fprintf('  Log-likelihood = %12.6f\n', ll1_hist(end));

% Display predictions
fprintf('\n  Obs  Income   Gender  y   pi_hat   Predicted_class\n');
fprintf('  %s\n', repmat('-',1,50));
for i = 1:n
    cls = double(pi1(i) >= 0.5);
    fprintf('  %-4d %-8.2f %-7d %-4d %-8.4f %-d\n', ...
        i, income(i), gender(i), y(i), pi1(i), cls);
end

% Classification accuracy
acc1 = mean(double(pi1 >= 0.5) == y) * 100;
fprintf('\n  Classification accuracy (threshold=0.5): %.1f%%\n', acc1);

% (b) Model 2: Income + Gender
%     logit(pi) = b0 + b1*income + b2*gender
fprintf('\n=========================================\n');
fprintf(' (b) Model 2: Income + Gender\n');
fprintf('     logit(pi) = b0 + b1*income + b2*gender\n');
fprintf('=========================================\n');

X2_design = [ones(n,1), income, gender];
[beta2, pi2, ll2_hist] = newton_raphson_logistic(X2_design, y, 200, 1e-10);

fprintf('  b0 (intercept) = %12.6f\n', beta2(1));
fprintf('  b1 (income)    = %12.6f\n', beta2(2));
fprintf('  b2 (gender)    = %12.6f\n', beta2(3));
fprintf('  Log-likelihood = %12.6f\n', ll2_hist(end));

% Display predictions
fprintf('\n  Obs  Income   Gender  y   pi_hat   Predicted_class\n');
fprintf('  %s\n', repmat('-',1,50));
for i = 1:n
    cls = double(pi2(i) >= 0.5);
    fprintf('  %-4d %-8.2f %-7d %-4d %-8.4f %-d\n', ...
        i, income(i), gender(i), y(i), pi2(i), cls);
end

acc2 = mean(double(pi2 >= 0.5) == y) * 100;
fprintf('\n  Classification accuracy (threshold=0.5): %.1f%%\n', acc2);

% Standard errors and odds ratios
fprintf('\n--- Standard Errors & Odds Ratios (Model 1) ---\n');
W1   = diag(pi1 .* (1-pi1));
FI1  = X1_design' * W1 * X1_design;     % Fisher information at MLE
SE1  = sqrt(diag(inv(FI1)));
fprintf('  %-15s  %10s  %10s  %10s\n', 'Parameter', 'Estimate', 'Std Error', 'Odds Ratio');
pnames1 = {'b0 (intercept)','b1 (income)'};
for j = 1:2
    fprintf('  %-15s  %10.4f  %10.4f  %10.4f\n', pnames1{j}, beta1(j), SE1(j), exp(beta1(j)));
end

fprintf('\n--- Standard Errors & Odds Ratios (Model 2) ---\n');
W2   = diag(pi2 .* (1-pi2));
FI2  = X2_design' * W2 * X2_design;
SE2  = sqrt(diag(inv(FI2)));
pnames2 = {'b0 (intercept)','b1 (income)','b2 (gender)'};
for j = 1:3
    fprintf('  %-15s  %10.4f  %10.4f  %10.4f\n', pnames2{j}, beta2(j), SE2(j), exp(beta2(j)));
end

% Likelihood ratio test: Model 2 vs Model 1
G2   = 2 * (ll2_hist(end) - ll1_hist(end));    % likelihood ratio statistic
df_G = 1;                                        % 1 extra parameter in model 2
fprintf('\n--- Likelihood Ratio Test: Model 2 vs Model 1 ---\n');
fprintf('  G^2 = 2*(LL2 - LL1) = %.4f  (chi-sq, df=%d)\n', G2, df_G);
fprintf('  p-value = %.4f\n', 1 - chi2cdf(G2, df_G));

% Convergence plot (log-likelihood vs iteration)
figure('Name','Q4: Log-Likelihood Convergence');
subplot(1,2,1);
plot(ll1_hist, 'b-o', 'MarkerSize',4);
xlabel('Iteration'); ylabel('Log-Likelihood');
title('Model 1 (Income only)'); grid on;

subplot(1,2,2);
plot(ll2_hist, 'r-o', 'MarkerSize',4);
xlabel('Iteration'); ylabel('Log-Likelihood');
title('Model 2 (Income + Gender)'); grid on;
sgtitle('Q4: Newton-Raphson Convergence');

fprintf('\n=== (c) Validation ===\n');
fprintf('  Compare the above coefficients and log-likelihood values\n');
fprintf('  with output from JMP or Minitab logistic regression.\n');
fprintf('  Expected: b0 ~ -3.5 to -4, b1 ~ 1.5 to 2 for income-only model.\n');
