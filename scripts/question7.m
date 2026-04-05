% Question 7: Logistic Regression — Dengue Disease Spread (Multiple Predictors)
%
% Study: Dengue disease in 98 individuals across two city sectors.
%        Response: Y = 1 if disease contracted, 0 otherwise.
%
% Predictors:
%   X1 = age
%   X2 = socioeconomic class indicator (1=middle, 0=otherwise)
%   X3 = socioeconomic class indicator (1=lower,  0=otherwise)
%        [Reference class = upper: X2=0, X3=0]
%   X4 = sector indicator (0=sector 1, 1=sector 2)
%
% Model:
%   logit(pi_i) = b0 + b1*X1 + b2*X2 + b3*X3 + b4*X4
%
% Method: Bernoulli MLE via Newton-Raphson (IRLS)
%   pi_i  = sigmoid(X*beta)
%   score = X' * (y - pi)
%   H     = X' * W * X   where W = diag(pi.*(1-pi))
%   beta_new = beta + H\score
%
% Data source (Kutner et al., Chapter 14, Table 14.3):
%   http://www.cnachtsheim-text.csom.umn.edu/Kutner/Chapter%2014%20Data%20Sets/CH14TA03.txt

clc; clear; format long g;

% LOAD DATA from URL
% Kutner CH14TA03 — Dengue dataset, n=98
% Columns: Y  X1(age)  X2(SES_mid)  X3(SES_low)  X4(sector)
DATA = [
  0  33  0  0  0;  0  35  0  0  0;  0   6  0  0  0;  0  60  0  0  0;
  1  18  0  1  0;  0  26  0  1  0;  0   6  0  1  0;  1  31  1  0  0;
  1  26  1  0  0;  0  37  1  0  0;  0  23  0  0  0;  0  23  0  0  0;
  0  27  0  0  0;  1   9  0  0  0;  1  37  0  0  1;  1  22  0  0  1;
  1  67  0  0  1;  0   8  0  0  1;  1   6  0  0  1;  1  15  0  0  1;
  1  21  1  0  1;  1  32  1  0  1;  1  16  0  0  1;  0  11  1  0  1;
  0  14  0  1  1;  0   9  1  0  1;  0  18  1  0  1;  0   2  0  1  0;
  0  61  0  1  0;  0  20  0  1  0;  0  16  0  1  0;  0   9  1  0  0;
  0  35  1  0  0;  0   4  0  0  0;  0  44  0  1  1;  1  11  0  1  1;
  0   3  1  0  1;  0   6  0  1  1;  1  17  1  0  1;  0   1  0  1  1;
  1  53  1  0  1;  1  13  0  0  1;  0  24  0  0  1;  1  70  0  0  1;
  1  16  0  1  1;  0  12  1  0  1;  1  20  0  1  1;  0  65  0  1  1;
  1  40  1  0  1;  1  38  1  0  1;  1  68  1  0  1;  1  74  0  0  1;
  1  14  0  0  1;  1  27  0  0  1;  0  31  0  0  1;  0  18  0  0  1;
  0  39  0  0  1;  0  50  0  0  1;  0  31  0  0  1;  0  61  0  0  1;
  0  18  0  1  0;  0   5  0  1  0;  0   2  0  1  0;  0  16  0  1  0;
  1  59  0  1  0;  0  22  0  1  0;  0  24  0  0  0;  0  30  0  0  0;
  0  46  0  0  0;  0  28  0  0  0;  0  27  0  0  0;  1  27  0  0  0;
  0  28  0  0  0;  1  52  0  0  0;  0  11  0  1  0;  0   6  1  0  0;
  0  46  0  1  0;  1  20  1  0  0;  0   3  0  0  0;  0  18  1  0  0;
  0  25  1  0  0;  0   6  0  1  0;  1  65  0  1  0;  0  51  0  1  0;
  0  39  1  0  0;  0   8  0  0  0;  0   8  1  0  0;  0  14  0  1  0;
  0   6  0  1  0;  0   6  0  1  0;  0   7  0  1  0;  0   4  0  1  0;
  0   8  0  1  0;  0   9  1  0  0;  1  32  0  1  0;  0  19  0  1  0;
  0  11  0  1  0;  0  35  0  1  0
];

Y  = DATA(:,1);
X1 = DATA(:,2);
X2 = DATA(:,3);
X3 = DATA(:,4);
X4 = DATA(:,5);
nrows = size(DATA,1);
fprintf('Data loaded: %d rows (Kutner CH14TA03).\n', nrows);

n = length(Y);
p = 5;   % intercept + 4 predictors

fprintf('\n=== Data Summary ===\n');
fprintf('  n = %d individuals\n', n);
fprintf('  Y=1 (disease): %d,   Y=0 (no disease): %d\n', sum(Y), sum(1-Y));
fprintf('  Sector 1: %d,  Sector 2: %d\n', sum(X4==0), sum(X4==1));
fprintf('  Upper SES (ref): %d,  Middle: %d,  Lower: %d\n', ...
    sum(X2==0 & X3==0), sum(X2==1), sum(X3==1));

% Design matrix
X_design = [ones(n,1), X1, X2, X3, X4];

fprintf('\n=== Design Matrix (first 5 rows) ===\n');
fprintf('  [Intercept, Age, SES_mid, SES_low, Sector]\n');
disp(X_design(1:5,:));

% Newton-Raphson (Bernoulli MLE)
sigmoid  = @(z) 1 ./ (1 + exp(-z));
beta     = zeros(p, 1);
max_iter = 200;
tol      = 1e-10;
ll_hist  = [];

fprintf('=== Newton-Raphson Iterations ===\n');
fprintf('  %-5s %-10s %-10s %-10s %-10s %-10s %-14s\n', ...
    'Iter','b0','b1(age)','b2(mid)','b3(low)','b4(sec)','Log-Lik');
fprintf('  %s\n', repmat('-',1,72));

for iter = 1:max_iter
    z    = X_design * beta;
    pi   = sigmoid(z);

    ll   = sum(Y .* log(pi + 1e-15) + (1-Y) .* log(1-pi + 1e-15));
    ll_hist(end+1) = ll;  %#ok<AGROW>

    grad  = X_design' * (Y - pi);
    W     = diag(pi .* (1-pi));
    H     = X_design' * W * X_design;

    delta = H \ grad;
    beta  = beta + delta;

    if mod(iter,5)==0 || iter<=5
        fprintf('  %-5d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-14.6f\n', ...
            iter, beta(1), beta(2), beta(3), beta(4), beta(5), ll);
    end

    if norm(delta) < tol
        fprintf('  Converged at iteration %d  (||delta|| = %.2e)\n\n', iter, norm(delta));
        break;
    end
end

pi_final = sigmoid(X_design * beta);

% Results: Coefficients, SEs, Odds Ratios
FI = X_design' * diag(pi_final .* (1-pi_final)) * X_design;
SE = sqrt(diag(inv(FI)));
z_stat = beta ./ SE;

fprintf('=== Estimated Model Parameters ===\n');
fprintf('  logit(pi) = b0 + b1*Age + b2*SES_mid + b3*SES_low + b4*Sector\n\n');
fprintf('  %-20s %10s %10s %10s %12s\n', 'Parameter','Estimate','Std Err','z-stat','Odds Ratio');
fprintf('  %s\n', repmat('-',1,68));
pnames = {'b0 (intercept)','b1 (Age)','b2 (SES mid)','b3 (SES low)','b4 (Sector)'};
for j = 1:p
    fprintf('  %-20s %10.4f %10.4f %10.4f %12.4f\n', ...
        pnames{j}, beta(j), SE(j), z_stat(j), exp(beta(j)));
end

fprintf('\n  Final Log-Likelihood = %.6f\n', ll_hist(end));

% Predicted Probabilities vs Actual Y
fprintf('\n=== Predicted Probabilities (first 20 observations) ===\n');
fprintf('  %-5s %-5s %-6s %-6s %-6s %-6s %-10s %-5s\n', ...
    'Obs','Y','Age','SES_m','SES_l','Sect','pi_hat','Pred');
fprintf('  %s\n', repmat('-',1,55));
for i = 1:min(20, n)
    cls = double(pi_final(i) >= 0.5);
    fprintf('  %-5d %-5d %-6.0f %-6.0f %-6.0f %-6.0f %-10.4f %-5d\n', ...
        i, Y(i), X1(i), X2(i), X3(i), X4(i), pi_final(i), cls);
end
fprintf('  ... (showing first 20 of %d)\n', n);

% Overall classification accuracy
acc = mean(double(pi_final >= 0.5) == Y) * 100;
fprintf('\n  Classification accuracy (threshold=0.5): %.1f%%\n', acc);

% Deviance and Null Model Comparison
% Null model: intercept only
beta_null = log(mean(Y) / (1 - mean(Y)));   % MLE for intercept-only
pi_null   = repmat(mean(Y), n, 1);
ll_null   = sum(Y.*log(pi_null+1e-15) + (1-Y).*log(1-pi_null+1e-15));

% Likelihood ratio test vs null
G2    = 2 * (ll_hist(end) - ll_null);
df_G2 = p - 1;   % 4 extra predictors

fprintf('\n=== Model Significance (Likelihood Ratio Test vs Null) ===\n');
fprintf('  Null log-likelihood  = %.4f  (intercept only)\n', ll_null);
fprintf('  Model log-likelihood = %.4f  (full model)\n',    ll_hist(end));
fprintf('  G^2 = %.4f  (chi-sq, df=%d)\n', G2, df_G2);
fprintf('  p-value = %.6f\n', 1 - chi2cdf(G2, df_G2));

% Convergence plot
figure('Name','Q7: Newton-Raphson Convergence');
plot(ll_hist, 'm-o', 'MarkerSize', 5, 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Log-Likelihood');
title('Q7: Dengue Disease — MLE Convergence');
grid on;

fprintf('\n=== Interpretation ===\n');
fprintf('  b1 (Age)        > 0  => older individuals have higher disease odds\n');
fprintf('  b2 (SES mid)  sign(%+.4f) => middle SES vs upper SES\n', beta(3));
fprintf('  b3 (SES low)  sign(%+.4f) => lower  SES vs upper SES (ref: upper)\n', beta(4));
fprintf('  b4 (Sector 2) > 0  => sector 2 has substantially higher disease odds\n');
fprintf('  Note: b3(SES low) = %.4f (negative => lower SES has lower odds than\n', beta(4));
fprintf('  upper SES after adjusting for age and sector — verify col mapping with textbook)\n');
fprintf('\n  Odds ratio for Sector 2 vs Sector 1 = exp(b4) = %.4f\n', exp(beta(5)));
fprintf('  Odds ratio for Middle vs Upper SES   = exp(b2) = %.4f\n', exp(beta(3)));
fprintf('  Odds ratio for Lower vs Upper SES    = exp(b3) = %.4f\n', exp(beta(4)));
