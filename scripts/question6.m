% Question 6: Logistic Regression — Ungrouped Bernoulli MLE
%             with Repeated Averaged Income Predictor Values
%
% The dataset is the Q5 coupon data "ungrouped" into 30 individual Bernoulli
% observations, but with each person's income replaced by the GROUP AVERAGE
% (not the actual individual income). The response y remains individual (0/1).
%
% Grouped averages used as income predictor:
%   Group 1 (originally X=30): avg income = 2.717  (6 obs)
%   Group 2 (originally X=20): avg income = 2.562  (6 obs)
%   Group 3 (originally X=15): avg income = 2.02   (6 obs)
%   Group 4 (originally X=10): avg income = 1.72   (6 obs)
%   Group 5 (originally X=5 ): avg income = 1.5567 (6 obs)
%
% Task: Fit Bernoulli MLE logistic regression and compare the estimated
%       parameters with the Binomial MLE estimates from Question 5.
%
% Key result: Bernoulli MLE with grouped-average predictors gives the
%             SAME beta estimates as Binomial MLE on the grouped data
%             (sufficient statistic argument / equivalence theorem).

clc; clear; format long g;

% DATA
income = [
  2.717; 2.717; 2.717; 2.717; 2.717; 2.717;   % group 1
  2.562; 2.562; 2.562; 2.562; 2.562; 2.562;   % group 2
  2.02;  2.02;  2.02;  2.02;  2.02;  2.02;    % group 3
  1.72;  1.72;  1.72;  1.72;  1.72;  1.72;    % group 4
  1.556667; 1.556667; 1.556667; 1.556667; 1.556667; 1.556667   % group 5
];

y = [
  1; 0; 1; 0; 1; 1;   % group 1
  1; 1; 1; 0; 1; 1;   % group 2
  1; 1; 1; 0; 0; 0;   % group 3
  0; 0; 1; 0; 1; 0;   % group 4
  0; 0; 0; 1; 1; 0;   % group 5
];

n = length(y);
fprintf('n = %d observations\n', n);
fprintf('y=1: %d,  y=0: %d\n\n', sum(y), sum(1-y));

% Newton-Raphson — Bernoulli MLE
% Model: logit(pi_i) = b0 + b1 * income_i
% Log-likelihood: sum_i [ y_i*log(pi_i) + (1-y_i)*log(1-pi_i) ]
% Score:  X' * (y - pi)
% Hessian: X' * W * X  where W = diag(pi.*(1-pi))
% Update: beta_new = beta + (X'WX)^{-1} * X'(y - pi)

X_design = [ones(n,1), income];
sigmoid  = @(z) 1 ./ (1 + exp(-z));

beta    = zeros(2,1);
max_iter = 200;
tol      = 1e-10;
ll_hist  = [];

fprintf('=== Newton-Raphson Iterations (Bernoulli MLE) ===\n');
fprintf('  %-5s %-12s %-12s %-16s\n', 'Iter', 'beta_0', 'beta_1', 'Log-Lik');
fprintf('  %s\n', repmat('-',1,50));

for iter = 1:max_iter
    z    = X_design * beta;
    pi   = sigmoid(z);

    ll   = sum(y .* log(pi + 1e-15) + (1-y) .* log(1-pi + 1e-15));
    ll_hist(end+1) = ll;  %#ok<AGROW>

    grad  = X_design' * (y - pi);           % score
    W     = diag(pi .* (1-pi));             % weight matrix
    H     = X_design' * W * X_design;      % Fisher information (Hessian)

    delta = H \ grad;                       % Newton step
    beta  = beta + delta;

    if mod(iter,5)==0 || iter<=5
        fprintf('  %-5d %-12.6f %-12.6f %-16.6f\n', iter, beta(1), beta(2), ll);
    end
    if norm(delta) < tol
        fprintf('  Converged at iteration %d  (||delta|| = %.2e)\n', iter, norm(delta));
        break;
    end
end

pi_final = sigmoid(X_design * beta);

% Results
fprintf('\n=== Estimated Parameters (Bernoulli MLE, ungrouped) ===\n');
fprintf('  b0 (intercept) = %12.6f\n', beta(1));
fprintf('  b1 (income)    = %12.6f\n', beta(2));
fprintf('  Log-likelihood = %.6f\n\n', ll_hist(end));

% Standard errors
FI = X_design' * diag(pi_final .* (1-pi_final)) * X_design;
SE = sqrt(diag(inv(FI)));
fprintf('  %-15s  %10s  %10s\n', 'Parameter','Estimate','Std Error');
fprintf('  %-15s  %10.4f  %10.4f\n', 'b0 (intercept)', beta(1), SE(1));
fprintf('  %-15s  %10.4f  %10.4f\n', 'b1 (income)',    beta(2), SE(2));

% Predicted probabilities per group
group_income = [2.717; 2.562; 2.02; 1.72; 1.556667];
pi_group     = sigmoid(beta(1) + beta(2)*group_income);

% Observed proportions per group (from the y data)
obs_prop = zeros(5,1);
for g = 1:5
    idx = (income == group_income(g));
    obs_prop(g) = mean(y(idx));
end

fprintf('\n=== Predicted vs Observed (by group) ===\n');
fprintf('  %-12s %-12s %-12s\n', 'Group_income','Obs prop','Pred pi');
for g = 1:5
    fprintf('  %-12.4f %-12.4f %-12.4f\n', group_income(g), obs_prop(g), pi_group(g));
end

% COMPARISON with Grouped Binomial MLE (same dataset, grouped form)
% Group the same chocolate/income data into 5 groups.
% Each group shares one income value; count successes and group size.
%   Group income: [2.717, 2.562, 2.02, 1.72, 1.556667]
%   n_j          = [6, 6, 6, 6, 6]   (6 obs per group)
%   Y_j          = sum(y) per group

group_income = [2.717; 2.562; 2.02; 1.72; 1.556667];
n_grp = zeros(5,1);
Y_grp = zeros(5,1);
for g = 1:5
    idx      = (income == group_income(g));
    n_grp(g) = sum(idx);
    Y_grp(g) = sum(y(idx));
end

% Binomial MLE via Newton-Raphson
%   LL   = sum_j [ Y_j*log(pi_j) + (n_j-Y_j)*log(1-pi_j) ]
%   score = X'*(Y_grp - n_grp.*pi_grp)
%   H     = X'*diag(n_grp.*pi_grp.*(1-pi_grp))*X
X_grp  = [ones(5,1), group_income];
beta_b = zeros(2,1);
for iter = 1:500
    pi_b  = 1 ./ (1 + exp(-X_grp * beta_b));
    sc_b  = X_grp' * (Y_grp - n_grp .* pi_b);
    H_b   = X_grp' * diag(n_grp .* pi_b .* (1 - pi_b)) * X_grp;
    d_b   = H_b \ sc_b;
    beta_b = beta_b + d_b;
    if norm(d_b) < 1e-12; break; end
end

% Standard errors from grouped binomial Fisher information
FI_b  = X_grp' * diag(n_grp .* (1./(1+exp(-X_grp*beta_b))) .* (1 - 1./(1+exp(-X_grp*beta_b)))) * X_grp;
SE_b  = sqrt(diag(inv(FI_b)));

fprintf('\n=== COMPARISON: Bernoulli (Q6) vs Grouped Binomial (same data) ===\n');
fprintf('  %-20s %-16s %-16s\n', 'Parameter','Bernoulli (Q6)','Binomial (grouped)');
fprintf('  %-20s %-16.6f %-16.6f\n', 'b0 (intercept)', beta(1), beta_b(1));
fprintf('  %-20s %-16.6f %-16.6f\n', 'b1 (income)',    beta(2), beta_b(2));
fprintf('\n  Difference b0: %.2e\n', abs(beta(1) - beta_b(1)));
fprintf('  Difference b1: %.2e\n', abs(beta(2) - beta_b(2)));
fprintf('\n  Standard errors:\n');
fprintf('  %-20s %-16s %-16s\n', 'Parameter','Bernoulli SE','Binomial SE');
fprintf('  %-20s %-16.4f %-16.4f\n', 'b0 (intercept)', SE(1), SE_b(1));
fprintf('  %-20s %-16.4f %-16.4f\n', 'b1 (income)',    SE(2), SE_b(2));
fprintf('\n  INTERPRETATION:\n');
fprintf('  Point estimates are IDENTICAL (to numerical precision) because\n');
fprintf('  the sufficient statistic X''y is the same for both likelihoods\n');
fprintf('  when each individual''s predictor equals the group average.\n');
fprintf('  Standard errors are also IDENTICAL here because all individuals within\n');
fprintf('  a group share the exact same predictor value, making the Bernoulli\n');
fprintf('  and Binomial Fisher information matrices algebraically equal.\n');
fprintf('  SEs would differ only if individuals had varying predictor values.\n');

% Convergence plot
figure('Name','Q6: Bernoulli MLE Convergence');
plot(ll_hist, 'g-o', 'MarkerSize', 5);
xlabel('Iteration'); ylabel('Log-Likelihood');
title('Q6: Bernoulli MLE — Newton-Raphson Convergence');
grid on;
