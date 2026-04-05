% Question 2: Three-Component Mixture Regression — Yarn Elongation
% Model (Scheffe special cubic with 6 terms, NO intercept):
%   y_hat = b1*x1 + b2*x2 + b3*x3 + b12*x1*x2 + b13*x1*x3 + b23*x2*x3
%
% Parts:
%  a) Explanation of how mixture regression differs from conventional LR
%  b) X matrix, beta vector, Y vector
%  c) Parameter estimation
%  d) Total, regression, error SS
%  e) Source of error / pure error explanation
%  f) SS via matrix and summation methods
%  g) Variance-covariance matrix
%  h) R^2, adj R^2, PRESS, R^2_PRESS
%  i) Maximize yarn elongation (constrained optimization via fmincon)
%  j) Minimize yarn elongation (constrained optimization via fmincon)

clc; clear; format long g;

% (a) EXPLANATION: Mixture vs Conventional Regression
fprintf('=== (a) Mixture vs Conventional Regression ===\n');
fprintf([
'  In conventional polynomial regression, predictors X1, X2, ... are\n', ...
'  independent and can take any values in their ranges. An intercept (b0)\n', ...
'  is always included, and the design matrix columns are linearly independent.\n\n', ...
'  In MIXTURE regression:\n', ...
'   - The components must sum to 1: x1 + x2 + x3 = 1  (mixture constraint)\n', ...
'   - If an intercept were added, the intercept column (all 1s) would equal\n', ...
'     x1+x2+x3, making the design matrix singular (linear dependency).\n', ...
'   - Therefore, the intercept is OMITTED and the model is parameterized\n', ...
'     in terms of the mixture components directly (Scheffe polynomial).\n', ...
'   - The design space is a simplex (equilateral triangle for 3 components).\n', ...
'   - All standard regression formulas still apply EXCEPT intercept-related\n', ...
'     adjustments (e.g., corrected SS must account for the absence of intercept\n', ...
'     in the X matrix, but we still center about the grand mean for R^2).\n', ...
]);

% DATA — {3,2} Simplex Lattice Design with replicates
% Design matrix columns: [x1, x2, x3, x1*x2, x1*x3, x2*x3]
% Each row is one observation.

% Design point 1:  (1, 0, 0)  — 2 replicates
% Design point 2:  (1/2, 1/2, 0) — 3 replicates
% Design point 3:  (0, 1, 0)  — 2 replicates
% Design point 4:  (0, 1/2, 1/2) — 3 replicates
% Design point 5:  (0, 0, 1)  — 2 replicates
% Design point 6:  (1/2, 0, 1/2) — 3 replicates

X_comp = [
  1,   0,   0;
  1,   0,   0;
  1/2, 1/2, 0;
  1/2, 1/2, 0;
  1/2, 1/2, 0;
  0,   1,   0;
  0,   1,   0;
  0,   1/2, 1/2;
  0,   1/2, 1/2;
  0,   1/2, 1/2;
  0,   0,   1;
  0,   0,   1;
  1/2, 0,   1/2;
  1/2, 0,   1/2;
  1/2, 0,   1/2;
];

Y = [
  11.0; 12.4;                 % point 1
  15.0; 14.8; 16.1;           % point 2
  8.8;  10.0;                 % point 3
  10.0; 9.7;  11.8;           % point 4
  16.8; 16.0;                 % point 5
  17.7; 16.4; 16.6;           % point 6
];

n = length(Y);        % 15
p = 6;                % model parameters

% Group membership (for pure error calculation)
group_id = [1;1; 2;2;2; 3;3; 4;4;4; 5;5; 6;6;6];

% (b) Build X matrix and display matrices
x1 = X_comp(:,1);
x2 = X_comp(:,2);
x3 = X_comp(:,3);

% Design matrix (no intercept — Scheffe model)
X = [x1, x2, x3, x1.*x2, x1.*x3, x2.*x3];

fprintf('\n=== (b) Design Matrix X (15x6), Y vector ===\n');
fprintf('  Columns: [x1, x2, x3, x1*x2, x1*x3, x2*x3]\n');
fprintf('  X = \n'); disp(X);
fprintf('  Y = \n'); disp(Y');

% (c) Parameter Estimation:  beta = (X''X)^{-1} X''Y
XtX  = X' * X;
XtY  = X' * Y;
beta = XtX \ XtY;

Yhat = X * beta;
e    = Y - Yhat;
Ybar = mean(Y);

fprintf('=== (c) Estimated Parameters ===\n');
pnames = {'b1 (x1)', 'b2 (x2)', 'b3 (x3)', 'b12 (x1*x2)', 'b13 (x1*x3)', 'b23 (x2*x3)'};
for j = 1:p
    fprintf('  %-20s = %12.6f\n', pnames{j}, beta(j));
end

% (d & f) Sum of Squares — Matrix and Summation approaches

%--- MATRIX approach ---
% Note: no intercept in X, but we compute corrected SS (about grand mean)
SST_mat = Y'*Y - n*Ybar^2;          % corrected total
SSE_mat = Y'*Y - beta' * XtY;       % residual (model)
SSR_mat = SST_mat - SSE_mat;         % regression

%--- SUMMATION approach ---
SST_sum = sum((Y - Ybar).^2);
SSE_sum = sum(e.^2);
SSR_sum = SST_sum - SSE_sum;

%--- Pure Error SS from replicates ---
groups = unique(group_id);
SSPE = 0;
df_PE = 0;
for g = groups'
    idx = group_id == g;
    yg  = Y(idx);
    SSPE  = SSPE  + sum((yg - mean(yg)).^2);
    df_PE = df_PE + (sum(idx) - 1);
end
SSLOF = SSE_mat - SSPE;               % lack-of-fit SS

fprintf('\n=== (d & f) Sum of Squares (Matrix vs Summation) ===\n');
fprintf('  %-30s %10.4f   (summation: %10.4f)\n', 'SST (corrected total)', SST_mat, SST_sum);
fprintf('  %-30s %10.4f   (summation: %10.4f)\n', 'SSR (regression)',      SSR_mat, SSR_sum);
fprintf('  %-30s %10.4f   (summation: %10.4f)\n', 'SSE (residual)',        SSE_mat, SSE_sum);
fprintf('  %-30s %10.4f\n', 'SSPE (pure error)',   SSPE);
fprintf('  %-30s %10.4f\n', 'SSLOF (lack of fit)', SSLOF);
fprintf('  df_PE = %d,  df_LOF = %d,  df_SSE = %d\n', df_PE, n-p-df_PE, n-p);

% (e) Source of Error — Pure Error Explanation
fprintf('\n=== (e) Source of Error & Pure Error ===\n');
fprintf([
'  In this {3,2} simplex lattice design, there are 6 unique design points\n', ...
'  and 6 model parameters (b1...b23). The model therefore perfectly fits\n', ...
'  the mean response at each unique design point.\n\n', ...
'  The ONLY residual variation comes from REPLICATE observations at the\n', ...
'  same composition settings. This error is called "PURE ERROR" because:\n', ...
'   - It reflects genuine experimental variability (measurement noise,\n', ...
'     batch-to-batch variation, etc.) at fixed predictor settings.\n', ...
'   - It is NOT confounded with any systematic lack-of-fit of the model.\n', ...
'   - In this saturated design: SSE = SSPE (all residual is pure error).\n', ...
'  This is why SSLOF ~= 0 above.\n', ...
]);

% (g) Variance-Covariance Matrix
% Use pure error MS as the error variance estimate
s2     = SSPE / df_PE;
VarCov = s2 * inv(XtX);

fprintf('=== (g) Variance-Covariance Matrix ===\n');
fprintf('  s^2 (pure error MS) = %.6f   (sigma_hat = %.4f)\n', s2, sqrt(s2));
fprintf('  Note: given sigma_hat = 0.85 => sigma^2 = %.4f\n\n', 0.85^2);
fprintf('  Var-Cov Matrix = s^2 * (X''X)^{-1}:\n');
disp(VarCov);

% (h) R^2, Adj R^2, PRESS, R^2_PRESS
R2    = SSR_mat / SST_mat;
% adj R^2: degrees of freedom adjustments (no intercept => df_reg = p, df_tot = n-1)
adjR2 = 1 - (SSE_mat / (n - p)) / (SST_mat / (n - 1));

% Hat matrix and PRESS
H   = X * (XtX \ X');
hii = diag(H);
PRESS    = sum((e ./ (1 - hii)).^2);
R2_PRESS = 1 - PRESS / SST_mat;

fprintf('=== (h) Model Fit Statistics ===\n');
fprintf('  R^2          = %.6f\n', R2);
fprintf('  Adj R^2      = %.6f\n', adjR2);
fprintf('  PRESS        = %.4f\n',  PRESS);
fprintf('  R^2_PRESS    = %.6f\n', R2_PRESS);

% Predicted values at each design point
fprintf('\n  Predicted vs Observed:\n');
fprintf('  %-5s %-8s %-8s %-8s\n','Obs','Y','Y_hat','Residual');
for i = 1:n
    fprintf('  %-5d %-8.2f %-8.4f %-8.4f\n', i, Y(i), Yhat(i), e(i));
end

% (i) & (j) Constrained optimization over the simplex
%           x1+x2+x3=1,  x1,x2,x3 >= 0
% 
% Method: reduce to 2D via x3=1-x1-x2, then apply KKT conditions.
%   Step 1 — interior: solve grad f = 0 (2x2 linear system).
%   Step 2 — edges: each edge fixes one xi=0, giving a 1D quadratic;
%             set derivative=0 and clip to valid range.
%   Step 3 — vertices: evaluate at (1,0,0),(0,1,0),(0,0,1).
% No external toolbox required.

% f(x1,x2) with x3 = 1-x1-x2
b = beta;
f2d = @(u) b(1)*u(1) + b(2)*u(2) + b(3)*(1-u(1)-u(2)) + ...
           b(4)*u(1)*u(2) + b(5)*u(1)*(1-u(1)-u(2)) + b(6)*u(2)*(1-u(1)-u(2));

% Gradient of f w.r.t. (x1,x2):
%   df/dx1 = (b1-b3+b5) + (b4-b5-b6)*x2 - 2*b5*x1
%   df/dx2 = (b2-b3+b6) + (b4-b5-b6)*x1 - 2*b6*x2
% Setting grad=0: A*[x1;x2] = c
A_kkt = [-2*b(5),          b(4)-b(5)-b(6);
          b(4)-b(5)-b(6), -2*b(6)       ];
c_kkt = [-(b(1)-b(3)+b(5));
         -(b(2)-b(3)+b(6))];

% Collect all candidate points
cands = zeros(0,2);

% --- Interior critical point (if A is non-singular) ---
if abs(det(A_kkt)) > 1e-12
    u_int = A_kkt \ c_kkt;
    if u_int(1)>=0 && u_int(2)>=0 && sum(u_int)<=1
        cands(end+1,:) = u_int';
    end
end

% --- Edge x1=0: f(0,x2) = b3+(b2-b3)*x2+b6*x2*(1-x2)
%     df/dx2 = (b2-b3)+b6*(1-2*x2) = 0  =>  x2* = ((b2-b3)+b6)/(2*b6)
if abs(b(6)) > 1e-12
    x2s = ((b(2)-b(3))+b(6)) / (2*b(6));
    x2s = max(0, min(1, x2s));
    cands(end+1,:) = [0, x2s];
end
cands(end+1,:) = [0, 0];   % vertex (0,0) -> (0,0,1)
cands(end+1,:) = [0, 1];   % vertex (0,1) -> (0,1,0)

% --- Edge x2=0: f(x1,0) = b3+(b1-b3)*x1+b5*x1*(1-x1)
%     df/dx1 = (b1-b3)+b5*(1-2*x1) = 0  =>  x1* = ((b1-b3)+b5)/(2*b5)
if abs(b(5)) > 1e-12
    x1s = ((b(1)-b(3))+b(5)) / (2*b(5));
    x1s = max(0, min(1, x1s));
    cands(end+1,:) = [x1s, 0];
end
cands(end+1,:) = [1, 0];   % vertex (1,0) -> (1,0,0)

% --- Edge x3=0 (x1+x2=1): f(x1,1-x1) = b2+(b1-b2)*x1+b4*x1*(1-x1)
%     df/dx1 = (b1-b2)+b4*(1-2*x1) = 0  =>  x1* = ((b1-b2)+b4)/(2*b4)
if abs(b(4)) > 1e-12
    x1s = ((b(1)-b(2))+b(4)) / (2*b(4));
    x1s = max(0, min(1, x1s));
    cands(end+1,:) = [x1s, 1-x1s];
end

% Evaluate f at all candidates
fvals = arrayfun(@(k) f2d(cands(k,:)), 1:size(cands,1));

% (i) MAXIMIZE
fprintf('\n=== (i) Maximize Yarn Elongation ===\n');
[best_max, idx_max] = max(fvals);
u_max = cands(idx_max,:);
best_x_max = [u_max(1), u_max(2), 1-u_max(1)-u_max(2)];

fprintf('  Optimal composition for MAXIMUM elongation:\n');
fprintf('    x1 (polyethylene)   = %.6f\n', best_x_max(1));
fprintf('    x2 (polystyrene)    = %.6f\n', best_x_max(2));
fprintf('    x3 (polypropylene)  = %.6f\n', best_x_max(3));
fprintf('    Max predicted elongation = %.4f kg\n', best_max);

% (j) MINIMIZE
fprintf('\n=== (j) Minimize Yarn Elongation ===\n');
[best_min, idx_min] = min(fvals);
u_min = cands(idx_min,:);
best_x_min = [u_min(1), u_min(2), 1-u_min(1)-u_min(2)];

fprintf('  Optimal composition for MINIMUM elongation:\n');
fprintf('    x1 (polyethylene)   = %.6f\n', best_x_min(1));
fprintf('    x2 (polystyrene)    = %.6f\n', best_x_min(2));
fprintf('    x3 (polypropylene)  = %.6f\n', best_x_min(3));
fprintf('    Min predicted elongation = %.4f kg\n', best_min);

fprintf('\n=== (k) Validation ===\n');
fprintf('  Compare the above results with JMP/Minitab/Polymath.\n');
fprintf('  Known sigma_hat from replicates = 0.85 kg (given in problem).\n');
fprintf('  Check: sqrt(s2) = %.4f  (from our pure error variance)\n', sqrt(s2));
