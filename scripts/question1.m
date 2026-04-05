% Question 1: Second-Order Polynomial Regression — Power Cell Lifecycle
% Model: Y = b0 + b1*X1 + b2*X2 + b11*X1^2 + b22*X2^2 + b12*X1*X2
%
% Three cases:
%   X_raw : raw data
%   X_cod : coded by step size  x = (X - mu) / step_size
%   X_mm  : min-max scaled      z = (X - mu) / (max - min)
%
% Parts:
%  i.   Create raw design matrix X
%  ii.  Create step-size coded matrix x
%  iii. Create min-max scaled matrix z
%  iv.  Regression coefficients for each case
%  v.   SST, SSR, SSE via matrix and summation approaches
%  vi.  Variance-covariance matrix for each case (+ condition numbers)
%  vii. R^2, adj R^2, PRESS, R^2_PRESS for each case

clc; clear; format long g;

% DATA
Y  = [144; 89; 59; 278; 167; 141; 164; 129; 259; 245; 214];
X1 = [0.6; 1.0; 1.4; 0.6; 1.0; 1.0; 1.0; 1.4; 0.6; 1.0; 1.4];
X2 = [10;  10;  10;  20;  20;  20;  20;  20;  30;  30;  30];

n    = length(Y);
p    = 6;       % number of parameters (intercept + 5 terms)
Ybar = mean(Y);

fprintf('n = %d,  Ybar = %.4f\n\n', n, Ybar);

% (i)  RAW design matrix  X_raw = [1, X1, X2, X1^2, X2^2, X1*X2]
X_raw = [ones(n,1), X1, X2, X1.^2, X2.^2, X1.*X2];

fprintf('=== (i) Raw Design Matrix X_raw ===\n');
fprintf('  Columns: [1, X1, X2, X1^2, X2^2, X1*X2]\n');
disp(X_raw);

% (ii)  CODED matrix x  =>  x = (X - mu) / step_size
mu1   = mean(X1);    step1 = 0.4;   % levels: 0.6, 1.0, 1.4  => spacing = 0.4
mu2   = mean(X2);    step2 = 10;    % levels: 10, 20, 30      => spacing = 10

x1 = (X1 - mu1) / step1;
x2 = (X2 - mu2) / step2;

X_cod = [ones(n,1), x1, x2, x1.^2, x2.^2, x1.*x2];

fprintf('=== (ii) Step-Size Coded Matrix X_cod ===\n');
fprintf('  mu_X1 = %.4f,  step_X1 = %.1f\n', mu1, step1);
fprintf('  mu_X2 = %.4f,  step_X2 = %.1f\n', mu2, step2);
fprintf('  Transformation: x = (X - mu) / step\n');
disp(X_cod);

% (iii)  MIN-MAX scaled matrix z  =>  z = (X - mu) / (max - min)
range1 = max(X1) - min(X1);   % 1.4 - 0.6 = 0.8
range2 = max(X2) - min(X2);   % 30  - 10  = 20

z1 = (X1 - mu1) / range1;
z2 = (X2 - mu2) / range2;

X_mm = [ones(n,1), z1, z2, z1.^2, z2.^2, z1.*z2];

fprintf('=== (iii) Min-Max Scaled Matrix X_mm ===\n');
fprintf('  range_X1 = %.1f,  range_X2 = %.1f\n', range1, range2);
fprintf('  Transformation: z = (X - mu) / (max - min)\n');
disp(X_mm);

% (iv)-(vii)  Loop over the three cases
case_names  = {'RAW (X)', 'Coded step-size (x)', 'Min-Max (z)'};
design_mats = {X_raw, X_cod, X_mm};

store = struct();

for k = 1:3
    A    = design_mats{k};
    name = case_names{k};

    %% --- (iv) Regression coefficients ---
    % beta = (A'A)^{-1} A'Y  (normal equations)
    AtA  = A' * A;
    AtY  = A' * Y;
    beta = AtA \ AtY;          % more numerically stable than inv(A'A)*A'Y
    Yhat = A * beta;
    e    = Y - Yhat;           % residuals

    %% --- (v) Sum of Squares ---

    % ---- MATRIX approach ----
    SST_mat = Y'*Y - n*Ybar^2;                  % corrected total SS
    SSE_mat = Y'*Y - beta' * AtY;               % residual SS
    SSR_mat = SST_mat - SSE_mat;                % regression SS

    % ---- SUMMATION approach ----
    SST_sum = sum((Y - Ybar).^2);
    SSE_sum = sum(e.^2);
    SSR_sum = SST_sum - SSE_sum;

    %% --- (vi) Variance-covariance matrix ---
    s2     = SSE_mat / (n - p);
    VarCov = s2 * inv(AtA);                     % (A'A)^{-1} scaled by s^2
    condN  = cond(AtA);                         % condition number of (A'A)

    %% --- (vii) R^2, adj R^2, PRESS, R^2_PRESS ---
    R2       = SSR_mat / SST_mat;
    adjR2    = 1 - (s2) / (SST_mat / (n - 1));

    % Hat matrix diagonal  hii = [A(A'A)^{-1}A']_ii
    H   = A * (AtA \ A');
    hii = diag(H);

    % PRESS statistic
    PRESS    = sum((e ./ (1 - hii)).^2);
    R2_PRESS = 1 - PRESS / SST_mat;

    %% --- Display ---
    fprintf('\n########################################\n');
    fprintf('  CASE %d: %s\n', k, name);
    fprintf('########################################\n');

    fprintf('\n--- (iv) Regression Coefficients ---\n');
    labels = {'b0 (intercept)', 'b1  (X1)',  'b2  (X2)', ...
              'b11 (X1^2)',     'b22 (X2^2)','b12 (X1*X2)'};
    for j = 1:p
        fprintf('  %-20s = %14.6f\n', labels{j}, beta(j));
    end

    fprintf('\n--- (v) Sum of Squares ---\n');
    fprintf('  %-30s %12.4f\n', 'SST  (matrix)',    SST_mat);
    fprintf('  %-30s %12.4f\n', 'SST  (summation)', SST_sum);
    fprintf('  %-30s %12.4f\n', 'SSR  (matrix)',    SSR_mat);
    fprintf('  %-30s %12.4f\n', 'SSR  (summation)', SSR_sum);
    fprintf('  %-30s %12.4f\n', 'SSE  (matrix)',    SSE_mat);
    fprintf('  %-30s %12.4f\n', 'SSE  (summation)', SSE_sum);
    fprintf('  (Matrix and summation approaches agree: yes/no = %s)\n', ...
        ternary(abs(SSE_mat - SSE_sum) < 1e-6, 'YES', 'NO'));

    fprintf('\n--- (vi) Variance-Covariance Matrix ---\n');
    fprintf('  s^2 = %.6f\n', s2);
    disp(VarCov);
    fprintf('  Condition number of (A''A) = %.4e\n', condN);

    fprintf('\n--- (vii) Model Fit Statistics ---\n');
    fprintf('  R^2          = %.6f\n', R2);
    fprintf('  Adj R^2      = %.6f\n', adjR2);
    fprintf('  PRESS        = %.4f\n',  PRESS);
    fprintf('  R^2_PRESS    = %.6f\n', R2_PRESS);

    % Store for final summary
    store(k).name     = name;
    store(k).beta     = beta;
    store(k).SST      = SST_mat;
    store(k).SSR      = SSR_mat;
    store(k).SSE      = SSE_mat;
    store(k).s2       = s2;
    store(k).VarCov   = VarCov;
    store(k).condN    = condN;
    store(k).R2       = R2;
    store(k).adjR2    = adjR2;
    store(k).PRESS    = PRESS;
    store(k).R2PRESS  = R2_PRESS;
end

% Summary comparison across the three cases
fprintf('\n\n=================================================\n');
fprintf('  SUMMARY COMPARISON ACROSS TRANSFORMATIONS\n');
fprintf('=================================================\n');
fprintf('%-28s %14s %14s %14s\n', 'Metric', 'RAW (X)', 'Coded (x)', 'MinMax (z)');
fprintf('%s\n', repmat('-',1,74));
metrics = {'SST','SSR','SSE','R2','adjR2','PRESS','R2PRESS','condN'};
labels2 = {'SST','SSR','SSE','R^2','Adj R^2','PRESS','R^2_PRESS','Cond(A''A)'};
for m = 1:length(metrics)
    v = [store.(metrics{m})];
    fprintf('%-28s %14.4f %14.4f %14.4f\n', labels2{m}, v(1), v(2), v(3));
end
fprintf('\nKey Observations:\n');
fprintf('  1. SST, SSR, SSE are IDENTICAL across all three transformations.\n');
fprintf('     (Transformations are linear; they do not change the fit quality.)\n');
fprintf('  2. R^2 and adj R^2 are also identical.\n');
fprintf('  3. Condition number of (A''A) decreases with coding/scaling.\n');
fprintf('     => Transformed cases give more numerically stable coefficient estimates.\n');
fprintf('  4. The variance-covariance matrix values change across cases,\n');
fprintf('     but the standardized precision (relative to the scale) improves.\n');

% Helper function
function s = ternary(cond, a, b)
    if cond; s = a; else; s = b; end
end
