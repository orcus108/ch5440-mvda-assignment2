% Question 3: 1-Predictor Logistic Regression
% Data from class example: Experience (predictor) vs Hired (binary outcome)

Experience = [14; 29; 6; 25; 18; 4; 18; 12; 22; 6; 30; 11; 30; 5; 20; 13; 9; 32; 24; 13; 19; 4; 28; 22; 8];
y          = [0;  0;  0;  1;  1;  0;  0;  0;  1;  0;  1;  0;  1;  0;  1;  0;  0;  1;  0;  1;  0;  0;  1;  1;  1];

% Fit logistic regression model: logit(p) = b0 + b1*Experience
b = glmfit(Experience, y, 'binomial', 'link', 'logit');

fprintf('Logistic Regression Parameters:\n');
fprintf('  b0 (Intercept) = %.4f\n', b(1));
fprintf('  b1 (Experience) = %.4f\n', b(2));
fprintf('\nModel: P(y=1) = 1 / (1 + exp(-(%.4f + %.4f * Experience)))\n\n', b(1), b(2));

% Predicted probabilities
y_pred = glmval(b, Experience, 'logit');

% Display comparison table
fprintf('%-12s %-6s %-12s\n', 'Experience', 'y', 'y_Pred');
fprintf('%s\n', repmat('-', 1, 32));
for i = 1:length(y)
    fprintf('%-12d %-6d %-12.6f\n', Experience(i), y(i), y_pred(i));
end

% Plot
figure;
scatter(Experience, y, 60, 'b', 'filled', 'DisplayName', 'Actual (y)');
hold on;
x_range = linspace(min(Experience)-2, max(Experience)+2, 200)';
p_curve = 1 ./ (1 + exp(-(b(1) + b(2)*x_range)));
plot(x_range, p_curve, 'r-', 'LineWidth', 2, 'DisplayName', 'Logistic Curve');
scatter(Experience, y_pred, 40, 'g', 'DisplayName', 'Predicted P(y=1)');
xlabel('Experience (years)');
ylabel('Probability / Outcome');
title('Logistic Regression: Experience vs Hiring');
legend('Location', 'northwest');
grid on;
