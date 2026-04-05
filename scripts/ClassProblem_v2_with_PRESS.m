clear all;
clc;

% Ref.: Kutner, M. H., C. J. Nachtschiem, J. Netner, 
% Applied Linear Regression Models. 4th ed. New Delhi: McGraw Hill, 2004.

Data = [
  144  0.6  10
   89  1.0  10
   59  1.4  10
  278  0.6  20
  167  1.0  20
  141  1.0  20
  164  1.0  20
  129  1.4  20
  259  0.6  30
  245  1.0  30
  214  1.4  30
  ];

Lifecycle=Data(:,1);
Chargerate=Data(:,2);
Temperature=Data(:,3);

Y=Lifecycle;
X1=Chargerate;
X2=Temperature;

correlX1X2 = corr(X1,X2);
CorrelX1_X1squared = corr(X1,X1.*X1);
CorrelX2_X2squared = corr(X2,X2.*X2);

% Now scale the variables to reduce colinearity

% Ref.: Kutner, M. H., C. J. Nachtschiem, J. Netner, 
% Applied Linear Regression Models. 4th ed. New Delhi: McGraw Hill, 2004.

x1=(X1-mean(X1))/0.4;
x2=(X2-mean(X2))/10;

compareXx=[X1 X2 x1 x2];

X=[ones(numel(X1),1) X1 X2 X1.*X2 X1.^2 X2.^2];
x=[ones(numel(x1),1) x1 x2 x1.*x2 x1.^2 x2.^2];

betahat_X=(inv(X'*X))*(X'*Y);
betahat_x=(inv(x'*x))*(x'*Y);

Res_SSQ1_X=(Y-X*betahat_X)'*(Y-X*betahat_X);
Res_SSQ1_x=(Y-x*betahat_x)'*(Y-x*betahat_x);

YPred_X=X*betahat_X;
YPred_x=x*betahat_x;

Res_SSQ2_X=0;
Res_SSQ2_x=0;

% H Matrices

HX=X*(inv(X'*X))*X';
Hx=x*(inv(x'*x))*x';

%Check for idempotent nature
compare_HX=[HX HX^2];
compare_Hx=[Hx Hx^2];

PRESS_X=0.0;
PRESS_x=0.0;

for i=1:numel(X1)
    Res_SSQ2_X=Res_SSQ2_X+(Y(i)-YPred_X(i))^2;
    Res_SSQ2_x=Res_SSQ2_x+(Y(i)-YPred_x(i))^2;

    % PRESS Calculations
    PRESS_X=PRESS_X + ((Y(i)-YPred_X(i))/(1-HX(i,i)))^2;
    PRESS_x=PRESS_x + ((Y(i)-YPred_x(i))/(1-Hx(i,i)))^2;

end

compare_ResSSQ=[Res_SSQ1_X Res_SSQ2_X   Res_SSQ1_x Res_SSQ2_x];

% Computing Sum of Squaares
Total_SSQ=Y'*Y;
p=size(betahat_X,1); % verify if total number of parameters = 6
dof_Res = numel(X1)-p; % n-p

% Computing estimate of error variance using mean square residual

sigma_hat_X2=Res_SSQ2_X/dof_Res;
sigma_hat_x2=Res_SSQ2_x/dof_Res;

% Variance-Covariance Matrix based on X and x data 

% Comparing precision of parameters

Var_beta_X=inv(X'*X)*sigma_hat_X2; %Variance-Covariance Matrix based on X  data
Var_beta_x=inv(x'*x)*sigma_hat_x2; %Variance-Covariance Matrix based on x  data

% Coefficient of Determination R^2
R2_X=1-Res_SSQ2_X/(Y'*Y-numel(X1)*(mean(Y))^2);
R2_x=1-Res_SSQ2_x/(Y'*Y-numel(x1)*(mean(Y))^2);

R2_PRESSX=1-PRESS_X/(Y'*Y-numel(X1)*(mean(Y))^2);
R2_PRESSx=1-PRESS_x/(Y'*Y-numel(x1)*(mean(Y))^2);

% Adjusted R^2

MeanSq_Res_X=Res_SSQ2_X/(numel(X1)-p);
MeanSq_Res_x=Res_SSQ2_x/(numel(x1)-p);
MeanSq_Total=(Y'*Y-numel(X1)*(mean(Y))^2)/(numel(X1)-1);

AdjR2_X=1-MeanSq_Res_X/MeanSq_Total;
AdjR2_x=1-MeanSq_Res_x/MeanSq_Total;




% Do H matrix in Matrix form

szh=size(HX);

alfa=(Y-X*betahat_X)./(ones(szh(1),1)-diag(HX));
PRESS_XM=alfa'*alfa;

gamma=(Y-x*betahat_x)./(ones(szh(1),1)-diag(Hx));
PRESS_xM=gamma'*gamma;  %% 