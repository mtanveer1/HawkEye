function [RMSE,MAE,pos_error,neg_error,time] = Adam_function_HawkEye(alltrain,test,a,b,e,C,beta1,beta2,gamma,max_iter,del,t,m,mew)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%      Inputs of function
   %       alltrain denotes the training data.
   %       test denotes the test data.
   %       a,b, and e are HawkEye loss parameters.
   %       c and mew are trade-off parameter and kernel parameter,
   %       respectively.
   %       m denotes the size of mini-batch.
   %       max_iter denotes the number of maximum iteration.
   %       beta1 and beta2 are constants for computing moving averages of
   %       gradients and square grdients.
   %       gamma is the learning rate.
   %       del a samll constant added to prevent division by zero.
   %       t denotes the iteration number. 

   %%     Output of function
   %      RMSE, MAE, pos_error, and neg_error are the root mean
   %      square error, mean absolute error, positive error, and negative
   %      error on test data.
   %      time is the training time of the model.


l=size(alltrain,1);
rand_num=randperm(l);
rand_data=zeros(m,size(alltrain,2));

for i=1:m
    rand_data(i,:)=alltrain(rand_num(i),:);
end

%% xrand and yrand are the feature matrix and targets of m randomly selected training samples.
xrand=rand_data(:,1:end-1); yrand=rand_data(:,end);

%% Split the feature and target of the test set
Xtest=test(:,1:end-1);
Ytest=test(:,end);

%% Generating the kernel matrix for the randomly selected training data.
XX=sum(xrand.^2,2)*ones(1,m);
omega=XX+XX'-2*(xrand*xrand');
omega=exp(-omega./(2*mew^2));     % omega is the kernel matrix for data xrand.

%% Initialize the parameters
% n1=size(xrand,2);               % feature in dataset
alpha=0.01*ones(m,1);
r=0.01*ones(m,1);
v=0.01*ones(m,1);

%% finding xi_i

q=zeros(m,1);                     % This is summation term in xi_i
for i=1:m
    q(i)=sum(alpha.*omega(:,i));
end

u=zeros(m,1);                     % This is xi_i
for i=1:m
    u(i)=yrand(i)-q(i);
end

%% finding gradient and Optimization loop for Adam algorithm

% Derivate of loss
E=zeros(m,m);
for i=1:m
    if u(i)>= e                    % Here e is epsilon
        E(i,:)= -b*a^2*(u(i)-e)*exp(-a*(u(i)-e))*omega(i,:)';
    elseif u(i) > -e && u(i) < e
        E(i,:)= zeros(1,m);
    elseif u(i) <= -e
        E(i,:)= -b*a^2*(u(i)+e)*exp(a*(u(i)+e))*omega(i,:)';
    end
end

tic


for i = 1:max_iter
    t = t + 1;


    gradient= omega*alpha + C*sum(E,1)';    % Represents the Equation (9) of the paper



    % Update bias-corrected first and second moment estimates
    r = beta1 .* r + (1 - beta1) .* gradient;                        % First moment estimate (Equation (10) of the paper)
    v = (beta2 .* v) + ((1 - beta2) .* (gradient.^2));               % Second moment estimate (Equation (11) of the paper)
    r_hat = r ./ (1 - beta1^t);                                      % Corrected first moment (Equation (12) of the paper)      
    v_hat = v ./ (1 - beta2^t);                                      % Corrected second moment (Equation (13) of the paper)
    alpha = alpha - ((gamma .* r_hat) ./ (sqrt(v_hat) + del));       % Model parameter (Equation (14) of the paper)
end

   % Return optimal solution 
   alpha_opt = alpha;


XK=xrand; %storing X in another matrix so that all the upgradation while calculating kernel will be done in new matrix.

p=size(Xtest,1);
%HT=zeros(m,n);
omega1=-2*XK*Xtest';
XK=sum(XK.^2,2)*ones(1,p);
Xtest=sum(Xtest.^2,2)*ones(1,m);
omega1=omega1+XK+Xtest';
omega1=exp(-omega1./2*mew^2); %%omega1 is the kernel matrix corresponding to test data projected on training data(including univwersum)

HT=omega1.*yrand;


predicted_Y=HT'*alpha_opt;

time=toc;
nn=size(Xtest,1);
e_nn=ones(nn,1);
error=(Ytest-predicted_Y);
mean_predicted_Y=mean(predicted_Y);
mean_Ytest=mean(Ytest);
%% Finding MAE, RMSE, Pos_error, and Neg_erroe
MAE=sum(abs(error))/nn;
MSE=sum(error.^2)/nn;
RMSE=sqrt(MSE);

% Finding positive and negative errors
% Initialize arrays to store positive and negative errors
positive_error = [];
negative_error = [];

% Iterate through the errors and classify them
for i = 1:length(error)
    if error(i) > 0
        positive_error = [positive_error, error(i)];
    elseif error(i) < 0
        negative_error = [negative_error, error(i)];
    end
end

pos_error=sum(abs(positive_error))/length(positive_error);
neg_error=sum(abs(negative_error))/length(negative_error);

end
