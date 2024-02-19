close all;clear all;clc;

load("Train.txt");
load("Test.txt");



    %% Set default values for parameters
    beta1 = 0.9;          % Constants for computing moving averages of
   %       gradients and square grdients.
    beta2 = 0.999;
    del = 1e-8;           % a samll constant added to prevent division by zero.
    max_iter = 1000;      % Maximum iteration number
    t=0;
    b=1;                  % Loss parameters
    a=1;
    e=0.05;
    C=10^-6;              % Trade-off parameter
    mew=10^-6;            % kernel parameter
    m=2^5;                % Mini batch size
    gamma= 0.01;          % Learning parameter
    



[RMSE,MAE,pos_error,neg_error,time] = Adam_function_HawkEye(Train,Test,a,b,e,C,beta1,beta2,gamma,max_iter,del,t,m,mew);





 disp(RMSE);                                                                                                   