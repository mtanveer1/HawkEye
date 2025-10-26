# HawkEye: Advancing Robust Regression with Bounded, Smooth, and Insensitive Loss Function

If you are using this code, please cite the following paper:

Akhtar, M., Tanveer, M., & Arshad, M. (2025). HawkEye: A robust loss function for regression with bounded, smooth, and insensitive zone characteristics. Applied Soft Computing, 113118. https://doi.org/10.1016/j.asoc.2025.113118



If there is any issue/bug in the code please write to phd2101241004@iiti.ac.in or im.mushir.akh@gmail.com.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Description of the files in HawkEye-Code.
   
1. Adam_function_HawkEye.m: This file contains the function of Adam algorithm utilized to solve the proposed HE-LSSVR model. In Adam_function_HawkEye the inputs and their meanings are as follows:
   %       alltrain denotes the training data.
   %       test denotes the test data.
   %       a,b, and e are HawkEye loss parameters.
   %       c and mew are trade-off parameter and kernel parameter, respectively.
   %       m denotes the size of mini-batch.
   %       max_iter denotes the number of maximum iteration.
   %       beta1 and beta2 are constants for computing moving averages of gradients and square grdients.
   %       gamma is the learning rate.
   %       del a samll constant added to prevent division by zero.
   %       t denotes the iteration number. 
   The outputs of Adam_function_HawkEye and their meaning are as follows:
   %      RMSE, MAE, pos_error, and neg_error are the root mean square error, mean absolute error, positive error, and negative error on test data.
   %      time is the training time of the model.


2. Main.m: This is the main file of HE-LSSVR. To utilize this code, you simply need to import the data and execute this script. Within the script, you will be required to provide values for various parameters (such as loss function parameters, Adam algorithm parameters, trade-off parameter, kernel parameter etc.).
To replicate the results achieved with HE-LSSVR, you should adhere to the same instructions outlined in the above paper.
