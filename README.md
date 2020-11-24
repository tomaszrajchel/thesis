Python project concerning the application of the MVA algorithm in the analysis of physics data. 
BDT algorithm is part of the XGBOOST library
The numpy and pandas library are required to run the xgboost package

Before starting install the following packages: numpy, pandas, matplotlib, xgboost 

To run:
run.py

Output:
    output.log - logger 
    performance plots (not saved by default)

Description:

The training.csv file contains signal and background data. Each event is flagged by 's' for the signal and 'b' for the background (Label). 
The sample is split to train and test sample (default: 0.5).
BDT options are in the option.txt file. Leave them unless you know what you are doing.


read_data(input_path) - read input data file
prepare_test_data and prepare_train_data - splitting data file 
read_features(data) - reading variables, input variables have been choosen during preparation of input file

** feature_names = [*data.columns[1:2], *data.columns[4:data.shape[1]]] - skipping 'label' variable since it does't take part in the training of classifier

prepare_train_data_matrix(data_train, feature_names), prepare_test_data_matrix(data_test, feature_names) - preparing data matrix (data for xgboost)

read_option() - read option from option file
read_num_of_trees() - silly function to read number of tree 

train_classifier(param, train_data_matrix, num_trees) - training of classifier

plot_performance_plots(test_data_matrix, booster) - plot performances plot

CM_analysis(test_data_matrix, booster) - matrices calculation and plotting addition performances plot

plot_correlation_matrix(data_train) - plotting correlation matrix