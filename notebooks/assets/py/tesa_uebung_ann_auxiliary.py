from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=18)  # fontsize of the figure title
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys


def get_input_ratio():
    return 10

def load_data(fname, output_horizon=24, test_size=0.2):
    """
    Read file and contruct training and testing data for downstream model
    Parameters:
        fname:
            name of the data file
        output_horizon:
            number of future values to be predicted
        test_size:
            ratio of testing data in all data

    Outputs:
        X_train:    training input
        X_test:     testing input
        Y_train:    training output
        Y_test:     testing output
        x_scaler:   scaler of input to inversely transform transformed input to original scale
        y_scaler:   scaler of output to inversly transform transformed output to original scale
    """


    ###################################################################################################
    ################################ Check for correct output window ##################################
    ###################################################################################################

    if output_horizon < 24 or output_horizon > 336:
        raise ValueError('The output horizon must be an integer value between 24 (1 day) and 336 (14 days).')


    ###################################################################################################
    ################################ Load data from file and plot it ##################################
    ###################################################################################################

    # add capability to read in  csv and xlsx
    if fname.endswith(".csv"):
        df = pd.read_csv(fname, decimal = ",", sep =";")
    elif fname.endswith(".xls") or fname.endswith(".xlsx"):
        df = pd.read_excel(open(fname, "rb"))
    
    load = np.array(df['load'])
    date = np.array(df['ds'])
    if fname.endswith(".csv"): # in csv, dates are read in as str -> convert to datetime for correct handling in plot
        date = np.array(pd.to_datetime(df["ds"]))
    load_series = pd.Series(load, index=df['ds'])


    plt.figure(figsize=(16, 9))
    load_series.plot()
    plt.title("Electricity Load in the German Power System")
    plt.ylabel("Load [MW]")
    plt.xlabel("Point in Time")
    plt.savefig("output/neural_network/load.png")
    plt.close()
    # plt.show()

    
    ###################################################################################################
    ################ Seperate input data into data for training and testing data #######################
    ###################################################################################################

    number_of_samples = load.shape[0]
    input_window = output_horizon * get_input_ratio()

    # Chunks are equally sized sample subsets that have the same size as the required output dataset
    # The model is trained based on chunks - the prediction is then again made based on chunks
    number_of_chunks = (number_of_samples-input_window-output_horizon)//output_horizon
    
    # Instantiate variables for training and testing data
    X_all = np.empty([number_of_chunks, input_window])
    Y_all = np.empty([number_of_chunks, output_horizon])
    
    # Allocate entries from the load data set to the chunks
    for idx in range(number_of_chunks):
        st_in = output_horizon * idx
        end_in = input_window + st_in
        end_out = end_in + output_horizon
        X_all[idx,:] = load[st_in:end_in]
        Y_all[idx,:] = load[end_in:end_out]

    # Split dataset into testing and training
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=test_size, shuffle=False)

    # Use only 1 chunk for testing (no rolling prediction)
    X_test = X_test[:1]
    Y_test = Y_test[:1]

    # Prepare the date labels for the final results
    date_start_idx = int(number_of_chunks*output_horizon*(1-test_size))
    date_end_idx = int(date_start_idx + output_horizon)
    date_test_labels = date[date_start_idx:date_end_idx]


    ###################################################################################################
    ####################################### Normalize data ############################################
    ###################################################################################################

    # Normalize the data with MinMaxScaler for better convergence
    # The data is transformed to values between 0 and 1 - for later retransformation the respective scalers are calculated
    x_scaler = None
    y_scaler = None
    
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(X_train)
    y_scaler.fit(Y_train)

    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)

    Y_train = y_scaler.transform(Y_train)
    Y_test = y_scaler.transform(Y_test)
    
    return X_train, X_test, Y_train, Y_test, x_scaler, y_scaler, date_test_labels

def plot_training_history_alt(a):
    return a

def plot_training_history(history, key):
    """
    Plot the training history for the error measures used during the training process over the epochs
    Parameters:
        history:
            contains all the training statistics

    """
    meanings = {"mae" : "Mean Absolute Error", "mse" : "Mean Squared Error", 
                "rmse" : "Root Mean Squared Error", "mape" : "Mean Absolute Percentage Error"}
    
    plt.figure(figsize=(16, 9))
    plt.plot(history[key])
    plt.plot(history['val_{}'.format(key)])
    plt.title('Training and Validation for: {}'.format(meanings[key]))
    plt.ylabel(meanings[key])
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.savefig('output/neural_network/history_{}.png'.format(key))
    plt.show()
    return meanings



def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate the mean absolute percentage error, i.e. the percentage wise deviation between true
    and forecasted load data.
    Parameters:
        y_true:
            time-dependent value of load data that represents the actual load observed
        y_pred:
            time-dependent value of load data that represents the load data forecasted by the model

    Output:
        Mean absolute percentage error as a single scalar
    """

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true+1e-9))) * 100


def postprocess_prediction_results(Y_predictor, Y_test, y_scaler, date_test_labels):
    """
    Calculate the error measures for the model testing. Plot the forecasted against the true load data.
    Parameters:
        Y_predictor:
            Forecasted load data
        Y_test:
            True load data
        y_scaler:
            Conversion factors for retransforming the normalized load data back onto the original scale.
            The normalized load data ranges between 0 and 1 while the original scale represents the
            load in MW.

    """

    # Retransformate data back to the original scale (data has been normalized during process)
    Y_predictor = y_scaler.inverse_transform(Y_predictor)
    Y_test = y_scaler.inverse_transform(Y_test)

    # calculate error measures for testing
    test_mae_orig = mean_absolute_error(Y_test, Y_predictor)
    test_mse_orig = mean_squared_error(Y_test, Y_predictor)
    test_rmse_orig = np.sqrt(test_mse_orig)
    test_mape_orig = mean_absolute_percentage_error(Y_test, Y_predictor)

    print("\nError Measures of Prediction")
    print("Mean Absolute Error (MAE): {}".format(test_mae_orig))
    print("Mean Squared Error (MSE): {}".format(test_mse_orig))
    print("Root Mean Squared Error (RMSE): {}".format(test_rmse_orig, ))
    print("Mean Absolute Percentage Error (MAPE): {}".format(test_mape_orig))

    # Reshape true testing and prediction results to prepare them for plotting: 2D -> 1D 
    y_test_series = np.squeeze(Y_test.reshape([-1, 1]))
    Y_predictor_series = np.squeeze(Y_predictor.reshape([-1, 1]))

    # Plot y_test and Y_predictor
    plt.figure(figsize=(16, 9))
    plt.plot(date_test_labels, y_test_series)
    plt.plot(date_test_labels, Y_predictor_series)
    plt.legend(["True data", "Prediction"])
    plt.title("True and Predicted Testing Data (mae: {0:.2f}, rmse: {1:.2f}, mape: {2:.2f}%)".format(test_mae_orig, test_rmse_orig, test_mape_orig))
    plt.ylabel('Load [MW]')
    plt.xlabel('Point in Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/neural_network/true_pred_test.png")
    plt.show()


def plot_activations(activations_list, x_lower, x_upper, y_lower = -2, y_upper = 2 ):
    """
    Create a plot of the different activation functions.
    Parameters:
        x_lower: 
            Lower boundary of x-values for which to calculate activation function values.
            Also limits x-axis of the plot. The default is -2.
        x_upper:
            Upper boundary of x-values for which to calculate activation function values.
            Also limits x-axis of the plot. The default is 2.
        y_lower:
            Lower y-axis boundary of the plot
        y_upper:
            Upper y-axis boundary of the plot
        size:
            Size of the overall figure to plot. Identical to figsize in plt
        
        Returns:
            None
    """
    # Create DataFrame with x-values and calculate activation function values
    act_df = pd.DataFrame()
    act_df["x"] = np.linspace(x_lower, x_upper)
    act_df["tanh"] = np.tanh(act_df.x)
    act_df["sigmoid"] = 1 / (1 + np.exp(- act_df.x))
    act_df["relu"] = np.maximum(0, act_df.x)
    act_df["linear"] = act_df.x
    
    # Create plot with subplots and populate subplots with the graphs
    no_elements = len(activations_list)
    size = (4 * no_elements, 3)
    
    fig, axes = plt.subplots(1, no_elements, sharex = "col", sharey = "row", figsize = size)
    plt.subplots_adjust(wspace=0.4)
    for i, ax in enumerate(fig.axes):
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(activations_list[i])
        ax.plot(act_df.x, act_df[activations_list[i]]) 
    return None
   

def show_data_excerpt(filename, sheet_name=None):
    """
    Show an excerpt of the data as pandas Dataframe.

    Parameters
    ----------
    filename : string
        filepath of file to read in.
    sheet_name : string, optional
        sheet name to read in. The default is None.

    Returns
    -------
    sample : pd.DataFrame
        Excerpt of the data as Dataframe.

    """
    if ".xls" in filename:
        data = pd.read_excel(filename, sheet_name = sheet_name)
        data.set_index("ds", inplace = True)
    elif ".csv" in filename:
        data= pd.read_csv(filename, sep =";")
        data.set_index("ds", inplace = True)
        sample = data.copy()
        #sample = sample.loc()
        
    return sample

def plot_data_excerpt(filename, start_date="2012-01-01 01:00:00", end_date="2012-12-31 23:00:00", column = "windspeed"):
    """
    Generate a plot of the data the model will be trained with

    Parameters
    ----------
    filename : 
        filepath of file to read in. Is passed into pd.read_csv
    start_date : datetime, optional
        Start date for the plot. The default is "2012-01-01 01:00:00".
    end_date : datetime, optional
        End date for the plot. The default is "2012-12-31 23:00:00".
    column : string, optional
        Which column from the dataset to plot. The default is "windspeed".

    Returns
    -------
    None.

    """
    # Read in data
    data = pd.read_csv(filename, decimal = ",", sep =";")
    data["ds"] = pd.to_datetime(data["ds"])
    data.set_index("ds", inplace = True)
    units = {"windspeed" : "m/s", "temperature" : "°C", "radiation_direct" : "W/m²", "radiation_diffuse" : "W/m²", "load" : "MW"}
    
    
    # Create sample DataFrame
    sample = data.copy()
    sample = sample.loc[start_date : end_date]
    
    # Create and plot
    fig, ax = plt.subplots(1,1, figsize = (16,9))
    ax.plot(sample.index, sample[column])
    ylabel = f"{column} [{units[column]}]"
    ax.set_ylabel(ylabel)
    ax.set_title(column)
    
    #Format date labels
    DateFormat = mdates.DateFormatter("%d.%m.%Y")
    if end_date.toordinal() - start_date.toordinal() <= 4:
        DateFormat = mdates.DateFormatter("%d.%m.%Y %H:%M")
    ax.xaxis.set_major_formatter(DateFormat)
    ax.figure.autofmt_xdate()    
    # Show plot
    plt.show()
                
    return None


