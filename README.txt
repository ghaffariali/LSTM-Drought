This folder contains the codes and the input data to predict agricultural droughts for 12 months into the future
based on Enhanced Combined Drought Index (ECDI). Droughts are also measured by using SPI-3 (Standardized Precipitation Index) and the final results are evaluated against USDM (US Drought Monitor) as reference.
The default code uses data from 2001-2021 to train the LSTM prediction model and predict climatic variables (precipitation, temperature, NDVI, and soil moisture) from 2022-01 to 2022-12 at once. All input data are from remote sensing sources from Earth Engine Data Catalogue (EEDC) which are explained in detail in RS_data.csv. There are also additional prediction modules such as 1D-CNN and SARIMA which are used for comparison purposes with the proposed LSTM.

folders:
data: contains input data for climatic variables in the State of Texas
other models: contains codes for other prediction methods and also analysis of input data using FFT.
output: contains prediction results including csv files, maps, tiff files and png files.
usdm: contains codes for processing and visualization of USDM maps.

scripts:
combine_predictions: combines predictions results for different cells into one csv file
confusion_matrix: determines suggested ranges for comparing ECDI vs USDM
corr_matrix: checks the correlation between inputs
functions: contains all functions required to download data from Google Earth Engine and run the code if the same code is to be used for another case study, functions in here can be used to download data from remote sensing sources.
download_data: download data for region of interest using coordinates for the bounding box.
get_cdi_actual: calculate ECDI for historic period (2001-2021) using SPI
get_cdi_predicted_allzscore: calculate ECDI for historic period (2001-2022) but using Z-score instead of SPI
plot_accuracy_maps: plot accuracy maps
plot_accuracy_maps_deep: plot accuracy maps from using deep neural networks as prediction module
plot_inputs: plot input files
plot_learningcurves: plot learning curves for finetuning results
plot_outputs: plot output maps
predict_lstm: main prediction module using LSTM
predict_lstm_finetune - multiple: run multiple scenarios for finetuning the LSTM model
predict_lstm_finetune: run a single scenario for finetuning the LSTM model