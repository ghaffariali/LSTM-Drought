# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:45:25 2024

@author: ghaff

this script plots learning curves based on train and test loss values. this 
script uses input files which are the output of the predict_lstm_funetune.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def column_to_numpy(df, column):
    arr = []
    for string_value in df[column]:
        string_value = string_value.strip('[]')  # remove square brackets
        values_list = [float(value) for value in string_value.split(',')]  # split by commas and convert to float
        numpy_array = np.array(values_list)
        arr.append(numpy_array)
    return arr

input_dir = r'C:\Projects\Drought\Code\output\lstm\finetunning'
for v in ['NDVI','P','SM','T']:
    frame = []
    for l in ['east','west']:
        for e in ['50','100','150']:
            for c in ['high', 'low','moderate']:
                file = f'{l}_e{e}_48_{v}_0_2067_ft_{c}'                                 # csv file
                result = pd.read_csv(os.path.join(input_dir, f'{v}\\{file}.csv'))       # read csv file
                mse = result['mse'].mean()                                              # calculate average mse
                nse = result['nse'].mean()                                              # calculate average nse
                
                ### plot train and test loss for each input file
                train_loss = column_to_numpy(result, 'train_loss')
                test_loss = column_to_numpy(result, 'test_loss')
                fig, ax = plt.subplots(9,2, figsize=(10,12))
                overfit_train, overfit_test = 0, 0                                      # counter for overfitting in train and test sets
                for i in range(9):
                    ax[i,0].plot(train_loss[i], color='b', label='Train loss')
                    ax[i,1].plot(test_loss[i], color='r', label='Test loss')
                    ax[i,0].set_ylim(0.001,0.1)
                    ax[i,1].set_ylim(0.001, 0.2)
                    ax[i,0].tick_params(axis='y', labelsize=12, labelright=False, labelleft=True)
                    ax[i,1].tick_params(axis='y', labelsize=12, labelright=False, labelleft=True)
                    
                    ### check for overfitting
                    # if train_loss[i][-1] > np.min(train_loss[i]):
                    #     overfit_train += 1
                    #     ax[i,0].text(0.9, 0.5, 'Overfitting', transform=ax[i,0].transAxes, fontsize=12, 
                    #                  verticalalignment='center', horizontalalignment='left')
                    # if test_loss[i][-1] > np.min(test_loss[i]):
                    #     overfit_test += 1
                    #     ax[i,1].text(0.9, 0.5, 'Overfitting', transform=ax[i,1].transAxes, fontsize=12, 
                    #                  verticalalignment='center', horizontalalignment='left')
                    for j in range(2):
                        if i!=8:
                            ax[i,j].set_xticks([])
                            ax[i,j].set_xticklabels([])        
                        # ax[j,i].set_yticks([0.01, 0.03])
                        ax[i,j].tick_params(axis='x', labelsize=12)
                        # ax[j,i].set_xticks([0, 50, 100])
                        ax[i,j].set_title(f'Cell {result["cell"][i]}, {c} complexity', fontsize=14, y=1.0)
                plt.subplots_adjust(wspace=0.15)  # Adjust the vertical spacing between subplots
                plt.subplots_adjust(hspace=0.6)  # Adjust the vertical spacing between subplots
                fig.text(0.5, 0.07, 'Epochs', ha='center', fontsize=18)                             # X-label
                fig.text(0.04, 0.5, 'Loss', va='center', rotation='vertical', fontsize=18)          # Y-label
                fig.suptitle(f'train/test loss results for variable {v}', fontsize=20)
                
                # Add a legend at the bottom with "Train loss" and "Test loss"
                train_handle, _ = ax[0, 0].get_legend_handles_labels()  # Train loss label from ax[0,0]
                test_handle, _ = ax[0, 1].get_legend_handles_labels()   # Test loss label from ax[0,1]
                
                # Combine train and test handles and create a single legend
                fig.legend([train_handle[0], test_handle[0]], ['Train loss', 'Test loss'], loc='lower center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, 0.02))
                fig.savefig(os.path.join(input_dir, f'plots\\{file}.png'), bbox_inches='tight', dpi=300)
                plt.show()
                
                # frame.append([v,l,c,e,mse,nse])                                             # append results to frame
                frame.append([v,l,c,e,mse,nse, overfit_train, overfit_test])                # append results to frame

    # df_out = pd.DataFrame(frame, columns=['variable','location','complexity','epochs','MSE_avg','NSE_avg'])
    df_out = pd.DataFrame(frame, columns=['variable','location','complexity','epochs','MSE_avg','NSE_avg', 'of_train', 'of_test'])
    df_out.to_csv(os.path.join(input_dir, f'df_{v}.csv'), index=None)



