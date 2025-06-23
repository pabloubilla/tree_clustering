import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  # 10pt font size for text elements
LINEWIDTH = 6.30045
# import tikzplotlib
# # from matplotlib.backends.backend_pgf import _tex_escape as mpl_common_texification

# Define the path to the folder containing the CSV files
method = sys.argv[1]
if method == 'gmm_error1.0_scl':
    color = 'darkmagenta'
    color2 = 'thistle'
    name = 'Scaled'
if method == 'gmm_error1.0':
    color = 'darkorange'
    color2 = 'navajowhite'
    name = 'Not scaled'
output_dir = f'output/consensus/{method}/full_data/'
save_dir = 'output/consensus/gmm_error1.0_scl/full_data/' # BEST METHOD DIR
folder_path = os.path.join(output_dir, 'BIC')
# csv_files = sorted([os.path.join(folder_path, f'BIC_{i}.csv') for i in range(1, 21)])
csv_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])


# Read all CSV files into a list of dataframes
dataframes = [pd.read_csv(file) for file in csv_files if os.path.isfile(file)]

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# Ensure that BIC values are numeric
combined_df['BIC'] = pd.to_numeric(combined_df['BIC'])

# Group by 'N_components' and calculate the mean and standard deviations
grouped = combined_df.groupby('N_components')['BIC']
mean_bic = grouped.mean()
std_bic = grouped.std()

# Plot the results ALL CURVES ##
length = 6.30045 * 0.48
plt.figure(figsize=(length, length*3/4))

S = len(dataframes)

# Plot all individual BIC values with transparency
for i, df in enumerate(dataframes):
    if i == 0:
        # Create a separate line just for the legend with the higher alpha
        plt.plot([], [], color=color2, alpha=1.0, label=f'BIC curves', linewidth=1)
        # Plot the actual curve with lower alpha
        plt.plot(df['N_components'], df['BIC'], color=color2, alpha=0.4, linewidth=.7)
    else:
        plt.plot(df['N_components'], df['BIC'], color=color2, alpha=0.4, linewidth=.7)

# Plot the mean BIC line on top with a more pastel color
plt.plot(mean_bic.index, mean_bic, label='Mean BIC', color=color, linewidth=1)
print(mean_bic)

# Adding titles and labels
plt.xlabel('Number of groups $(G)$')
plt.ylabel('BIC')
# plt.title(name)
plt.legend(title = name)
G_min, G_max = plt.xlim() # Get limits for other plots
plt.tight_layout(pad = 0.1)
# Save the plot
plt.savefig(os.path.join(save_dir, 'images', f'BIC_w_lines_{name}.pdf'))
# tikzplotlib.save(os.path.join(save_dir, 'images', f'BIC_w_lines_{name}.tex'))

plt.close()



### OPTIMAL G ###
# Find the optimal G for each CSV file
optimal_G = []


for df in dataframes:
    min_bic_idx = df['BIC'].idxmin()
    optimal_G.append(df.loc[min_bic_idx, 'N_components'])

# Plot histogram of optimal G values
plt.figure(figsize=(length, length*2/3))
# plt.hist(optimal_G, bins=range(int(min(optimal_G)), int(max(optimal_G)) + 2), color='skyblue', edgecolor='black')
sns.histplot(optimal_G, bins=len(np.unique(optimal_G)), discrete=True, color=color)
plt.xlabel('Optimal Number of Groups $(G^{*})$')
plt.ylabel('Frequency')
# Set x-ticks to be discrete
# plt.xticks(np.arange(min(optimal_G)-1, max(optimal_G) + 2))
# plt.xlim(min(optimal_G)-2,max(optimal_G) + 2)
plt.xlim(G_min, G_max)
plt.tight_layout(pad = 0.1)
# Save the histogram
plt.savefig(os.path.join(save_dir, 'images', f'Optimal_G_Histogram_{name}.pdf'))
# tikzplotlib.save(os.path.join(save_dir, 'images', f'Optimal_G_Histogram_{name}.text'))

plt.close()




# #### WITH STD
# # Plot the results
# plt.figure(figsize=(8, 6))

# # Plot the mean BIC line
# plt.plot(mean_bic.index, mean_bic, label='Mean BIC', color='darkblue', linewidth=2)

# # Plot 1 standard deviation
# plt.fill_between(mean_bic.index, mean_bic - std_bic, mean_bic + std_bic, color='lightblue', alpha=0.5, label=r'$\pm \sigma$')

# # Plot 2 standard deviations
# plt.fill_between(mean_bic.index, mean_bic - 2 * std_bic, mean_bic + 2 * std_bic, color='lightsteelblue', alpha=0.3, label=r'$\pm 2\sigma$')

# # Adding titles and labels
# plt.xlabel('Number of group $(G)$', fontsize=14)
# plt.ylabel('BIC', fontsize=14)
# plt.legend(fontsize=14)

# # Save the plot
# plt.savefig(os.path.join(output_dir, 'images', 'BIC.pdf'))
# plt.close()

# # Show the plot
# # plt.show()



######## COMBINED #####
# # Create subplots with shared x-axis
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# S = len(dataframes)

# # Plot all individual BIC values with transparency on the first axis (ax1)
# for i, df in enumerate(dataframes):
#     if i == 0:
#         ax1.plot(df['N_components'], df['BIC'], color='lightskyblue', alpha=0.5, label=f'BIC curves ($S = {S}$)')
#     else:
#         ax1.plot(df['N_components'], df['BIC'], color='lightskyblue', alpha=np.random.rand()/2)

# # Plot the mean BIC line on top with a more pastel color
# ax1.plot(mean_bic.index, mean_bic, label='Mean BIC', color='black', linewidth=2)

# # Adding labels
# ax1.set_ylabel('BIC', fontsize=14)
# ax1.legend(fontsize=14)
# ax1.grid(False)  # Remove grid

# # Find the optimal G for each CSV file
# optimal_G = [df.loc[df['BIC'].idxmin(), 'N_components'] for df in dataframes]

# # Plot histogram of optimal G values on the second axis (ax2), inverted
# sns.histplot(optimal_G, bins=len(np.unique(optimal_G)), discrete=True, color='skyblue', ax=ax2, alpha=0.6)

# # Invert the histogram and adjust labels
# ax2.invert_yaxis()
# ax2.set_ylabel('Frequency', fontsize=14)
# ax2.grid(False)  # Remove grid

# # Set the x-axis range to match the BIC curves range
# min_x, max_x = ax1.get_xlim()
# ax2.set_xlim(min_x, max_x)

# # Adjust the spacing so the x-labels are between the plots
# fig.subplots_adjust(hspace=0.1)

# # Move the x-axis labels to be centered between the two plots
# ax2.set_xlabel('Number of Groups $(G)$', fontsize=14, labelpad=20)

# # Ensure that tick labels are visible and not overlapped
# ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# ax2.xaxis.set_label_coords(0.5, -0.1)  # Adjust label position

# # Save the combined plot
# # plt.savefig(os.path.join(output_dir, 'images', 'Combined_BIC_and_Histogram.pdf'))
# plt.show()
# plt.close()

# exit()