import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)  # 10pt font size for text elements
LINEWIDTH = 6.30045

# Define the methods and corresponding readable names
methods = ['gmm_error1.0_scl', 'gmm_error1.0_rnd_scl', 'gmm_error1.0', 'gmm_error1.0_rnd']

method_names = ['Scaled & argmax assignation', 'Scaled & random assignation', 'Not scaled & argmax assignation', 'Not scaled & random assignation']

# Define line styles and pastel colors
line_styles = ['-', ':', '-', ':']
pastel_colors = ['darkmagenta','darkmagenta', 'darkorange', 'darkorange']
markers = ['o', 'v', 'o', 'v']  # Different markers for solid and dashed lines

# Plotting the silhouette scores
plt.figure(figsize=(6.30045, 6.30045 *0.43))

for i in range(len(methods)):
    # Construct the file path
    file_path = f'output/consensus/{methods[i]}/full_data/silhouette_scores_{methods[i]}.csv'
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Find the optimal K (max silhouette score)
    optimal_index = df['Silhouette Score'].idxmax()
    optimal_K = df.loc[optimal_index, 'K']
    optimal_score = df.loc[optimal_index, 'Silhouette Score']
    
    # Plot the data
    plt.scatter(optimal_K, optimal_score, color=pastel_colors[i], linewidth=1, edgecolors='black', marker=markers[i])  # Highlight the optimal K
    plt.plot(df['K'], df['Silhouette Score'], label=f'{method_names[i]}', linestyle=line_styles[i], color=pastel_colors[i], linewidth=1.5)
    
    # Add text for optimal K to the bottom right of the scatter plot with distinct bboxes
    bbox_props = dict(facecolor=pastel_colors[i], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2') if line_styles[i] == '-' else dict(facecolor='white', alpha=0.7, edgecolor=pastel_colors[i], boxstyle='round,pad=0.2')
    
    plt.text(optimal_K + 0.3, optimal_score - 0.012, 
             f'$K^* = {optimal_K}$', fontweight='heavy',
             color='black' if line_styles[i] == ':' else 'white', 
             fontsize=6,  # Smaller font size
             ha='center',
             bbox=bbox_props)  # Transparency with alpha and different box styles for dashed lines

# Add labels, title, and legend
plt.xlabel('Number of clusters ($K$)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.tight_layout(pad = 0.05)
# plt.show()


for method in methods:
    image_path= f'output/consensus/{method}/full_data/images/sil_score_compared.pdf'
    plt.savefig(image_path)
plt.clf()




# # Plotting the density plots
# plt.figure(figsize=(12, 8))

# for method, readable_name, pastel_color in zip(methods.keys(), methods.values(), pastel_colors):
#     # Construct the file path
#     file_path = f'output/consensus/{method}/full_data/consensus_matrix.parquet'
    
#     # Read the parquet file
#     consensus_matrix = pd.read_parquet(file_path)
    
#     # Extract the upper triangle values (excluding the diagonal)
#     upper_triangle_values = consensus_matrix.where(np.triu(np.ones(consensus_matrix.shape), k=1).astype(bool)).stack().values

#     # random sample
#     upper_triangle_values = np.random.choice(upper_triangle_values, 1000, replace = False)

    
#     # Compute the density plot
#     sns.kdeplot(upper_triangle_values, color=pastel_color, label=readable_name, cumulative = True)
    
#     # Compute the integral (area under the curve)
#     # density = sns.kdeplot(upper_triangle_values, color=pastel_color).get_lines()[-1].get_data()
#     # integral_value = np.trapz(density[1], density[0])
    
#     # Update the legend with the integral value
#     # plt.plot([], [], color=pastel_color, label=f'{readable_name} (Area = {integral_value:.2f})')

# # Add labels, title, and legend
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
# plt.show()