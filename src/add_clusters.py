import pandas as pd
import os

def add_labels_and_order_species(clusters_path, species_traits_path, taxonomy_path, output_path):
    # Read the clusters file (assumed to have a single column of labels)
    clusters = pd.read_csv(clusters_path, header=None, names=['Label'])

    # Read the species traits file
    species_traits = pd.read_csv(species_traits_path)

    # Add the labels to the species traits dataframe
    species_traits['Label'] = clusters['Label']

    # Round the values for traits (all columns except 'accepted_bin')
    traits_columns = species_traits.columns.difference(['accepted_bin'])
    species_traits[traits_columns] = species_traits[traits_columns].round()

    # Reorder columns to place 'Label' at the beginning
    columns = ['Label'] + [col for col in species_traits.columns if col != 'Label']
    species_traits = species_traits[['Label', 'accepted_bin']]

    # Sort species by the frequency of labels
    label_counts = species_traits['Label'].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']
    species_traits = species_traits.merge(label_counts, on='Label')
    species_traits = species_traits.sort_values(by='Count', ascending=False).drop(columns='Count')

    # Read the taxonomy file
    taxonomy = pd.read_csv(taxonomy_path)

    # Merge the species traits with the taxonomy file on a common column (assuming 'Species' as the common column)
    final_df = pd.merge(species_traits, taxonomy, on='accepted_bin', how='left')

    # Save the final dataframe to the output path
    final_df.to_csv(output_path, index=False)

# def add_labels_to_species(clusters_path, species_traits_path, taxonomy_path, output_path):
#     # Read the clusters file (assumed to have a single column of labels)
#     clusters = pd.read_csv(clusters_path, header=None, names=['Label'])

#     # Read the species traits file
#     species_traits = pd.read_csv(species_traits_path)

#     # Add the labels to the species traits dataframe
#     species_traits['Label'] = clusters['Label']

#     # Read the taxonomy file
#     taxonomy = pd.read_csv(taxonomy_path)

#     # Merge the species traits with the taxonomy file on a common column (assuming 'Species' as the common column)
#     final_df = pd.merge(species_traits, taxonomy, on='accepted_bin', how='left')

#     # Save the final dataframe to the output path
#     final_df.to_csv(output_path, index=False)

# Paths to the input files
clusters_path = os.path.join('server','gmm_error1.0','full_data','final_clusters.csv')
species_traits_path = os.path.join('data', 'traits_pred_log.csv')
taxonomy_path = os.path.join('data', 'taxonomic_information.csv')
output_path = os.path.join('server','gmm_error1.0','full_data', 'clusters_with_species.csv')

# Execute the function
add_labels_and_order_species(clusters_path, species_traits_path, taxonomy_path, output_path)