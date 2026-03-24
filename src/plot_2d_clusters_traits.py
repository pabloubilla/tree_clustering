import pandas as pd
import numpy as np
import os
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial'
plt.rc('font', size=8)
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from adjustText import adjust_text


def linewidth():
    return 6.30045

def plot_tsne_interactive(data, tsne_components, clusters, images_dir, color_list):
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2'])
    df['cluster'] = clusters
    df['species'] = data.index

    unique_clusters = np.sort(np.unique(clusters))

    fig = go.Figure()
    for cl in unique_clusters:
        sub = df[df['cluster'] == cl]
        color_hex = to_hex(color_list[(cl-1) % len(color_list)])
        fig.add_trace(go.Scatter3d(
            x=sub['tsne1'], y=sub['tsne2'], z=np.zeros(len(sub)),
            mode='markers',
            marker=dict(size=2, color=color_hex, opacity=0.8),
            hovertext=sub['species'] + ' (cluster ' + str(cl) + ')',
            hoverinfo='text',
            name=str(cl),
        ))

    centroids = df.groupby('cluster')[['tsne1', 'tsne2']].mean()
    for cl in unique_clusters:
        cx, cy = centroids.loc[cl]
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[0],
            mode='text',
            text=[str(cl)],
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hoverinfo='skip',
        ))

    scale = np.ptp(tsne_components, axis=0).mean() * 0.3
    for trait in data.columns:
        reg = LinearRegression().fit(tsne_components, data[trait])
        vec = reg.coef_ / np.linalg.norm(reg.coef_) * scale
        r = np.sqrt(reg.score(tsne_components, data[trait]))
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, 0],
            mode='lines+text',
            text=['', f'{trait} (r={r:.2f})'],
            textposition='top center',
            textfont=dict(size=18, color='black'),
            line=dict(width=3, color='black'),
            showlegend=False,
        ))

    fig.update_layout(scene=dict(
        xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False, showline=False, showbackground=False),
        yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False, showline=False, showbackground=False),
        zaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False, showline=False, showbackground=False, visible=False),
        camera=dict(eye=dict(x=0, y=0, z=2), up=dict(x=0, y=1, z=0)),
        dragmode='pan',
    ))
    fig.write_html(
        os.path.join(images_dir, 'tsne_2d_interactive.html'),
        include_plotlyjs='cdn',
        config={'modeBarButtonsToRemove': ['orbitRotation', 'tableRotation']},
    )


def plot_tsne_all_with_traits(data, tsne_components, clusters, images_dir, color_list, show_traits=True):
    df = pd.DataFrame(tsne_components, columns=['tsne1', 'tsne2'])
    df['cluster'] = clusters

    unique_clusters = np.sort(np.unique(clusters))
    color_map = {cl: color_list[(cl-1) % len(color_list)] for cl in unique_clusters}

    fig, ax = plt.subplots(figsize=(linewidth(), linewidth()))
    markers = ['o','s','P','X','^','p','D'] 

    sns.scatterplot(
        x='tsne1', y='tsne2', hue='cluster', palette=color_map,
        style='cluster', markers=markers, legend='full', data=df, s=5, alpha=0.8, ax=ax
    )
    if show_traits:
        scale = np.ptp(tsne_components, axis=0).mean() * 0.3
        texts = []
        for trait in data.columns:
            reg = LinearRegression().fit(tsne_components, data[trait])
            r = np.sqrt(reg.score(tsne_components, data[trait]))
            if r < 0.3:
                continue
            vec = reg.coef_ / np.linalg.norm(reg.coef_) * scale
            ax.annotate('', xy=(vec[0], vec[1]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.75))
            texts.append(ax.text(vec[0] * 1.12, vec[1] * 1.12, f'{trait}\n(r={r:.2f})',
                    fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)))
        adjust_text(texts)

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=7, fontsize=6, markerscale=2, frameon=False)
    fig.subplots_adjust(top=0.85)
    fname = 'tsne_all_with_traits.png' if show_traits else 'tsne_all.png'
    plt.savefig(os.path.join(images_dir, fname), dpi=400, bbox_inches='tight')
    plt.clf()
    
def main():

    # color_list = [
    #     'olive', 'salmon', 'lightcoral',  'blue', 'magenta', 'lightsalmon', 'darkred', 'lightgreen', 'darkblue',
    #     'green', 'darkgreen', 'deepskyblue', 'orange', 'indigo', 'darkorange', 'lightblue', 'purple',
    #     'lightseagreen', 'pink', 'teal', 'peru', 'plum', 'black', 'sandybrown', 'darkmagenta', 'lime',
    #     'brown', 'lightgreen', 'coral', 'darkcyan', 'khaki', 'darkviolet', 'violet', 'mediumseagreen',
    #     'tomato', 'gray', 'gold', 'salmon', 'orchid', 'yellow', 'turquoise', 'tan']

    color_list = [
        'olive', 'blue', 'teal', 'magenta', 'salmon', 'lightsalmon', 'darkred', 'lightgreen', 'darkblue', 'green', 
        'darkgreen', 'deepskyblue', 'orange', 'indigo', 'darkorange', 'lightblue', 'purple', 'lightseagreen', 'pink', 'lightcoral', 
        'peru', 'plum', 'black', 'sandybrown', 'darkmagenta', 'lime', 'brown', 'lightgreen', 'coral', 'darkcyan', 
        'khaki', 'darkviolet', 'violet', 'mediumseagreen', 'tomato', 'gray', 'gold', 'salmon', 'red', 'yellow', 
        'turquoise', 'tan']


    cluster_dir = "output/consensus/gmm_error1.0_scl/full_data/ward"   
    output_dir = "output/"
    data_dir = "data/processed/"


    clusters = pd.read_csv(os.path.join(cluster_dir, 'final_clusters.csv'), header=None).values.flatten()
    data = pd.read_csv(os.path.join(data_dir, 'traits_pred_log.csv'), index_col=0)
    data = data.drop(columns=['cluster'], errors='ignore')

    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(data)
    plot_tsne_all_with_traits(data, tsne_components, clusters, output_dir, color_list)
    plot_tsne_interactive(data, tsne_components, clusters, output_dir, color_list)
    plot_tsne_all_with_traits(data, tsne_components, clusters, output_dir, color_list, show_traits=False)

if __name__ == '__main__':
    main()