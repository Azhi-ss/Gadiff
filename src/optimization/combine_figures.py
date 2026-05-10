import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

def combine_images():
    base_dir = "/root/code/polyga_project_mmpolymer/results_formal_run/figures"
    img_paths = {
        "fitness": os.path.join(base_dir, "fitness_distribution.png"),
        "tanimoto": os.path.join(base_dir, "tanimoto_trend.png"),
        "tsne": os.path.join(base_dir, "tsne_trajectory_perp30.png")
    }

    # Check files
    for name, path in img_paths.items():
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return

    # Load images
    img_fitness = mpimg.imread(img_paths["fitness"])
    img_tanimoto = mpimg.imread(img_paths["tanimoto"])
    img_tsne = mpimg.imread(img_paths["tsne"])

    # Create figure
    # Layout: 
    # Top: Fitness (spanning full width)
    # Bottom: Tanimoto (Left), t-SNE (Right)
    
    # Adjust figsize to accommodate the images at high resolution
    fig = plt.figure(figsize=(20, 18), dpi=300, facecolor='white')
    
    # Height ratios: Fitness is wide and short (15x6), others are squarer. 
    # Let's give top row less relative height so it fits tightly.
    gs = GridSpec(2, 2, height_ratios=[0.8, 1.2], hspace=0.05, wspace=0.05, figure=fig)

    # 1. Fitness Distribution (Top)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(img_fitness)
    ax1.axis('off')
    # Add label 'a'
    ax1.text(0.0, 1.0, 'a', transform=ax1.transAxes, fontsize=30, fontweight='bold', va='top', ha='left', color='#333333')

    # 2. Tanimoto Trend (Bottom Left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(img_tanimoto)
    ax2.axis('off')
    # Add label 'b'
    ax2.text(0.0, 0.95, 'b', transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top', ha='left', color='#333333')

    # 3. t-SNE Trajectory (Bottom Right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(img_tsne)
    ax3.axis('off')
    # Add label 'c'
    ax3.text(0.0, 0.95, 'c', transform=ax3.transAxes, fontsize=30, fontweight='bold', va='top', ha='left', color='#333333')

    plt.tight_layout()
    
    out_path = os.path.join(base_dir, "combined_analysis_v2.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Combined image saved to {out_path}")

if __name__ == "__main__":
    combine_images()
