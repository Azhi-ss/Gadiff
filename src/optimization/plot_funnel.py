
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'

# Data
stages = ["Generation", "Physical Filter", "Feasibility Filter", "Top Tier"]
counts = [5274, 1897, 1313, 5]
# Visual widths (fixed steps for clarity as requested)
widths = np.array([1.0, 0.75, 0.55, 0.25]) 
colors = ['#D3D3D3', '#4A90E2', '#50E3C2', '#F5A623'] # Gray, Blue, Green, Gold

# Filter descriptions between stages
filters = [
    r"$T_g > 250^\circ C$ & $\epsilon < 3.0$",
    "SA Score < 5.0",
    "Fitness Ranking"
]

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate y positions
y_pos = np.arange(len(stages))
height = 0.6

# Draw bars centered at 0
for i in range(len(stages)):
    # Calculate left and right edges to center
    w = widths[i]
    ax.barh(y_pos[i], w, height=height, color=colors[i], edgecolor='white', align='center')
    
    # Annotate with Stage Name and Count inside the bar (or next to it if too small)
    # Since we use fixed widths, all bars should be wide enough.
    label = f"{stages[i]}\n(N={counts[i]})"
    ax.text(0, y_pos[i], label, ha='center', va='center', fontsize=12, fontweight='bold', color='black')

# Add arrows/labels between bars
for i in range(len(stages) - 1):
    # Position between current and next bar
    y_mid = (y_pos[i] + y_pos[i+1]) / 2
    
    # Draw an arrow or just text
    # Let's use an annotation with arrow style
    ax.annotate('', xy=(0, y_pos[i+1] + height/2 + 0.05), xytext=(0, y_pos[i] - height/2 - 0.05),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Add text label for the filter
    # Place it slightly to the right or left of the arrow?
    # Or put the arrow on the side?
    # Let's put the text in a box on the arrow shaft
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    ax.text(0.35, -y_mid - 0.5, filters[i], ha='left', va='center', fontsize=10, bbox=bbox_props) 
    # Wait, y coordinates are 0, 1, 2... increasing upwards? 
    # By default barh plots from bottom up if y_pos is 0,1,2...
    # But usually funnels are top-down. Let's invert the y-axis or arrange y_pos.
    
# Let's arrange y_pos to be top-down: 0 at top.
# Actually standard barh: 0 is bottom.
# Let's define y_pos so Stage 1 is at top.
y_pos = np.arange(len(stages))[::-1] # [3, 2, 1, 0]

# Re-draw loop with correct y_pos
ax.clear()

# Draw funnel from top (Generation) to bottom (Top Tier)
for i in range(len(stages)):
    y = y_pos[i]
    w = widths[i]
    
    # Draw bar
    ax.barh(y, w, height=height, color=colors[i], edgecolor='white', align='center')
    
    # Text
    label = f"{stages[i]}\n(N={counts[i]})"
    text_color = 'white' if i == 1 else 'black' # White text on blue? Maybe black is safer for all.
    # Stage 1 (Gray): Black
    # Stage 2 (Blue): White or Black? Black is fine on light blue. 
    # Stage 3 (Green): Black
    # Stage 4 (Gold): Black
    ax.text(0, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='black')

# Add arrows/labels between bars
for i in range(len(stages) - 1):
    # Current stage is i (top), next is i+1 (below)
    y_top = y_pos[i]
    y_bot = y_pos[i+1]
    y_mid = (y_top + y_bot) / 2
    
    # Draw arrow from bottom of top bar to top of bottom bar
    # Top bar bottom edge: y_top - height/2
    # Bottom bar top edge: y_bot + height/2
    
    ax.annotate('', xy=(0, y_bot + height/2), xytext=(0, y_top - height/2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#555555'))
    
    # Add text label
    # Place text to the right of the arrow
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8)
    ax.text(0.05, y_mid, filters[i], ha='left', va='center', fontsize=10, bbox=bbox_props, color='#333333')

# Remove axes
ax.axis('off')

# Set limits to ensure everything fits
ax.set_xlim(-1, 1) # Since widths are up to 1.0, centered at 0. width/2 = 0.5. 
ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)

# Title
plt.title("Polymer Screening Funnel", fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('Filter_Funnel.png', dpi=300)
print("Saved Filter_Funnel.png")
