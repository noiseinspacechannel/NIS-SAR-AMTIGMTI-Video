import numpy as np
import matplotlib.pyplot as plt
from vehicle_targets import generate_car, generate_tank, generate_fighter_jet, generate_f35, generate_destroyer

def get_metrics(targets):
    positions = np.array([t['position'] for t in targets])
    rcs_values = np.array([t['rcs'] for t in targets])
    
    total_rcs = np.sum(rcs_values)
    
    min_xyz = np.min(positions, axis=0)
    max_xyz = np.max(positions, axis=0)
    dims = max_xyz - min_xyz
    
    return positions, rcs_values, total_rcs, dims

def plot_vehicle(ax, targets, title):
    positions, rcs_vals, tot_rcs, dims = get_metrics(targets)
    
    # Scatter plot
    # Size of marker proportional to RCS (log scale maybe for visibility if huge range?)
    # F-35 has very small RCS elements, Destroyer huge. 
    # Let's simple linear scale but clipped min size so we see points.
    
    sizes = np.clip(rcs_vals * 2, 10, 200) 
    
    sc = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=rcs_vals, cmap='viridis', s=sizes, alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Auto-scale to fit
    max_range = np.array([positions[:,0].max()-positions[:,0].min(), positions[:,1].max()-positions[:,1].min(), positions[:,2].max()-positions[:,2].min()]).max() / 2.0
    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    info_text = f"{title}\nTotal RCS: {tot_rcs:.1f} m^2\nSize: {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} m"
    ax.set_title(info_text, fontsize=10)
    return sc

def main():
    generators = [
        (generate_tank, "Tank"),
        (generate_fighter_jet, "4th Gen Jet"),
        (generate_f35, "F-35 (Low RCS)"),
        (generate_car, "Car"),
        (generate_destroyer, "Destroyer")
    ]
    
    fig = plt.figure(figsize=(18, 10))
    
    # 2 rows, 3 cols
    for i, (gen_func, name) in enumerate(generators):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        targets = gen_func()
        sc = plot_vehicle(ax, targets, name)
        
    # Add a colorbar for the last one to show RCS scale? 
    # RCS scales vary wildly (1 to 1000). 
    # A single colorbar might be misleading. Let's just rely on size/color per plot?
    # Or maybe just print the colorbar.
    
    plt.tight_layout()
    out_file = "targets_preview.png"
    plt.savefig(out_file)
    print(f"Comparison plot saved to {out_file}")

if __name__ == "__main__":
    main()
