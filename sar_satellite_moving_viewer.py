import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    print("Launching Satellite Moving Destroyer Viewer...")
    
    # Scenarios config
    scenarios = [
        {'label': 'Stationary', 'file': 'sar_satellite_moving_scen_stationary.npz'},
        {'label': 'Moving 0°', 'file': 'sar_satellite_moving_scen_0deg.npz'},
        {'label': 'Moving 45°', 'file': 'sar_satellite_moving_scen_45deg.npz'},
        {'label': 'Moving 90°', 'file': 'sar_satellite_moving_scen_90deg.npz'},
        {'label': 'Moving 135°', 'file': 'sar_satellite_moving_scen_135deg.npz'},
    ]
    
    current_scen_idx = 0
    data = None
    steps = []
    
    fig = plt.figure(figsize=(16, 9))
    
    # UI State
    use_db = True
    is_geo_mode = False
    img_display = None
    ui_refs = {}
    shared_zoom = None  # Single shared zoom for all scenarios

    def load_scenario(idx):
        nonlocal data, steps, current_scen_idx
        fname = scenarios[idx]['file']
        if not os.path.exists(fname):
            print(f"File not found: {fname}")
            return False
            
        print(f"Loading {fname}...")
        try:
            data = np.load(fname)
            current_scen_idx = idx
            
            steps[:] = []
            steps.append({
                'data': data['final_image'], 
                'title': f"Satellite SAR: {scenarios[idx]['label']}", 
                'xlabel': 'Range (m)', 
                'ylabel': 'Cross Range (m)',
                'extent': [data['range_axis'][0], data['range_axis'][-1], data['cross_range'][0], data['cross_range'][-1]]
            })
            
            return True
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return False

    def plot_geometry(ax_3d):
        if data is None: return
        
        R_e = 6371000.0
        h = data['orbit_alt']
        look = data['look_ang']
        inc_ang = data['inc_ang']
        
        gamma = np.radians(inc_ang - look)
        Rs = R_e + h
        
        # Ship Motion
        s_speed = data['ship_speed']
        s_head = data['ship_heading']
        s_vel = data['ship_vel']
        
        # Ground
        u = np.linspace(-2000, 2000, 20)
        v = np.linspace(-2000, 2000, 20)
        X, Y = np.meshgrid(u, v)
        Z = np.zeros_like(X)
        ax_3d.plot_wireframe(X, Y, Z, color='green', alpha=0.3, label='Sea Surface')
        
        # Target (Destroyer) at 0,0,0
        ax_3d.scatter([0], [0], [0], color='red', s=100, label='Destroyer')
        
        # Velocity Vector for Ship
        ax_3d.quiver(0, 0, 0, s_vel[0], s_vel[1], s_vel[2], length=500.0, color='red', linewidth=3, arrow_length_ratio=0.2, label='Ship Velocity')
        
        # Satellite Position (simplified - just show direction)
        sy = -Rs * np.sin(gamma)
        sz = Rs * np.cos(gamma) - R_e
        
        # Scale satellite position for visualization
        sat_scale = 5000.0 / np.sqrt(sy**2 + sz**2)
        ax_3d.scatter([0], [sy * sat_scale], [sz * sat_scale], color='blue', s=100, label='Satellite (scaled)')
        
        info = (f"Scenario: {scenarios[current_scen_idx]['label']}\n"
                f"Ship Speed: {s_speed:.1f} m/s\n"
                f"Heading: {s_head:.0f}°\n"
                f"Orbit Alt: {h/1000:.0f} km\n"
                f"Look Angle: {look:.1f}°")
        ax_3d.text2D(0.05, 0.95, info, transform=ax_3d.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        ax_3d.legend()
        ax_3d.set_title(f"Geometry: {scenarios[current_scen_idx]['label']}")
        
        limit = 3000.0
        ax_3d.set_xlim(-limit, limit)
        ax_3d.set_ylim(-limit, limit)
        ax_3d.set_zlim(0, limit)

    def get_visible_stats(data_mat, extent, xlim, ylim):
        img_data = data_mat
        rows, cols = img_data.shape
        if extent is None:
            x_min, x_max, y_min, y_max = 0, cols, 0, rows
        else:
            x_min, x_max, y_min, y_max = extent
            
        x0, x1 = xlim
        y0, y1 = ylim
        x0_Use = max(min(x0, x1), x_min)
        x1_Use = min(max(x0, x1), x_max)
        y0_Use = max(min(y0, y1), y_min)
        y1_Use = min(max(y0, y1), y_max)
        
        if (x1_Use <= x0_Use) or (y1_Use <= y0_Use): return np.array([0])
        
        c0 = int((x0_Use - x_min) / (x_max - x_min + 1e-9) * cols)
        c1 = int((x1_Use - x_min) / (x_max - x_min + 1e-9) * cols)
        r0 = int((y0_Use - y_min) / (y_max - y_min + 1e-9) * rows)
        r1 = int((y1_Use - y_min) / (y_max - y_min + 1e-9) * rows)
        
        c0 = max(0, min(cols, c0))
        c1 = max(0, min(cols, c1))
        r0 = max(0, min(rows, r0))
        r1 = max(0, min(rows, r1))
        
        c_start, c_end = sorted([c0, c1])
        r_start, r_end = sorted([r0, r1])
        if c_end <= c_start: c_end = c_start + 1
        if r_end <= r_start: r_end = r_start + 1
            
        return img_data[r_start:r_end, c_start:c_end]

    def update_clim(event=None):
        nonlocal img_display, shared_zoom
        if img_display is None or is_geo_mode: return
        try:
            ax = plt.gca()
            if event is not None:
                # Save zoom limits to shared state
                shared_zoom = (ax.get_xlim(), ax.get_ylim())
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            step_data = steps[0]['data']
            # Use extent directly (Fixed to match Destroyer viewer)
            ext = steps[0]['extent']
            subset = get_visible_stats(step_data, ext, xlim, ylim)
            
            if subset.size == 0: return

            if use_db:
                val_data = 20 * np.log10(np.abs(subset) + 1e-12)
                vmin = np.percentile(val_data, 1)
                vmax = np.percentile(val_data, 99)
            else:
                val_data = np.abs(subset)
                vmin = 0
                vmax = np.max(val_data)
            img_display.set_clim(vmin, vmax)
        except Exception: pass

    def update_view():
        nonlocal img_display
        
        if is_geo_mode:
             fig.delaxes(plt.gca())
             ax = fig.add_axes([0.25, 0.1, 0.7, 0.8], projection='3d')
             plot_geometry(ax)
        else:
            if len(fig.axes) > 0:
                 for ax in fig.axes:
                     if ax not in ui_refs.values():
                         fig.delaxes(ax)
            
            ax = fig.add_axes([0.25, 0.1, 0.7, 0.8])
            
            if not steps:
                ax.text(0.5, 0.5, "No Data Loaded", ha='center')
                fig.canvas.draw_idle()
                return

            step = steps[0]
            data_mat = step['data']
            
            # No transpose - data is already in correct orientation from sim
            plot_src = data_mat
            
            if use_db:
                plot_data = 20 * np.log10(np.abs(plot_src) + 1e-12)
                lbl = 'dB'
                cmap = 'jet'
                ui_refs['btn_log_obj'].label.set_text("Scale: dB")
            else:
                plot_data = np.abs(plot_src)
                lbl = 'Linear'
                cmap = 'gray'
                ui_refs['btn_log_obj'].label.set_text("Scale: Linear")
            
            # Fixed: Use extent directly (No Swap)
            ext = step['extent']
            
            img_display = ax.imshow(plot_data, aspect='auto', cmap=cmap, origin='lower',
                                    extent=ext)
            
            ax.set_title(step['title'])
            ax.set_xlabel(step['xlabel'])
            ax.set_ylabel(step['ylabel'])
            
            # Restore shared zoom limits across all scenarios
            if shared_zoom is not None:
                lx, ly = shared_zoom
                ax.set_xlim(lx)
                ax.set_ylim(ly)
                
            update_clim()
            ax.callbacks.connect('xlim_changed', update_clim)
            ax.callbacks.connect('ylim_changed', update_clim)
            
        fig.canvas.draw_idle()

    # Callbacks
    def on_scen_clicked(label):
        for i, s in enumerate(scenarios):
            if s['label'] == label:
                if load_scenario(i):
                    update_view()
                break
        
    def toggle_log(event):
        nonlocal use_db
        use_db = not use_db
        if not is_geo_mode: update_view()
        
    def toggle_geo(event):
        nonlocal is_geo_mode
        is_geo_mode = not is_geo_mode
        update_view()

    # Initialize UI
    rax = plt.axes([0.02, 0.6, 0.18, 0.3], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(rax, [s['label'] for s in scenarios])
    radio.on_clicked(on_scen_clicked)
    ui_refs[rax] = rax 
    
    ax_log = plt.axes([0.02, 0.5, 0.18, 0.08])
    btn_log = Button(ax_log, 'Scale: dB')
    btn_log.on_clicked(toggle_log)
    ui_refs[ax_log] = ax_log
    ui_refs['btn_log_obj'] = btn_log
    
    ax_geo = plt.axes([0.02, 0.4, 0.18, 0.08])
    btn_geo = Button(ax_geo, 'Geometry')
    btn_geo.on_clicked(toggle_geo)
    ui_refs[ax_geo] = ax_geo

    # Load default (Stationary)
    print("Loading default scenario...")
    if load_scenario(0):
        update_view()
    else:
        print("Waiting for simulation to create files...")
    
    print("Viewer Initialized.")
    plt.show()

if __name__ == "__main__":
    main()
