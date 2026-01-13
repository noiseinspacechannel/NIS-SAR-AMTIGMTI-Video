import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    print("Launching Satellite SAR Viewer...")
    filename = 'sar_satellite_data.npz'
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run sar_satellite_sim.py first.")
        return
        
    print(f"Loading {filename}...")
    data = np.load(filename)

    # Extract processing data
    steps = [
        {'data': data['raw_phist'], 'title': 'Raw Phase History', 'xlabel': 'Fast Time', 'ylabel': 'Slow Time', 'extent': None},
        {'data': data['range_comp'], 'title': 'Step 1: Range Compressed', 'xlabel': 'Fast Time', 'ylabel': 'Slow Time', 'extent': None},
        {'data': data['rd_map'], 'title': 'Step 2: Range-Doppler Map', 'xlabel': 'Range (m)', 'ylabel': 'Doppler (Hz)', 
         'extent': [data['range_axis'][0], data['range_axis'][-1], data['doppler_axis'][0], data['doppler_axis'][-1]]},
        {'data': data['rd_rcmc'], 'title': 'Step 3: RCMC Corrected', 'xlabel': 'Range (m)', 'ylabel': 'Doppler (Hz)',
         'extent': [data['range_axis'][0], data['range_axis'][-1], data['doppler_axis'][0], data['doppler_axis'][-1]]},
        {'data': data['final_image'], 'title': 'Step 4: Focused SAR Image', 'xlabel': 'Range (m)', 'ylabel': 'Cross Range (m)',
         'extent': [data['range_axis'][0], data['range_axis'][-1], data['cross_range'][0], data['cross_range'][-1]]}
    ]
    # Note: final_image in steps extracted as 'final_image' directly.
    # We will handle transposition logic in update_view based on step index.
    
    # Check for geometry data
    has_geo = 'orbit_alt' in data.files
    
    fig = plt.figure(figsize=(14, 9))
    
    # State
    current_idx = 4 
    use_db = True 
    img_display = None
    is_geo_mode = False
    
    # Zoom Persistence
    zoom_limits = {} 
    
    # Placeholder for UI elements
    ui_refs = {}

    def plot_geometry(ax_3d):
        R_e = 6371000.0
        h = data['orbit_alt']
        look = data['look_ang']
        inc_ang = data['inc_ang']
        
        gamma = np.radians(inc_ang - look)
        Rs = R_e + h
        
        # Wireframe Earth Patch
        u = np.linspace(-50000, 50000, 20)
        v = np.linspace(-50000, 50000, 20)
        X, Y = np.meshgrid(u, v)
        Z = -(X**2 + Y**2) / (2*R_e)
        
        ax_3d.plot_wireframe(X, Y, Z, color='green', alpha=0.3, label='Earth Surface')
        ax_3d.scatter([0], [0], [0], color='red', s=50, label='Target (Destroyer)')
        
        sy = -Rs * np.sin(gamma)
        sz = Rs * np.cos(gamma) - R_e
        
        ax_3d.scatter([0], [sy], [sz], color='blue', s=100, label='Satellite')
        
        # 1. Beam Cone
        V_st = np.array([0, -sy, -sz])
        Len_st = np.linalg.norm(V_st)
        Dir_st = V_st / Len_st
        
        Cone_R = 5000.0 
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.cross(Dir_st, n1)
        
        uc = np.linspace(0, 1, 10)
        vc = np.linspace(0, 2*np.pi, 20)
        UC, VC = np.meshgrid(uc, vc)
        
        XC = 0 + UC * (n1[0] * Cone_R * np.cos(VC) + n2[0] * Cone_R * np.sin(VC) + Dir_st[0] * Len_st)
        YC = sy + UC * (n1[1] * Cone_R * np.cos(VC) + n2[1] * Cone_R * np.sin(VC) + Dir_st[1] * Len_st)
        ZC = sz + UC * (n1[2] * Cone_R * np.cos(VC) + n2[2] * Cone_R * np.sin(VC) + Dir_st[2] * Len_st)
        
        ax_3d.plot_surface(XC, YC, ZC, color='yellow', alpha=0.2, shade=False)
        
        # 2. Ground Spot Highlight
        th_spot = np.linspace(0, 2*np.pi, 20)
        r_spot = np.linspace(0, Cone_R, 5)
        TH, R_S = np.meshgrid(th_spot, r_spot)
        
        X_S = R_S * np.cos(TH)
        Y_S = R_S * np.sin(TH)
        Z_S = -(X_S**2 + Y_S**2) / (2*R_e)
        
        ax_3d.plot_surface(X_S, Y_S, Z_S, color='red', alpha=0.5, zorder=10)
        
        t_track = np.linspace(-20, 20, 50) 
        ox = t_track * 1000 
        oy = np.full_like(ox, sy)
        oz = np.full_like(ox, sz) 
        ax_3d.plot(ox, oy, oz, color='white', linewidth=2, label='Orbit Track')
        
        info = (f"Altitude: {h/1000:.1f} km\n"
                f"Velocity: {data['orbit_vel']:.0f} m/s\n"
                f"Look Angle: {look:.1f}°\n"
                f"Slant Range: {data['r0']/1000:.1f} km")
        ax_3d.text2D(0.05, 0.95, info, transform=ax_3d.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        ax_3d.legend()
        ax_3d.set_title("Orbital Geometry Scene")
        
        ax_3d.set_xlim(-50000, 50000)
        ax_3d.set_ylim(sy - 50000, 50000)
        ax_3d.set_zlim(0, sz + 50000)

    def get_visible_stats(data_mat, extent, xlim, ylim):
        # No image transformation - just use data as-is
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
        nonlocal img_display
        if img_display is None or is_geo_mode: return
        try:
            ax = plt.gca()
            # Save limits if this is triggered by zoom/pan
            if event is not None:
                zoom_limits[current_idx] = (ax.get_xlim(), ax.get_ylim())
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            step = steps[current_idx]
            # Use swapped extent to match imshow
            ext = step['extent']
            swapped_extent = [ext[2], ext[3], ext[0], ext[1]] if ext else None
            subset = get_visible_stats(step['data'], swapped_extent, xlim, ylim)
            
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
            # Clean up 3D
            fig.delaxes(plt.gca())
            ax = fig.add_axes([0.25, 0.1, 0.7, 0.8], projection='3d')
            plot_geometry(ax)
        else:
            # Clean up 2D
            if len(fig.axes) > 0:
                 for ax in fig.axes:
                     if ax not in ui_refs.values():
                         fig.delaxes(ax)
            
            ax = fig.add_axes([0.25, 0.1, 0.7, 0.8])
            
            step = steps[current_idx]
            data_mat = step['data']
            
            # No image transformation - use data as-is
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
            
            # Swap extent: original is [xmin, xmax, ymin, ymax], swap to [ymin, ymax, xmin, xmax]
            ext = step['extent']
            swapped_extent = [ext[2], ext[3], ext[0], ext[1]] if ext else None
            
            img_display = ax.imshow(plot_data, aspect='auto', cmap=cmap, origin='lower',
                                    extent=swapped_extent)
            
            ax.set_title(step['title'])
            # Use original labels (no swap needed)
            ax.set_xlabel(step['xlabel'])
            ax.set_ylabel(step['ylabel'])
            
            # Restore View Limits if saved
            if current_idx in zoom_limits:
                lx, ly = zoom_limits[current_idx]
                ax.set_xlim(lx)
                ax.set_ylim(ly)
            
            # Force update clim based on new view
            update_clim()
            ax.callbacks.connect('xlim_changed', update_clim)
            ax.callbacks.connect('ylim_changed', update_clim)
            
        fig.canvas.draw_idle()

    # Callbacks
    def on_radio_clicked(label):
        nonlocal current_idx, is_geo_mode
        for i, s in enumerate(steps):
            if s['title'] == label:
                current_idx = i
                break
        if is_geo_mode: is_geo_mode = False 
        update_view()
    
    def toggle_log(event):
        nonlocal use_db
        use_db = not use_db
        if not is_geo_mode: update_view()
        
    def toggle_geo(event):
        nonlocal is_geo_mode
        is_geo_mode = not is_geo_mode
        update_view()
        
    def on_keypress(event):
        if event.key == 'escape':
            try: fig.canvas.manager.full_screen_toggle()
            except: pass

    # Initialize UI
    rax = plt.axes([0.02, 0.6, 0.18, 0.25], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(rax, [s['title'] for s in steps])
    radio.on_clicked(on_radio_clicked)
    ui_refs[rax] = rax 
    
    # Sync Radio State to Default (Step 4)
    # radio.set_active(current_idx) # Moved to end to avoid crash 
    
    ax_log = plt.axes([0.02, 0.5, 0.18, 0.08])
    btn_log = Button(ax_log, 'Scale: dB')
    btn_log.on_clicked(toggle_log)
    ui_refs[ax_log] = ax_log
    ui_refs['btn_log_obj'] = btn_log
    
    if has_geo:
        ax_geo = plt.axes([0.02, 0.4, 0.18, 0.08])
        btn_geo = Button(ax_geo, 'Orbit Geometry')
        btn_geo.on_clicked(toggle_geo)
        ui_refs[ax_geo] = ax_geo

    # Sync Radio Sate - Call last to ensure all buttons exist
    radio.set_active(current_idx)

    # Initial Render
    update_view()
    
    print("Satellite Viewer launched.")
    plt.show()

if __name__ == "__main__":
    main()
