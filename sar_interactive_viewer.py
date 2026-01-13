import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    print("Launching Enhanced SAR Interactive Viewer...")
    filename = 'sar_simulation_data.npz'
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found. Run sar_vehicle_sim.py first.")
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
        {'data': data['rd_az_comp'], 'title': 'Step 4: Azimuth Compressed (Freq)', 'xlabel': 'Range (m)', 'ylabel': 'Doppler (Hz)',
         'extent': [data['range_axis'][0], data['range_axis'][-1], data['doppler_axis'][0], data['doppler_axis'][-1]]},
        {'data': data['final_image'], 'title': 'Step 5: Focused SAR Image', 'xlabel': 'Range (m)', 'ylabel': 'Cross Range (m)',
         'extent': [data['range_axis'][0], data['range_axis'][-1], data['cross_range'][0], data['cross_range'][-1]]}
    ]
    
    # Check for geometry data
    has_geo = 'platform_alt' in data.files
    
    fig = plt.figure(figsize=(14, 9))
    
    # State
    current_idx = 5 
    use_db = True 
    show_phase = False
    img_display = None
    is_geo_mode = False
    
    # Zoom Persistence
    zoom_limits = {} 
    
    # UI Refs
    ui_refs = {}

    def plot_geometry(ax_3d):
        # Platform Geometry (Flat Earth / Straight Line)
        h = data['platform_alt']
        vel = data['platform_vel']
        look = data['look_ang']
        r0 = data['r0']
        
        # Calculate Duration from Data
        # We need num_pulses and PRF
        # raw_phist shape is (FastTime, SlowTime) due to no-transpose save in vehicle sim?
        # Sim saves: raw_phist=raw_data.T (lines 232, 246). 
        # So shape is (pulses, samples) or (samples, pulses)? 
        # Sim: raw_data shape (pulses, samples).
        # phist_in = raw_data.T -> (samples, pulses).
        # So shape[1] is num_pulses.
        
        num_pulses = data['raw_phist'].shape[1]
        prf = data['prf'] if 'prf' in data.files else 2000.0
        T_int = num_pulses / prf
        
        # Scene Box
        extent = max(5000.0, vel * T_int / 1.5)
        
        # Ground Plane (Flat)
        u = np.linspace(-extent, extent, 20)
        v = np.linspace(-extent, extent, 20)
        X, Y = np.meshgrid(u, v)
        Z = np.zeros_like(X)
        
        ax_3d.plot_wireframe(X, Y, Z, color='green', alpha=0.3, label='Ground')
        ax_3d.scatter([0], [0], [0], color='red', s=50, label='Target (Destroyer)')
        
        # Platform Position (at center of aperture)
        py = 0 # Center of aperture
        px = -r0 * np.sin(np.radians(look))
        pz = r0 * np.cos(np.radians(look))
        
        ax_3d.scatter([px], [py], [pz], color='blue', s=100, label='Platform')
        
        # Flight Path (Full Synthetic Aperture)
        t_track = np.linspace(-T_int/2, T_int/2, 100)
        path_y = py + vel * t_track
        path_x = np.full_like(path_y, px)
        path_z = np.full_like(path_y, pz)
        
        ax_3d.plot(path_x, path_y, path_z, color='magenta', linewidth=3, linestyle='-', label='Flight Path')
        
        # Beam Cone / Aperture Pattern
        # Apex at (px, py, pz). 
        # Center of beam points to target (0,0,0)
        V_pt = np.array([-px, -py, -pz])
        Len_pt = np.linalg.norm(V_pt)
        Dir_pt = V_pt / Len_pt
        
        # Visualize Beamwidth roughly
        # If aperture L ~ V*T_int ? No T_int is huge synthetic aperture.
        # Real beamwidth is small.
        # But we want to see the "pattern". 
        # Let's show the ground spot illuminated by the beam.
        
        # Cone Radius at ground
        # 3dB Beamwidth approx 3 degrees -> 0.05 rad.
        # R * 0.05 ~ 1500m at 30km. Values seem reasonable.
        Cone_R = r0 * np.radians(3.0) 
        
        n1 = np.array([0.0, 1.0, 0.0])
        n2 = np.cross(Dir_pt, n1)
        
        uc = np.linspace(0, 1, 10)
        vc = np.linspace(0, 2*np.pi, 20)
        UC, VC = np.meshgrid(uc, vc)
        
        XC = px + UC * (n1[0] * Cone_R * np.cos(VC) + n2[0] * Cone_R * np.sin(VC) + Dir_pt[0] * Len_pt)
        YC = py + UC * (n1[1] * Cone_R * np.cos(VC) + n2[1] * Cone_R * np.sin(VC) + Dir_pt[1] * Len_pt)
        ZC = pz + UC * (n1[2] * Cone_R * np.cos(VC) + n2[2] * Cone_R * np.sin(VC) + Dir_pt[2] * Len_pt)
        
        ax_3d.plot_surface(XC, YC, ZC, color='yellow', alpha=0.2, shade=False)
        
        # Ground Spot
        th_spot = np.linspace(0, 2*np.pi, 20)
        r_spot = np.linspace(0, Cone_R, 5)
        TH, R_S = np.meshgrid(th_spot, r_spot)
        X_S = R_S * np.cos(TH)
        Y_S = R_S * np.sin(TH)
        Z_S = np.zeros_like(X_S) 
        
        ax_3d.plot_surface(X_S, Y_S, Z_S, color='red', alpha=0.5, zorder=10)
        
        info = (f"Altitude: {h/1000:.1f} km\n"
                f"Velocity: {vel:.1f} m/s\n"
                f"Look Angle: {look:.1f}°\n"
                f"Slant Range: {r0/1000:.1f} km\n"
                f"Integration: {T_int:.1f} s")
        ax_3d.text2D(0.05, 0.95, info, transform=ax_3d.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.7))
        
        ax_3d.legend()
        ax_3d.set_title("Scene Geometry (Straight Line Flight)")
        
        # Auto-scale
        max_coord = max(abs(px), abs(py), pz) * 1.1
        limit = max(max_coord, vel * T_int / 1.5) 
        
        ax_3d.set_xlim(-limit, limit)
        ax_3d.set_ylim(-limit, limit)
        ax_3d.set_zlim(0, limit)

    def get_visible_stats(data_mat, extent, xlim, ylim):
        img_data = data_mat # Originals were not transposed in this viewer logic
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
            if event is not None:
                zoom_limits[current_idx] = (ax.get_xlim(), ax.get_ylim())
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            step = steps[current_idx]
            subset = get_visible_stats(step['data'], step['extent'], xlim, ylim)
            
            if subset.size == 0: return

            if show_phase:
                 # Fixed limits for phase
                 img_display.set_clim(-np.pi, np.pi)
                 return

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
            
            step = steps[current_idx]
            data_mat = step['data']
            
            # Original Logic: plot_data.T was used in previous viewer?
            # Let's check previous file content or assume Standard logic.
            # In old viewer: `img_display = ax.imshow(plot_data.T, ...)`
            # So we keep that.
            
            if show_phase:
                # Phase Visualization
                plot_data = np.angle(data_mat)
                lbl = 'Phase (rad)'
                cmap = 'twilight'
                vmin, vmax = -np.pi, np.pi
                ui_refs['btn_phase_obj'].label.set_text("View: Phase")
            elif use_db:
                # Magnitude dB
                plot_data = 20 * np.log10(np.abs(data_mat) + 1e-12)
                lbl = 'dB'
                cmap = 'jet'
                ui_refs['btn_log_obj'].label.set_text("Scale: dB")
                ui_refs['btn_phase_obj'].label.set_text("View: Mag")
                # Reset vmin/vmax for auto-scaling or reuse saved zoom logic if we want, 
                # but explicit clim usually handled by update_clim if not set here.
                # Let's let update_clim handle vmin/vmax for magnitude
                vmin, vmax = None, None
            else:
                # Magnitude Linear
                plot_data = np.abs(data_mat)
                lbl = 'Linear'
                cmap = 'gray'
                ui_refs['btn_log_obj'].label.set_text("Scale: Linear")
                ui_refs['btn_phase_obj'].label.set_text("View: Mag")
                vmin, vmax = None, None
            
            # Steps 0-4 need transpose, step 5 (final image) does not
            if current_idx < 5:
                img_display = ax.imshow(plot_data.T, aspect='auto', cmap=cmap, origin='lower',
                                        extent=step['extent'])
            else:
                img_display = ax.imshow(plot_data, aspect='auto', cmap=cmap, origin='lower',
                                        extent=step['extent'])
            
            # Set vmin/vmax for phase immediately
            if show_phase:
                img_display.set_clim(vmin, vmax)
            
            ax.set_title(step['title'])
            ax.set_xlabel(step['xlabel'])
            ax.set_ylabel(step['ylabel'])
            
            if current_idx in zoom_limits:
                lx, ly = zoom_limits[current_idx]
                ax.set_xlim(lx)
                ax.set_ylim(ly)
                
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

    def toggle_phase(event):
        nonlocal show_phase
        show_phase = not show_phase
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
    
    ax_log = plt.axes([0.02, 0.5, 0.18, 0.08])
    btn_log = Button(ax_log, 'Scale: dB')
    btn_log.on_clicked(toggle_log)
    ui_refs[ax_log] = ax_log
    ui_refs[ax_log] = ax_log
    ui_refs['btn_log_obj'] = btn_log
    
    ax_phase = plt.axes([0.02, 0.44, 0.18, 0.05]) # Reduced height slightly to fit
    btn_phase = Button(ax_phase, 'View: Mag')
    btn_phase.on_clicked(toggle_phase)
    ui_refs[ax_phase] = ax_phase
    ui_refs['btn_phase_obj'] = btn_phase
    
    if has_geo:
        ax_geo = plt.axes([0.02, 0.35, 0.18, 0.08]) # Moved down
        btn_geo = Button(ax_geo, 'Scene Geometry')
        btn_geo.on_clicked(toggle_geo)
        ui_refs[ax_geo] = ax_geo

    # Sync Radio Sate
    radio.set_active(current_idx)

    # Initial Render
    update_view()
    
    print("Enhanced Plane Viewer launched.")
    plt.show()

if __name__ == "__main__":
    main()
