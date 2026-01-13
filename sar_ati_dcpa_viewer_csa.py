"""
SAR ATI/DPCA Interactive Viewer - CSA Version
Features: 
- Loads CSA processed data (sar_ati_dpca_data_csa.npz)
- Stable layout (fixed colorbar axis)
- Dynamic stats on zoom/pan
- Auto-adjusting dynamic range (clim) on zoom
- Phase Balancing calibration
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
import os

def main():
    print("Launching SAR ATI/DPCA Viewer (CSA)...")
    
    # --- 1. Load Data ---
    fname = "sar_ati_dpca_data_csa.npz"
    if not os.path.exists(fname):
        print(f"Data file {fname} not found. Please run sar_ati_dcpa_sim_csa.py first.")
        return

    print("Loading data...")
    data = np.load(fname)
    slc1 = data['slc1'].T
    slc2 = data['slc2'].T
    rax = data['range_axis']
    cax = data['cross_range']
    
    # Extent: [Range Min, Range Max, Cross-Range Min, Cross-Range Max]
    extent = [rax[0], rax[-1], cax[0], cax[-1]]
    
    # --- 2. Derived Products Storage ---
    class SARData:
        def __init__(self, s1, s2):
            self.s1 = s1
            self.s2 = s2
            self.cal_phase = 0.0
            self.compute_all()
            
        def compute_all(self):
            s2_cal = self.s2 * np.exp(1j * self.cal_phase)
            self.prods = {
                'Ch1 Magnitude': np.abs(self.s1),
                'Ch1 Phase': np.angle(self.s1),
                'Ch2 Magnitude': np.abs(s2_cal),
                'Ch2 Phase': np.angle(s2_cal),
                'DPCA Magnitude': np.abs(self.s1 - s2_cal),
                'DPCA Phase': np.angle(self.s1 - s2_cal),
                'ATI Phase': np.angle(self.s1 * np.conj(s2_cal))
            }
            
        def get(self, mode):
            return self.prods.get(mode)

    sar = SARData(slc1, slc2)

    # --- 3. UI State ---
    state = {
        'mode': 'Ch1 Magnitude',
        'scale': 'dB',
        'mask_threshold': 0.0,
        'im_handle': None,
        'cbar_handle': None,
        'zoom_stats_enabled': True
    }

    # --- 4. Main Figure Setup ---
    # Using 'Qt5Agg' or similar backend usually handled by env, 
    # but let's stick to standard plt.figure()
    fig = plt.figure(figsize=(15, 9))
    plt.subplots_adjust(left=0.25, bottom=0.15, right=0.85) # Fixed layout spacing

    ax_main = fig.add_axes([0.25, 0.15, 0.60, 0.75])
    ax_cbar = fig.add_axes([0.88, 0.15, 0.02, 0.75]) # Dedicated colorbar slot

    # --- 5. Statistics Function ---
    def print_visible_stats(ax):
        if not state['zoom_stats_enabled']: return
        
        # Get visible data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Map axes to indices
        # rax is Range Axis.
        # cax is Cross Range Axis.
        # Check directions for indices
        
        r_mask = (rax >= min(xlim)) & (rax <= max(xlim))
        c_mask = (cax >= min(ylim)) & (cax <= max(ylim))
        
        r_indices = np.where(r_mask)[0]
        c_indices = np.where(c_mask)[0]
        
        if len(r_indices) == 0 or len(c_indices) == 0:
            return
            
        # Slice current mode data
        # Data shape is (N_cross, N_range) usually?
        # Check how we load slc1. 
        # In sim, we save sim result which is (N_range, N_cross) transposed?
        # The viewer displayed correctly before.
        # Let's trust shape follows (cax, rax).
        
        raw_data = sar.get(state['mode'])
        
        # Slice: [rows(cross_range), cols(range)]
        # We need to be careful with meshgrid vs imshow origin.
        # origin='lower' means index 0 is at bottom.
        
        visible_raw = raw_data[np.ix_(c_indices, r_indices)]
        
        print(f"\n--- Visible Stats: {state['mode']} ---")
        print(f"Region: Range [{min(xlim):.1f}, {max(xlim):.1f}], Cross-Range [{min(ylim):.1f}, {max(ylim):.1f}]")
        
        if 'Phase' in state['mode']:
            # Phase stats
            print(f"Mean: {np.mean(visible_raw):.4f} rad")
            print(f"Median: {np.median(visible_raw):.4f} rad")
            print(f"Std: {np.std(visible_raw):.4f} rad")
            print(f"Range: [{np.min(visible_raw):.4f}, {np.max(visible_raw):.4f}]")
        else:
            # Mag stats
            if state['scale'] == 'dB':
                data = 20 * np.log10(visible_raw + 1e-12)
                unit = "dB"
            else:
                data = visible_raw
                unit = "Units"
                
            print(f"Mean: {np.mean(data):.2f} {unit}")
            print(f"Median: {np.median(data):.2f} {unit}")
            print(f"Std: {np.std(data):.2f} {unit}")
            print(f"Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
            
            # Special DPCA cancellation check
            if 'DPCA' in state['mode']:
                ref_mag = sar.get('Ch1 Magnitude')[np.ix_(c_indices, r_indices)]
                # Ratio of energies
                ratio = np.mean(ref_mag) / (np.mean(visible_raw) + 1e-9)
                print(f"Local Cancellation Ratio: {ratio:.2f}")

        # Update CLIM based on visible region to maximize contrast
        if state['im_handle']:
            if 'Phase' in state['mode']:
                vmin, vmax = -np.pi, np.pi
            else:
                vmax = np.percentile(data, 99.9)
                # Ensure decent dynamic range display
                vmin = vmax - 60 if state['scale'] == 'dB' else 0
            state['im_handle'].set_clim(vmin, vmax)
            fig.canvas.draw_idle()

    # Callback for zoom/pan
    def on_lim_change(event_ax):
        print_visible_stats(event_ax)

    ax_main.callbacks.connect('xlim_changed', on_lim_change)
    ax_main.callbacks.connect('ylim_changed', on_lim_change)

    # --- 6. Plot Updating ---
    def update_plot():
        mode = state['mode']
        scale = state['scale']
        mask_val = state['mask_threshold']
        
        data_raw = sar.get(mode)
        
        if 'Phase' in mode:
            # Masking logic
            display_data = np.copy(data_raw)
            if mask_val > 0:
                # Use Ch1 Mag for masking
                ref_mag = sar.get('Ch1 Magnitude')
                mask = ref_mag < (np.max(ref_mag) * mask_val)
                display_data[mask] = 0
            
            vmin, vmax = -np.pi, np.pi
            cmap = 'hsv'
            lbl = f"{mode} (rad)"
        else:
            if scale == 'dB':
                display_data = 20 * np.log10(data_raw + 1e-12)
                vmax = np.percentile(display_data, 99.9)
                vmin = vmax - 60
                cmap = 'magma' if 'DPCA' in mode else 'bone'
                lbl = f"{mode} (dB)"
            else:
                display_data = data_raw
                vmax = np.percentile(display_data, 99.9)
                vmin = 0
                cmap = 'magma' if 'DPCA' in mode else 'gray'
                lbl = f"{mode} (Linear)"

        if state['im_handle'] is None:
            state['im_handle'] = ax_main.imshow(display_data, aspect='auto', origin='lower', 
                                                extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            state['cbar_handle'] = fig.colorbar(state['im_handle'], cax=ax_cbar)
        else:
            state['im_handle'].set_data(display_data)
            state['im_handle'].set_cmap(cmap)
            state['im_handle'].set_clim(vmin, vmax)
        
        state['cbar_handle'].set_label(lbl)
        ax_main.set_title(f"Focus (CSA): {mode}")
        ax_main.set_xlabel("Range (m)")
        ax_main.set_ylabel("Cross-Range (m)")
        
        fig.canvas.draw_idle()
        # Trigger stats for initial view
        print_visible_stats(ax_main)

    # --- 7. Controls ---
    # Mode
    ax_radio_mode = fig.add_axes([0.02, 0.6, 0.18, 0.3], facecolor='#FFF8DC')
    modes = ['Ch1 Magnitude', 'Ch1 Phase', 'Ch2 Magnitude', 'Ch2 Phase', 'DPCA Magnitude', 'DPCA Phase', 'ATI Phase']
    radio_mode = RadioButtons(ax_radio_mode, modes)
    def set_mode(label):
        state['mode'] = label
        update_plot()
    radio_mode.on_clicked(set_mode)

    # Scale
    ax_radio_scale = fig.add_axes([0.02, 0.45, 0.18, 0.1], facecolor='#E0FFFF')
    radio_scale = RadioButtons(ax_radio_scale, ['dB', 'Linear'])
    def set_scale(label):
        state['scale'] = label
        update_plot()
    radio_scale.on_clicked(set_scale)

    # Mask
    ax_slider_mask = fig.add_axes([0.25, 0.05, 0.4, 0.03])
    slider_mask = Slider(ax_slider_mask, 'Phase Mask', 0.0, 0.5, valinit=0.0)
    def set_mask(val):
        state['mask_threshold'] = val
        if 'Phase' in state['mode']:
            update_plot()
    slider_mask.on_changed(set_mask)

    # Balance
    ax_btn_bal = fig.add_axes([0.7, 0.05, 0.1, 0.04])
    btn_bal = Button(ax_btn_bal, 'Auto-Balance')
    def do_balance(event):
        print("Calibrating Channel Phase...")
        # Calibration based on clutter (whole scene average)
        # Or better: average of complex product
        avg_interf = np.mean(slc1 * np.conj(slc2))
        sar.cal_phase = np.angle(avg_interf)
        print(f"Applied Offset: {np.degrees(sar.cal_phase):.3f} deg")
        sar.compute_all()
        update_plot()
    btn_bal.on_clicked(do_balance)

    # Reset Zoom
    ax_btn_reset = fig.add_axes([0.82, 0.05, 0.06, 0.04])
    btn_reset = Button(ax_btn_reset, 'Reset')
    def do_reset(event):
        ax_main.set_xlim(rax[0], rax[-1])
        ax_main.set_ylim(cax[0], cax[-1])
        update_plot()
    btn_reset.on_clicked(do_reset)

    update_plot()
    plt.show()

if __name__ == "__main__":
    main()
