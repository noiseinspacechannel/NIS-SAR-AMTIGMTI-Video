"""
SAR Vehicle Simulation - Destroyer Target
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Redirect output to log file
sys.stdout = open('sim_vehicle_log.txt', 'w', buffering=1)
sys.stderr = sys.stdout
print("Starting SAR Vehicle Simulation...", flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.signal.windows import hamming
from vehicle_targets import generate_destroyer

# --- Constants ---
C = 299792458.0
Re = 6378137.0
Rs = Re + 20000.0   # 20 km Altitude
V_plat = 150.0      # 150 m/s Platform Velocity
FC = 10e9           # 10 GHz
Lambda = C / FC

print("--- Geometry Setup ---")
print(f"Altitude: {(Rs-Re)/1000:.1f} km")
print(f"Velocity: {V_plat:.1f} m/s")

# Geometry
theta_look_deg = 45.0
theta_look_rad = np.radians(theta_look_deg)
h = Rs - Re
R0 = h / np.cos(theta_look_rad)
print(f"Slant Range R0: {R0/1000.0:.2f} km")

# Smaller number of pulses for quick test
nominal_prp = 500e-6 # 2kHz PRF
tuned_prp = nominal_prp
num_pulses = 32768  # Full synthetic aperture size
T_int = num_pulses * tuned_prp
cross_range_extent = V_plat * T_int / 2

# Scene extent
scene_extent = 150.0  # ±150m

print(f"Integration Time: {T_int:.2f} s")
print(f"Num Pulses: {num_pulses}")
print(f"Cross-range extent: {cross_range_extent:.1f} m")
print(f"Scene extent: ±{scene_extent:.1f} m")

t_delay_center = 2 * R0 / C
print(f"PRP: {tuned_prp*1e6:.1f} us")

t_vec = np.linspace(-T_int/2, T_int/2, num_pulses)

# Trajectory
sat_x = -R0 * np.sin(theta_look_rad)
sat_z = R0 * np.cos(theta_look_rad)

pos = np.zeros((num_pulses, 3))
# vel = np.zeros((num_pulses, 3))

for i, t in enumerate(t_vec):
    pos[i, 0] = sat_x
    pos[i, 1] = V_plat * t
    pos[i, 2] = sat_z
    # vel[i, 1] = V_plat

# --- Vehicle Target Scene ---
print("\n--- Generating Destroyer Target ---")
sim_targets = generate_destroyer(center_pos=(0,0,0))
    
for t in sim_targets:
    # Print first few for verification log, don't spam
    pass
print(f"Total targets: {len(sim_targets)}")

# --- Physics Engine ---
def run_custom_physics(targets, t_vec, pos, tuned_prp, t_p, fc, bw):
    print("\n--- Starting Physics Engine ---")
    fs = 360e6
    num_samples = 2048
    
    fast_times = np.linspace(0, num_samples/fs, num_samples)
    t_start_fast = (2 * R0 / C) - (num_samples/fs)/2
    
    raw = np.zeros((len(t_vec), num_samples), dtype=complex)
    k_rate = bw / t_p
    
    num_targets = len(targets)
    print(f"Simulating {num_targets} targets...")
    
    t_pos = np.array([t['position'] for t in targets])
    t_rcs = np.array([t['rcs'] for t in targets])
    
    t_fast_abs = t_start_fast + fast_times
    
    for i in range(len(t_vec)):
        if i % 500 == 0: print(f"Pulse {i}/{len(t_vec)}", end='\r')
        
        sat_p = pos[i]
        pulse_resp = np.zeros(num_samples, dtype=complex)
        
        diff = t_pos - sat_p 
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)
        
        tau = 2 * dist / C
        phase_base = -4.0 * np.pi * fc * dist / C
        amp = np.sqrt(t_rcs)
        
        for b in range(num_targets):
            t_local = t_fast_abs - tau[b]
            mask = np.abs(t_local - t_p/2) <= t_p/2
            chirp_phase = np.pi * k_rate * ((t_local - t_p/2)**2)
            sig = amp[b] * np.exp(1j * (phase_base[b] + chirp_phase)) * mask
            pulse_resp += sig
            
        raw[i, :] = pulse_resp
        
    print("\nPhysics Engine Complete.")
    return raw

# --- Radar System Parameters (Airborne Platform) ---
P_TX = 2000.0           # Transmit power (W) - typical airborne SAR
ANT_LENGTH = 1.5        # Antenna length (m) - azimuth
ANT_WIDTH = 0.3         # Antenna width (m) - elevation
T_SYS = 290.0           # System temperature (K)
NF_DB = 4.0             # Noise figure (dB)
LOSS_DB = 3.0           # System losses (dB)
K_BOLTZ = 1.380649e-23

SCR_DB = 10.0           # Signal-to-Clutter Ratio (dB)
K_NU = 1.0              # K-distribution shape

def calculate_snr_db(r_slant, rcs, wavelength, bandwidth, t_int, p_tx=P_TX, 
                     ant_l=ANT_LENGTH, ant_w=ANT_WIDTH, t_sys=T_SYS, nf_db=NF_DB, loss_db=LOSS_DB):
    ant_area = ant_l * ant_w * 0.6
    gain = 4 * np.pi * ant_area / (wavelength ** 2)
    gain_db = 10 * np.log10(gain)
    nf = 10 ** (nf_db / 10)
    loss = 10 ** (loss_db / 10)
    numerator = p_tx * (gain ** 2) * (wavelength ** 2) * rcs * t_int
    denominator = ((4 * np.pi) ** 3) * (r_slant ** 4) * K_BOLTZ * t_sys * bandwidth * loss * nf
    snr_db = 10 * np.log10(numerator / denominator)
    return snr_db, gain_db

def add_ocean_noise(raw_data, snr_db, scr_db=SCR_DB, k_nu=K_NU):
    print(f"Adding noise: SNR={snr_db:.1f}dB (calculated), SCR={scr_db}dB, K_nu={k_nu}")
    signal_power = np.mean(np.abs(raw_data)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    thermal_noise = np.sqrt(noise_power/2) * (
        np.random.randn(*raw_data.shape) + 1j*np.random.randn(*raw_data.shape))
    clutter_power = signal_power / (10 ** (scr_db / 10))
    texture = np.random.gamma(k_nu, 1/k_nu, raw_data.shape)
    speckle = np.random.exponential(1, raw_data.shape)
    k_intensity = clutter_power * texture * speckle
    clutter_phase = np.random.uniform(0, 2*np.pi, raw_data.shape)
    sea_clutter = np.sqrt(k_intensity) * np.exp(1j * clutter_phase)
    print(f"  Signal: {10*np.log10(signal_power):.1f}dB, Noise: {10*np.log10(noise_power):.1f}dB, Clutter: {10*np.log10(clutter_power):.1f}dB")
    return raw_data + thermal_noise + sea_clutter

# --- Radar Parameters ---
f_c = 10e9
bw = 300e6
t_p = 1.0e-6

# --- Run Simulation ---
raw_data = run_custom_physics(sim_targets, t_vec, pos, tuned_prp, t_p, f_c, bw)

# Calculate SNR from radar equation and add noise
avg_rcs = 50000.0  # m² - destroyer
snr_db, gain_db = calculate_snr_db(R0, avg_rcs, Lambda, bw, T_int)
print(f"Radar Equation: Gain={gain_db:.1f}dB, SNR={snr_db:.1f}dB")
raw_data = add_ocean_noise(raw_data, snr_db)

# --- RDA Processing ---
def sar_focus_rda(phist, center_wavelength_m, pulse_width_sec, chirp_rate_hzpsec, sample_rate_hz, prf_hz, platform_speed_mps, range_grp_m):
    print("\n--- Starting RDA Processing ---")
    
    speed_of_light_mps = 299792458
    num_ranges, num_pulses = phist.shape
    center_frequency_hz = speed_of_light_mps / center_wavelength_m

    if num_pulses % 2 == 0:
        slow_time_sec = (np.arange(num_pulses) - num_pulses / 2) / prf_hz
    else:
        slow_time_sec = (np.arange(num_pulses) - (num_pulses - 1) / 2) / prf_hz
    
    time_grp_sec = 2 * range_grp_m / speed_of_light_mps

    if num_ranges % 2 == 0:
        fast_time_sec = (np.arange(num_ranges) - num_ranges / 2) / sample_rate_hz + time_grp_sec
    else:
        fast_time_sec = (np.arange(num_ranges) - (num_ranges - 1) / 2) / sample_rate_hz + time_grp_sec

    # Range Compression
    print("Step 1: Range Compression")
    fast_time_step_sec = 1 / sample_rate_hz
    num_MF_samples = int(np.floor(pulse_width_sec / fast_time_step_sec)) + 1
    fast_time_MF = np.linspace(-pulse_width_sec/2, pulse_width_sec/2, num_MF_samples)
    transmitted_pulse = np.exp(1j * np.pi * chirp_rate_hzpsec * (fast_time_MF)**2)
    matched_filter = np.conj(transmitted_pulse)
    window_func = hamming(len(matched_filter))
    matched_filter = matched_filter * window_func
    matched_filter = matched_filter / np.linalg.norm(matched_filter)

    phist_compressed = np.zeros_like(phist, dtype=complex)
    for ii in range(num_pulses):
        received_signal = phist[:, ii]
        compressed_signal = convolve(received_signal, matched_filter, mode='same')
        phist_compressed[:, ii] = compressed_signal

    fast_time_compressed = fast_time_sec

    # Range-Doppler Transform
    print("Step 2: Range-Doppler Transform")
    win_az = hamming(num_pulses)
    phist_compressed_windowed = phist_compressed * win_az 
    range_doppler = np.fft.fftshift(np.fft.fft(np.fft.fftshift(phist_compressed_windowed, axes=1), axis=1), axes=1)

    PRF = prf_hz
    if num_pulses % 2 == 0:
        doppler_freq = np.arange(-num_pulses/2, num_pulses/2) * (PRF / num_pulses)
    else:
        doppler_freq = np.arange(-(num_pulses - 1)/2, (num_pulses - 1)/2 + 1) * (PRF / num_pulses)
    
    range_axis_m = fast_time_compressed * speed_of_light_mps / 2

    # RCMC - Fixed Direction (SUBTRACT)
    print("Step 3: RCMC (Fixed - Subtract)")
    lambd = speed_of_light_mps / center_frequency_hz
    V_r = platform_speed_mps

    delta_R_matrix = (range_axis_m[:, None] * (doppler_freq[None, :]**2) * lambd**2) / (8 * V_r**2)
    
    range_doppler_rcmc = np.zeros_like(range_doppler, dtype=complex)
    
    for k in range(num_pulses):
        delta_R = delta_R_matrix[:, k]
        shifted_range_axis = range_axis_m - delta_R  # Subtract to correct migration
        range_profile = range_doppler[:, k]
        
        if len(shifted_range_axis) > 1:
            f = interp1d(shifted_range_axis, range_profile, kind='linear', fill_value=0, bounds_error=False)
            range_profile_corrected = f(range_axis_m)
            range_doppler_rcmc[:, k] = range_profile_corrected
        else:
             range_doppler_rcmc[:, k] = range_profile

    # Azimuth Compression
    print("Step 4: Azimuth Compression")
    Ka = (2 * V_r**2) / (lambd * range_axis_m)
    inv_Ka = 1.0 / Ka
    H_matrix = np.exp(-1j * np.pi * (inv_Ka[:, None] * doppler_freq[None, :]**2))

    range_doppler_filtered = range_doppler_rcmc * H_matrix

    print("Step 5: Azimuth IFFT -> Image")
    sar_image = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(range_doppler_filtered, axes=1), axis=1), axes=1)
    sar_image_mag = np.abs(sar_image)

    cross_range_m = V_r * slow_time_sec
    range_midpoint = np.mean(range_axis_m)
    range_axis_centered = range_axis_m - range_midpoint

    sar_image_mag_transposed = sar_image_mag.T

    return (sar_image_mag_transposed, range_axis_centered, cross_range_m, 
            phist_compressed, range_doppler, range_doppler_rcmc, range_doppler_filtered, doppler_freq)

# Execute RDA
phist_in = raw_data.T 
prf_val = 1.0 / tuned_prp
lam_val = C / f_c
chirp_rate = bw / t_p
fs_val = 360e6

(sar_image, range_axis, cross_range, 
 phist_comp, rd_map, rd_rcmc, rd_az_comp, doppler_axis) = sar_focus_rda(
    phist_in, lam_val, t_p, chirp_rate, fs_val, prf_val, V_plat, R0
)

# Save arrays for interactive viewer
print("\nSaving data to sar_simulation_data.npz...")
np.savez('sar_simulation_data.npz', 
         raw_phist=phist_in,
         range_comp=phist_comp,
         rd_map=rd_map,
         rd_rcmc=rd_rcmc,
         rd_az_comp=rd_az_comp,
         final_image=sar_image,
         range_axis=range_axis,
         cross_range=cross_range,
         doppler_axis=doppler_axis,
         # Geometry Info for Viewer
         platform_alt=h,
         platform_vel=V_plat,
         look_ang=theta_look_deg,
         inc_ang=theta_look_deg, # Approx same for airborne flat-ish scene
         r0=R0,
         prf=prf_val)
print("Data saved.")

# Plotting (Standard outputs)
def plot_step(data_mat, title, fname, axis_x=None, axis_y=None, 
              label_x="Range (m)", label_y="Cross Range (m)", 
              xlim=None, ylim=None, do_log=True, cmap='jet'):
    plt.figure(figsize=(10, 8))
    
    if do_log:
        plot_data = 20 * np.log10(np.abs(data_mat) + 1e-12)
        vmin = np.percentile(plot_data, 1)
        vmax = np.percentile(plot_data, 99)
        lbl = 'Amplitude (dB)'
    else:
        plot_data = np.abs(data_mat)
        max_val_db = 20 * np.log10(np.max(plot_data) + 1e-10)
        vmin = 10**((max_val_db - 60) / 20)
        vmax = np.max(plot_data)
        lbl = 'Magnitude (Linear)'

    ext = None
    if axis_x is not None and axis_y is not None:
        ext = [axis_x[0], axis_x[-1], axis_y[0], axis_y[-1]]
        
    plt.imshow(plot_data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, origin='lower')
    plt.title(title)
    plt.colorbar(label=lbl)
    if ext is None:
        plt.xlabel("Fast Time / Range (Bins)")
        plt.ylabel("Slow Time / Doppler (Bins)")
    else:
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    
    plt.grid(visible=True, color='white', linestyle='--', alpha=0.3)
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()

# Generate plots
plot_step(phist_comp, "1. Range Compressed", "vehicle_step1_compressed.png")

plot_step(rd_map, "2. Range-Doppler Map (Before RCMC)", "vehicle_step2_rd.png",
          axis_x=range_axis, axis_y=doppler_axis, label_y="Doppler (Hz)")

plot_step(rd_rcmc, "3. RCMC Corrected (Fixed Direction)", "vehicle_step3_rcmc.png",
          axis_x=range_axis, axis_y=doppler_axis, label_y="Doppler (Hz)")

print("Plotting Final Image")
# Note: Final image from function is transposed (Cross-range x Range)
plot_step(sar_image.T, "4. Focused SAR Image (Destroyer)", "vehicle_step4_focused.png",
          axis_x=range_axis, axis_y=cross_range, 
          label_x="Range Offset (m)", label_y="Cross Range (m)",
          xlim=[-scene_extent, scene_extent], ylim=[-scene_extent, scene_extent],
          do_log=False, cmap='gray')

print("\n=== Vehicle Simulation Complete ===")
