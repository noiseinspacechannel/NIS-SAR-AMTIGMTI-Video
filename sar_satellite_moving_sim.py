"""
SAR Satellite Moving Destroyer Simulation
Combines satellite orbital SAR (350km) with moving destroyer targets.
"""
import sys
import os
import numpy as np

# Redirect output to log file
sys.stdout = open('sim_satellite_moving_log.txt', 'w', buffering=1)
sys.stderr = sys.stdout
print("Starting Satellite Moving Destroyer Simulation...", flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.signal.windows import hamming
from vehicle_targets import generate_destroyer

# Function to rotate points around Z axis
def rotate_points(points, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return np.dot(points, R.T)

# --- Constants & Specs (ICEYE/Capella-like + 350km Orbit) ---
C = 299792458.0
Re = 6371000.0      # Earth Radius (m)
h = 350000.0        # Orbit Altitude (m)
R_sat = Re + h      # Orbit Radius
GM = 3.986004418e14 # Earth Gravitational Parameter

# Orbital Velocity (Circular)
V_sat = np.sqrt(GM / R_sat) # ~7697 m/s

FC = 9.65e9         # 9.65 GHz (X-Band)
BW = 500e6          # 500 MHz Bandwidth
Lambda = C / FC
PRF = 6000.0        # 6 kHz
T_p = 20e-6         # 20 us Pulse Width

# Geometry Setup
theta_look_deg = 45.0 
theta_look_rad = np.radians(theta_look_deg)

# Calculate angles and ranges
theta_inc_rad = np.arcsin((R_sat / Re) * np.sin(theta_look_rad))
theta_inc_deg = np.degrees(theta_inc_rad)
gamma_rad = theta_inc_rad - theta_look_rad
gamma_deg = np.degrees(gamma_rad)
R0 = np.sqrt(Re**2 + R_sat**2 - 2 * Re * R_sat * np.cos(gamma_rad))

print("--- Simulation Parameters ---")
print(f"Orbit Altitude: {h/1000:.1f} km")
print(f"Orbit Radius: {R_sat/1000:.1f} km")
print(f"Orbital Velocity: {V_sat:.1f} m/s")
print(f"Look Angle (Sat): {theta_look_deg:.1f} deg")
print(f"Incidence Angle (Target): {theta_inc_deg:.1f} deg")
print(f"Earth Angle (Gamma): {gamma_deg:.2f} deg")
print(f"Slant Range R0: {R0/1000:.2f} km")
print(f"Center Freq: {FC/1e9:.2f} GHz")
print(f"Bandwidth: {BW/1e6:.0f} MHz")

# --- Scene & Timing ---
T_int = 1.2 # 1.2 seconds integration
num_pulses = int(np.ceil(T_int * PRF))
if num_pulses % 2 != 0: num_pulses += 1

print(f"Integration Time: {T_int:.2f} s")
print(f"Num Pulses: {num_pulses}")

t_vec = np.linspace(-T_int/2, T_int/2, num_pulses)

# --- Orbit Trajectory Generation ---
omega = V_sat / R_sat
sin_g = np.sin(gamma_rad)
cos_g = np.cos(gamma_rad)
S0_from_C = np.array([-R_sat * sin_g, 0, R_sat * cos_g])
V_unit = np.array([0.0, 1.0, 0.0])
C_offset = np.array([0, 0, -Re])

print("Generating Orbit Trajectory...")
pos_sat = np.zeros((num_pulses, 3))
vel_sat = np.zeros((num_pulses, 3))

for i, t in enumerate(t_vec):
    wt = omega * t
    P_vec = S0_from_C * np.cos(wt) + (R_sat * V_unit) * np.sin(wt)
    V_vec = (V_sat * V_unit) * np.cos(wt) - (S0_from_C * omega) * np.sin(wt)
    pos_sat[i] = P_vec + C_offset
    vel_sat[i] = V_vec

# Effective Velocity for processing
V_eff = V_sat * np.sqrt(Re / R_sat)
print(f"Effective Velocity: {V_eff:.1f} m/s")

# --- Generate Static Destroyer Model ---
print("\n--- Generating Destroyer Target ---")
base_targets_raw = generate_destroyer(center_pos=(0,0,0))

# Apply initial 90-degree rotation to match sar_satellite_sim.py orientation
# FIX: Removed to match sar_destroyer_moving_sim behavior (Forward motion)
base_targets = base_targets_raw
print(f"Base targets: {len(base_targets)} (Standard orientation)")

# --- Physics Engine with Motion ---
def run_moving_physics(targets, t_vec, pos_sat, vel_target):
    print(f"\nSimulating Physics (Target Vel={vel_target})...")
    
    fs = 600e6
    num_samples = int(22e-6 * fs)
    t_start_fast = (2 * R0 / C) - (T_p/2) - 1e-6
    fast_times = np.linspace(0, num_samples/fs, num_samples)
    t_fast_abs = t_start_fast + fast_times
    
    raw = np.zeros((len(t_vec), num_samples), dtype=complex)
    k_rate = BW / T_p
    
    # Original Positions at t=0
    t_pos_0 = np.array([t['position'] for t in targets])
    t_rcs = np.array([t['rcs'] for t in targets])
    
    # Target Velocity Vector
    vt = np.array(vel_target)
    
    for i in range(len(t_vec)):
        if i % 100 == 0: print(f"Pulse {i}/{len(t_vec)}", end='\r')
        
        t_curr = t_vec[i]
        sat_p = pos_sat[i]
        pulse_resp = np.zeros(num_samples, dtype=complex)
        
        # Update Target Positions: P(t) = P0 + V * t
        t_pos_current = t_pos_0 + vt * t_curr
        
        diff = t_pos_current - sat_p 
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)
        
        tau = 2 * dist / C
        phase_base = -4.0 * np.pi * FC * dist / C
        amp = np.sqrt(t_rcs)
        
        for b in range(len(targets)):
            tau_b = tau[b]
            t_local = t_fast_abs - tau_b
            mask = np.abs(t_local - T_p/2) <= T_p/2
            if not np.any(mask): continue
            chirp_phase = np.pi * k_rate * ((t_local - T_p/2)**2)
            sig = amp[b] * np.exp(1j * (phase_base[b] + chirp_phase)) * mask
            pulse_resp += sig
            
        raw[i, :] = pulse_resp
    print("\nPhysics Complete.")
    return raw, t_start_fast, fs

# --- Radar System Parameters (Satellite Platform - ICEYE/Capella-like) ---
P_TX = 1000.0           # Transmit power (W) - typical smallsat SAR (peak)
ANT_LENGTH = 3.5        # Antenna length (m) - azimuth direction
ANT_WIDTH = 0.5         # Antenna width (m) - elevation direction
T_SYS = 290.0           # System noise temperature (K)
NF_DB = 5.0             # Noise figure (dB) - space-qualified receiver
LOSS_DB = 3.0           # System losses (dB) - atmosphere, components
K_BOLTZ = 1.380649e-23  # Boltzmann constant (J/K)

# Sea clutter parameters
SCR_DB = 10.0           # Signal-to-Clutter Ratio (dB)
K_NU = 1.0              # K-distribution shape (sea state 3)

def calculate_snr_db(r_slant, rcs, wavelength, bandwidth, t_int, p_tx=P_TX, 
                     ant_l=ANT_LENGTH, ant_w=ANT_WIDTH, t_sys=T_SYS, nf_db=NF_DB, loss_db=LOSS_DB):
    """Calculate SNR using radar equation. Returns (snr_db, gain_db)."""
    ant_area = ant_l * ant_w * 0.6
    gain = 4 * np.pi * ant_area / (wavelength ** 2)
    gain_db = 10 * np.log10(gain)
    nf = 10 ** (nf_db / 10)
    loss = 10 ** (loss_db / 10)
    numerator = p_tx * (gain ** 2) * (wavelength ** 2) * rcs * t_int
    denominator = ((4 * np.pi) ** 3) * (r_slant ** 4) * K_BOLTZ * t_sys * bandwidth * loss * nf
    snr_linear = numerator / denominator
    snr_db = 10 * np.log10(snr_linear)
    return snr_db, gain_db

def add_ocean_noise(raw_data, snr_db, scr_db=SCR_DB, k_nu=K_NU):
    """Add K-distributed sea clutter and thermal noise using calculated SNR."""
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

# --- RDA Processing ---
def sar_focus_rda(phist, center_wavelength_m, pulse_width_sec, chirp_rate_hzpsec, sample_rate_hz, prf_hz, platform_speed_mps, range_grp_m):
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
    win_az = hamming(num_pulses)
    phist_compressed_windowed = phist_compressed * win_az 
    range_doppler = np.fft.fftshift(np.fft.fft(np.fft.fftshift(phist_compressed_windowed, axes=1), axis=1), axes=1)

    PRF = prf_hz
    if num_pulses % 2 == 0:
        doppler_freq = np.arange(-num_pulses/2, num_pulses/2) * (PRF / num_pulses)
    else:
        doppler_freq = np.arange(-(num_pulses - 1)/2, (num_pulses - 1)/2 + 1) * (PRF / num_pulses)
    
    range_axis_m = fast_time_compressed * speed_of_light_mps / 2

    # RCMC
    lambd = speed_of_light_mps / center_frequency_hz
    V_r = platform_speed_mps

    delta_R_matrix = (range_axis_m[:, None] * (doppler_freq[None, :]**2) * lambd**2) / (8 * V_r**2)
    range_doppler_rcmc = np.zeros_like(range_doppler, dtype=complex)
    
    for k in range(num_pulses):
        delta_R = delta_R_matrix[:, k]
        shifted_range_axis = range_axis_m - delta_R
        range_profile = range_doppler[:, k]
        if len(shifted_range_axis) > 1:
            f = interp1d(shifted_range_axis, range_profile, kind='linear', fill_value=0, bounds_error=False)
            range_doppler_rcmc[:, k] = f(range_axis_m)
        else:
             range_doppler_rcmc[:, k] = range_profile

    # Azimuth Compression
    Ka = (2 * V_r**2) / (lambd * range_axis_m)
    inv_Ka = 1.0 / Ka
    H_matrix = np.exp(-1j * np.pi * (inv_Ka[:, None] * doppler_freq[None, :]**2))
    range_doppler_filtered = range_doppler_rcmc * H_matrix
    sar_image = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(range_doppler_filtered, axes=1), axis=1), axes=1)
    sar_image_mag = np.abs(sar_image)

    cross_range_m = V_r * slow_time_sec
    range_midpoint = np.mean(range_axis_m)
    range_axis_centered = range_axis_m - range_midpoint

    return (sar_image_mag.T, range_axis_centered, cross_range_m)

# --- Main Simulation Loop ---
chirp_rate = BW / T_p

# Ship speed 15 m/s
ship_speed = 15.0

scenarios = [
    {'name': 'stationary', 'angle': 0, 'speed': 0, 'fname': 'sar_satellite_moving_scen_stationary.npz'},
    {'name': 'moving_0deg', 'angle': 0, 'speed': ship_speed, 'fname': 'sar_satellite_moving_scen_0deg.npz'},
    {'name': 'moving_45deg', 'angle': 45, 'speed': ship_speed, 'fname': 'sar_satellite_moving_scen_45deg.npz'},
    {'name': 'moving_90deg', 'angle': 90, 'speed': ship_speed, 'fname': 'sar_satellite_moving_scen_90deg.npz'},
    {'name': 'moving_135deg', 'angle': 135, 'speed': ship_speed, 'fname': 'sar_satellite_moving_scen_135deg.npz'},
]

for sc in scenarios:
    print(f"\n=== Scenario: {sc['name']} ===")
    
    # 1. Rotate Targets
    pos_mat = np.array([t['position'] for t in base_targets])
    pos_rot = rotate_points(pos_mat, sc['angle'])
    
    current_targets = []
    for j, t in enumerate(base_targets):
        new_t = t.copy()
        new_t['position'] = pos_rot[j, :]
        current_targets.append(new_t)
        
    # 2. Velocity Vector (0 deg = X-axis)
    theta_v = np.radians(sc['angle'])
    vx = sc['speed'] * np.cos(theta_v)
    vy = sc['speed'] * np.sin(theta_v)
    vz = 0.0
    velocity = [vx, vy, vz]
    
    print(f"Heading: {sc['angle']} deg, Speed: {sc['speed']} m/s, Vel: {velocity}")
    
    # 3. Physics
    raw_dat, t_start, fs_val = run_moving_physics(current_targets, t_vec, pos_sat, velocity)
    
    # 3.5. Calculate SNR from radar equation and add noise
    avg_rcs = 50000.0  # m² - typical destroyer RCS
    snr_db, gain_db = calculate_snr_db(R0, avg_rcs, Lambda, BW, T_int)
    print(f"Radar Equation: Gain={gain_db:.1f}dB, SNR={snr_db:.1f}dB")
    raw_dat = add_ocean_noise(raw_dat, snr_db)
    
    # 4. Processing
    (img, r_ax, cr_ax) = sar_focus_rda(raw_dat.T, Lambda, T_p, chirp_rate, fs_val, PRF, V_eff, R0)
    
    # 5. Save
    print(f"Saving {sc['fname']}...")
    np.savez(sc['fname'], 
             final_image=img, 
             range_axis=r_ax, 
             cross_range=cr_ax,
             # Geometry
             orbit_alt=h,
             orbit_vel=V_sat,
             look_ang=theta_look_deg,
             inc_ang=theta_inc_deg,
             r0=R0,
             v_eff=V_eff,
             prf=PRF,
             # Scenario Specifics
             scen_name=sc['name'],
             ship_speed=sc['speed'],
             ship_heading=sc['angle'],
             ship_vel=velocity)

print("\n=== All Scenarios Complete ===")
