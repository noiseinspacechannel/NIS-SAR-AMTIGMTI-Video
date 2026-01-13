"""
SAR Satellite Simulation - Commercial Specs / 350km Circular Orbit
Target: Destroyer
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Redirect output to log
sys.stdout = open('sim_satellite_log.txt', 'w', buffering=1)
sys.stderr = sys.stdout
print("Starting Satellite SAR Simulation...", flush=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.signal.windows import hamming
from vehicle_targets import generate_destroyer

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
# Desired Look Angle at Target center (broadside)
theta_look_deg = 45.0 
theta_look_rad = np.radians(theta_look_deg)

# Calculate Slant Range R0 and Earth Center Angle gamma
# Law of Sines/Cosines on triangle EarthCenter-Target-Sat
# Angle at Target (from vertical) is (180 - theta_look) ?? 
# No, Look angle is defined from Sat Nadir.
# Incidence angle is defined from Target Vertical.
# Let's use Incidence Angle calculation.
# sin(theta_inc) = (R_sat / Re) * sin(theta_look)
theta_inc_rad = np.arcsin((R_sat / Re) * np.sin(theta_look_rad))
theta_inc_deg = np.degrees(theta_inc_rad)

# Earth central angle gamma = theta_inc - theta_look
gamma_rad = theta_inc_rad - theta_look_rad
gamma_deg = np.degrees(gamma_rad)

# Slant Range R0 using Law of Cosines
# R0^2 = Re^2 + R_sat^2 - 2 Re R_sat cos(gamma)
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
# Azimuth Resolution goal ~0.5m
# L_ant approx 2 * rho_a. 
# rho_a = Lambda * R / (2 * L_sa) -> L_sa = Lambda * R / (2 * rho_a)
# Integration Time T_int = L_sa / V_sat? Actually L_sa / V_g roughly.
# Let's set a sufficient Integration Time.
# Beamwidth approx Lambda / L_ant.
# Footprint = R * Beamwidth.
# T_int = Footprint / V_g.
# Commercial high res spot/stripmap ~1-3 seconds.
T_int = 1.2 # 1.2 seconds integration
num_pulses = int(np.ceil(T_int * PRF))
# Make even number for FFT
if num_pulses % 2 != 0: num_pulses += 1

print(f"Integration Time: {T_int:.2f} s")
print(f"Num Pulses: {num_pulses}")

t_vec = np.linspace(-T_int/2, T_int/2, num_pulses)

# --- Orbit Trajectory Generation ---
# Coordinate System:
# Earth Center at (0, 0, -Re)
# Target at (0, 0, 0)
# Reference Sat Pos at t=0 (Broadside):
# Sat must be in Y-Z plane relative to Earth Center, but rotated by gamma?
# Wait, let's define the plane.
# Target is at North Pole of the Earth Sphere relative to Earth Center? 
# If Earth Center is (0,0,-Re), Target is at (0,0,0) (Top).
# Sat is at distance R_sat from (0,0,-Re).
# Angle gamma from Z axis.
# So Sat(0) = (0, R_sat * sin(gamma), R_sat * cos(gamma) - Re) ?
# If Sat is at y-offset, orbit is in Y-Z plane?
# Circular orbit around Earth Center.
# If orbit normal is X-axis (Motion in Y-Z plane), it would pass overhead if gamma=0.
# With gamma offset, we need the orbit plane to be TILTED so that the circle passes through Sat(0).
# Alternatively, simplier: 
# Let the Orbit be in the X-Z plane (Polar).
# Sat moves in X direction.
# Side looking angle looks in Y direction.
# Sat(0) position:
#   R vector from center magnitude R_sat.
#   Angle gamma 'left' of vertical.
#   S0_vec_from_center: (0, -R_sat*sin(gamma), R_sat*cos(gamma))
#   This puts Sat at y = negative (Look Right).
#   Velocity direction: X axis.
#   So Orbit Plane is defined by Normal Vector Y_rotated? No.
#   If Velocity is pure X at t=0, the orbit plane is roughly X-Z tilted by gamma?
#   Let's define Orbit Plane Normal roughly Y-ish?
#   If S0 is in Y-Z plane, and Velocity is X.
#   Then Orbit Plane is spanned by S0 and V.
#   S0 = (0, -Rs sin(g), Rs cos(g)).
#   V0 = (V, 0, 0).
#   This defines the plane.
#   Trajectory: S(t)_from_center = S0 * cos(wt) + (V0/w) * sin(wt).
#   This creates a great circle.
#   Angular velocity omega = V_sat / R_sat.

omega = V_sat / R_sat

# Initial Position vector from Earth Center (at t=0)
# Look Right -> Sat is Left (Negative Y)
sin_g = np.sin(gamma_rad)
cos_g = np.cos(gamma_rad)
S0_from_C = np.array([0, -R_sat * sin_g, R_sat * cos_g]) # (0, y, z)

# Velocity unit vector (Flying +X)
V_unit = np.array([1.0, 0.0, 0.0])

# Verify orthogonality (dot product should be 0)
# S0 . V = 0. Correct.

# Generate Positions
pos_sat = np.zeros((num_pulses, 3))
vel_sat = np.zeros((num_pulses, 3))

# Earth Center offset to shift back to Target Grid (0,0,0 at surface)
C_offset = np.array([0, 0, -Re])

print("Generatng Orbit Trajectory...")
for i, t in enumerate(t_vec):
    # Orbital Mechanics (Rotated Frame)
    # R(t) = S0 cos(wt) + (R_sat * V_hat) sin(wt)
    # Wait, R_sat * V_hat? 
    # V = dR/dt = -S0 w sin + R_sat V_hat w cos = V0 cos(wt) - S0 w sin(wt)
    # Correct.
    
    wt = omega * t
    
    P_vec = S0_from_C * np.cos(wt) + (R_sat * V_unit) * np.sin(wt)
    V_vec = (V_sat * V_unit) * np.cos(wt) - (S0_from_C * omega) * np.sin(wt)
    
    # Shift to Scene coords (Target @ 0,0,0)
    # Sat_Scene = P_vec + C_offset 
    # (Since P_vec is from Center. C_offset is Center position?? No.)
    # Earth Center is at (0,0,-Re).
    # So P_vec is relative to (0,0,-Re).
    # Absolute Pos = (0,0,-Re) + P_vec.
    
    pos_sat[i] = P_vec + C_offset
    vel_sat[i] = V_vec

# Effective Velocity for processing
# V_eff = V_sat * sqrt(Re / R_sat) for RDA focusing on flat processing plane approx?
# The standard "effective velocity" for stationary target on curve earth:
# V_eff = V_s * sqrt(Re / (Re + h)) ?
# Actually simpler: Match the curvature. 
# V_eff^2 approx V_sat * V_g.
# V_g = V_sat * (Re / R_sat).
# So V_eff = V_sat * sqrt(Re/R_sat).
V_eff = V_sat * np.sqrt(Re / R_sat)
print(f"Effective Velocity (Calculated): {V_eff:.1f} m/s")

# --- Function to rotate points around Z axis ---
def rotate_points(points, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return np.dot(points, R.T)

# --- Target Generation ---
print("\n--- Generating Destroyer Target ---")
base_targets = generate_destroyer(center_pos=(0,0,0))

# Rotate the destroyer 90 degrees so it aligns with Y-axis instead of X-axis
pos_mat = np.array([t['position'] for t in base_targets])
pos_rot = rotate_points(pos_mat, 90)  # 90 degrees counter-clockwise

sim_targets = []
for j, t in enumerate(base_targets):
    new_t = t.copy()
    new_t['position'] = pos_rot[j, :]
    sim_targets.append(new_t)

print(f"Total targets: {len(sim_targets)} (rotated 90 degrees)")

# --- Physics (Exact 3D) ---
def run_physics_engine(targets, pos_sat, t_vec):
    print("\n--- Starting Physics Engine ---")
    
    # Fast Time Setup
    # Window around R0
    # Range Extent needed? 
    # scene size ~200m. 
    # Differential range max ~ 200m * sin(theta) ~ 150m.
    # Pulse width 20us -> 3km length.
    # We need to capture enough samples.
    
    fs = BW * 1.2 # Nyquist + margin
    # fs = 600 MHz -> Huge data.
    # 20us pulse -> 12000 samples.
    # Range window: ~ 100-200m scene? 
    # We can use "De-chirped" or just simulate raw returns if memory allows.
    # 32000 pulses * 20000 samples is BIG (600M complex floats ~ 5GB).
    # Might be too slow for this environment.
    
    # Optimization: Use a smaller 'valid' window since we know where target is.
    # Compute min/max delay.
    p_targ = np.array([t['position'] for t in targets])
    # Roughly:
    dist_center = np.linalg.norm(pos_sat[len(pos_sat)//2])
    # dist_min ~ dist_center - 200, dist_max ~ dist_center + 200
    
    # Let's reduce fs if possible? No, need BW for resolution.
    # We can just window the receive time closely.
    # Echo duration = T_p + (MaxDist - MinDist)/c.
    # (200m)/c is small (<1us).
    # So window length ~ T_p + 1us = 21us.
    # 21us * 600MHz = 12600 samples.
    # 32k * 12.6k ~ 400M samples. ~3GB RAM. Should be okay?
    
    fs = 600e6 
    print(f"Sampling Rate: {fs/1e6} MHz")
    
    num_samples = int(22e-6 * fs) # 22us window
    print(f"Samples per pulse: {num_samples}")
    
    # Start time for window (near R0)
    t_start_fast = (2 * R0 / C) - (T_p/2) - 1e-6 
    
    fast_times = np.linspace(0, num_samples/fs, num_samples)
    t_fast_abs = t_start_fast + fast_times
    
    raw = np.zeros((len(t_vec), num_samples), dtype=complex)
    k_rate = BW / T_p
    
    t_pos = p_targ
    t_rcs = np.array([t['rcs'] for t in targets])
    
    print("Simulating pulses...")
    for i in range(len(t_vec)):
        if i % 100 == 0: print(f"Pulse {i}/{len(t_vec)}", end='\r')
        
        sat_p = pos_sat[i]
        
        # Vectorized target calcs
        diff = t_pos - sat_p 
        dist = np.sqrt(np.sum(diff**2, axis=1))
        
        tau = 2 * dist / C
        phase_base = -4.0 * np.pi * FC * dist / C
        amp = np.sqrt(t_rcs)
        
        # Accumulate response
        pulse_resp = np.zeros(num_samples, dtype=complex)
        
        # Loop targets or vectorize? 
        # Vectorizing 35 targets is fine.
        for b in range(len(targets)):
            # This target's delay
            tau_b = tau[b]
            
            # Time relative to this target's arrival
            # Chirp definition: rect((t - tau - Tp/2)/Tp) * exp(...)
            # Center of pulse is at tau + Tp/2
            
            t_local = t_fast_abs - tau_b
            
            # Mask valid times (within pulse width)
            # Centered at Tp/2 in local time
            mask = np.abs(t_local - T_p/2) <= T_p/2
            
            if not np.any(mask): continue
            
            chirp_phase = np.pi * k_rate * ((t_local - T_p/2)**2)
            sig = amp[b] * np.exp(1j * (phase_base[b] + chirp_phase)) * mask
            pulse_resp += sig
            
        raw[i, :] = pulse_resp
        
    print("\nPhysics Engine Complete.")
    return raw, t_start_fast, fs

# --- Radar System Parameters (Satellite - ICEYE/Capella-like) ---
P_TX = 1000.0           # Transmit power (W) - smallsat SAR peak
ANT_LENGTH = 3.5        # Antenna length (m) - azimuth
ANT_WIDTH = 0.5         # Antenna width (m) - elevation
T_SYS = 290.0           # System temperature (K)
NF_DB = 5.0             # Noise figure (dB) - space-qualified
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

# Run Simulation
raw_data, t_start, fs_val = run_physics_engine(sim_targets, pos_sat, t_vec)

# Calculate SNR from radar equation and add noise
avg_rcs = 50000.0  # m² - destroyer
snr_db, gain_db = calculate_snr_db(R0, avg_rcs, Lambda, BW, T_int)
print(f"Radar Equation: Gain={gain_db:.1f}dB, SNR={snr_db:.1f}dB")
raw_data = add_ocean_noise(raw_data, snr_db)

# --- Processing (Simple RDA with V_eff) ---
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
            phist_compressed, range_doppler, range_doppler_rcmc, doppler_freq)

# Calculate parameters for RDA
chirp_rate = BW / T_p
# Note: sar_focus_rda expects phist shape (Range, Azimuth), so pass raw_data.T
final_img, r_axis, cross_rng, rc_time_T, rd_map_T, rd_rcmc_T, dop_axis = sar_focus_rda(
    raw_data.T, 
    Lambda, # center_wavelength_m
    T_p,    # pulse_width_sec
    chirp_rate, # chirp_rate_hzpsec
    fs_val, # sample_rate_hz
    PRF,    # prf_hz
    V_eff,  # platform_speed_mps
    R0      # range_grp_m (approx R0)
)

# Transpose intermediate outputs back to (Azimuth, Range) for consistency with viewer expectations?
# Viewer currently does `data.T` on these.
# sar_satellite_viewer expects:
#   raw_phist: (Az, R)
#   range_comp: (Az, R) - derived from rc_time_T (R, Az) -> need to transpose
#   rd_map: (Az, R) - derived from rd_map_T (R, Az) -> need to transpose
#   rd_rcmc: (Az, R) - derived from rd_rcmc_T (R, Az) -> need to transpose
#   final_image: final_img is already (Az, R) (transposed inside function)

rc_time = rc_time_T.T
rd_map = rd_map_T.T
rd_rcmc = rd_rcmc_T.T

# Coordinates
# Cross Range (Azimuth)
# x = V_eff * t_slow
az_axis = t_vec * V_eff 

print("Saving Data...")
np.savez('sar_satellite_data.npz',
         raw_phist=raw_data, # No decimation
         range_comp=rc_time, # No decimation
         rd_map=rd_map,
         rd_rcmc=rd_rcmc,
         final_image=final_img,
         range_axis=r_axis,
         cross_range=az_axis,
         doppler_axis=dop_axis,
         # Save Geometry Info for viewer
         orbit_alt=h,
         orbit_vel=V_sat,
         look_ang=theta_look_deg,
         inc_ang=theta_inc_deg,
         bw=BW,
         r0=R0,
         fc=FC,
         v_eff=V_eff)
print("Saved sar_satellite_data.npz")
print("Satellite Simulation Complete.")
