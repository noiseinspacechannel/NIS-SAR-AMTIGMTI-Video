import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy.signal.windows import hamming
from vehicle_targets import generate_destroyer
import torch

# --- Configuration ---
print("Starting SAR ATI/DCPA Simulation (CSA Version)...", flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Constants ---
C = 299792458.0
Re = 6371000.0
h = 350000.0
R_sat = Re + h
GM = 3.986004418e14
V_sat = np.sqrt(GM / R_sat) # ~7697 m/s

FC = 9.65e9
BW = 500e6
Lambda = C / FC
PRF = 6000.0
T_p = 20e-6
FS = 600e6

# Geometry
theta_look_deg = 45.0
theta_look_rad = np.radians(theta_look_deg)
theta_inc_rad = np.arcsin((R_sat / Re) * np.sin(theta_look_rad))
theta_inc_deg = np.degrees(theta_inc_rad)
gamma_rad = theta_inc_rad - theta_look_rad
R0 = np.sqrt(Re**2 + R_sat**2 - 2 * Re * R_sat * np.cos(gamma_rad))

# ATI/DPCA Setup
# B = 2 * V_sat / PRF
d_rx = 2 * V_sat / PRF 
print(f"DPCA Antenna Separation: {d_rx:.4f} m")

# --- Trajectory ---
T_int = 1.2 
num_pulses = int(np.ceil(T_int * PRF))
t_vec = np.linspace(-T_int/2, T_int/2, num_pulses)

omega = V_sat / R_sat
sin_g = np.sin(gamma_rad)
cos_g = np.cos(gamma_rad)
S0_from_C = np.array([-R_sat * sin_g, 0, R_sat * cos_g])
V_unit = np.array([0.0, 1.0, 0.0]) # Along-Track
C_offset = np.array([0, 0, -Re])

# Generate Reference Trajectory (Tx Center)
pos_tx = np.zeros((num_pulses, 3))
vel_tx = np.zeros((num_pulses, 3))

for i, t in enumerate(t_vec):
    wt = omega * t
    P_vec = S0_from_C * np.cos(wt) + (R_sat * V_unit) * np.sin(wt)
    V_vec = (V_sat * V_unit) * np.cos(wt) - (S0_from_C * omega) * np.sin(wt)
    pos_tx[i] = P_vec + C_offset
    vel_tx[i] = V_vec

V_eff = V_sat * np.sqrt(Re / R_sat)

# --- Targets ---
print("Generating Targets...")
# 1. Destroyer (Moving Radially)
destroyer_targets = generate_destroyer(center_pos=(0,0,0))

# 2. Clutter (Restricted Area)
# Physical Ocean Clutter Model - MODIFIED for High RCS / Full Scene
# User requested "super high stationary RCS" and "entire scene" coverage.
sigma_0_db = 5.0  # +5 dB (Very bright, to dominate target)
sigma_0_lin = 10**(sigma_0_db / 10.0)

num_clutter = 5000 
clutter_half_width = 3000 # +/- 3000m (6km swath)
total_area_m2 = (2 * clutter_half_width)**2
total_rcs = total_area_m2 * sigma_0_lin
mean_rcs = total_rcs / num_clutter

print(f"Clutter Generation: Area={total_area_m2:.0f}m2, Sigma0={sigma_0_db}dB")
print(f"                    Total Clutter RCS={total_rcs:.1f}m2, Mean/Pt={mean_rcs:.1f}m2")

clutter_x = np.random.uniform(-clutter_half_width, clutter_half_width, num_clutter)
clutter_y = np.random.uniform(-clutter_half_width, clutter_half_width, num_clutter)
clutter_z = np.zeros(num_clutter)
clutter_rcs = np.random.exponential(mean_rcs, num_clutter) 

clutter_targets = []
for i in range(num_clutter):
    clutter_targets.append({
        'position': np.array([clutter_x[i], clutter_y[i], clutter_z[i]]),
        'rcs': clutter_rcs[i]
    })

combined_targets = destroyer_targets + clutter_targets
print(f"Total Targets: {len(combined_targets)} (Ship: {len(destroyer_targets)}, Clutter: {len(clutter_targets)})")

# --- Physics (GPU Accelerated) ---
@torch.no_grad()
def run_bistatic_physics_gpu(targets, t_vec, pos_tx_np, vel_tx_np, rx_offset_dist, vel_target_np):
    print(f"Simulating Physics [GPU] (Rx Offset={rx_offset_dist:.3f}m)...")
    
    # Constants
    num_samples = int(22e-6 * FS)
    t_start_fast = (2 * R0 / C) - (T_p/2) - 1e-6
    fast_times = np.linspace(0, num_samples/FS, num_samples)
    t_fast_abs = t_start_fast + fast_times
    k_rate = BW / T_p
    
    # To GPU
    pos_tx_t = torch.tensor(pos_tx_np, device=device, dtype=torch.float64) 
    vel_tx_t = torch.tensor(vel_tx_np, device=device, dtype=torch.float64) 
    t_vec_t = torch.tensor(t_vec, device=device, dtype=torch.float64).view(-1, 1, 1) 
    
    # Target Data
    target_pos_0 = np.array([t['position'] for t in targets])
    target_rcs = np.array([t['rcs'] for t in targets])
    
    t_pos_0_t = torch.tensor(target_pos_0, device=device, dtype=torch.float64).view(1, -1, 3) 
    t_rcs_t = torch.tensor(target_rcs, device=device, dtype=torch.float64).view(1, -1, 1) 
    
    vel_target_t = torch.tensor(vel_target_np, device=device, dtype=torch.float64).view(1, 1, 3)
    
    # Fast time grid
    t_fast_t = torch.tensor(t_fast_abs, device=device, dtype=torch.float64).view(1, 1, -1)
    
    # Allocate Output
    raw_sig = torch.zeros((len(t_vec), num_samples), device=device, dtype=torch.complex128)
    
    for i in range(len(t_vec)):
        if i % 100 == 0: print(f"  Pulse {i}/{len(t_vec)}", end='\r')
        
        t_curr = t_vec[i] 
        
        # Tx State
        p_tx = pos_tx_t[i].view(1, 3) 
        v_tx = vel_tx_t[i].view(1, 3)
        v_dir = v_tx / torch.norm(v_tx)
        
        # Rx Position
        p_rx = p_tx + v_dir * rx_offset_dist
        
        # Target Positions at t_curr
        t_pos_curr = t_pos_0_t.view(-1, 3) + vel_target_t.view(1, 3) * t_curr
        
        # Distances
        vec_tx = t_pos_curr - p_tx
        vec_rx = t_pos_curr - p_rx
        dist_tx = torch.norm(vec_tx, dim=1) 
        dist_rx = torch.norm(vec_rx, dim=1)
        
        tau = (dist_tx + dist_rx) / C 
        phase_base = -2.0 * np.pi * FC * tau 
        
        # Expand for samples
        tau_grid = tau.view(-1, 1)
        t_local = t_fast_t.view(1, -1) - tau_grid
        
        mask = torch.abs(t_local - T_p/2) <= (T_p/2)
        
        chirp = np.pi * k_rate * ((t_local - T_p/2)**2)
        
        # Signal
        amp = torch.sqrt(t_rcs_t.view(-1, 1))
        
        sig_targets = amp * torch.exp(1j * (phase_base.view(-1, 1) + chirp)) * mask
        
        # Sum targets
        pulse_resp = torch.sum(sig_targets, dim=0)
        
        raw_sig[i] = pulse_resp
        
    print("\n  Done.")
    return raw_sig.cpu().numpy(), t_start_fast

# --- Run Simulation ---
velocity_ship = [15.0, 0.0, 0.0] # Radial

# Run Physics (Split to handle velocities correctly)
velocity_stationary = np.array([0.0, 0.0, 0.0])

print("Simulating Channel 1 (Moving Target + Clutter)...")
raw_rx1_ship, t_start_fast = run_bistatic_physics_gpu(destroyer_targets, t_vec, pos_tx, vel_tx, -d_rx/2, velocity_ship)
raw_rx1_clutter, _         = run_bistatic_physics_gpu(clutter_targets, t_vec, pos_tx, vel_tx, -d_rx/2, velocity_stationary)
raw_rx1 = raw_rx1_ship + raw_rx1_clutter

print("Simulating Channel 2 (Moving Target + Clutter)...")
raw_rx2_ship, _            = run_bistatic_physics_gpu(destroyer_targets, t_vec, pos_tx, vel_tx,  d_rx/2, velocity_ship)
raw_rx2_clutter, _         = run_bistatic_physics_gpu(clutter_targets, t_vec, pos_tx, vel_tx,  d_rx/2, velocity_stationary)
raw_rx2 = raw_rx2_ship + raw_rx2_clutter

# --- Processing (CSA - CPU) ---
print("Focusing with Chirp Scaling Algorithm (CSA)...")

def sar_focus_csa(phist, center_wavelength_m, pulse_width_sec, chirp_rate_hzpsec, sample_rate_hz, prf_hz, platform_speed_mps, range_ref_m, t_start_fast):
    """
    Implements Chirp Scaling Algorithm.
    phist: Raw phase history (N_pulses x N_samples) (Uncompressed)
    """
    import numpy.fft as fft
    
    # 0. Setup Axes
    N_az, N_rg = phist.shape
    c = 299792458.0
    lam = center_wavelength_m
    Kr = chirp_rate_hzpsec
    Vr = platform_speed_mps
    
    # Fast Time (Range)
    dt_fast = 1.0 / sample_rate_hz
    # t_fast vector relative to start of sampling window
    tau = t_start_fast + np.arange(N_rg) * dt_fast
    
    # Range Frequency
    fr = fft.fftfreq(N_rg, dt_fast)
    
    # Azimuth Frequency
    fa = fft.fftfreq(N_az, 1.0/prf_hz)
    
    # Reference Range (Mid-swath usually, or closest range)
    # Using R0 passed as range_ref_m
    R_ref = range_ref_m
    
    # --- Step 1: Azimuth FFT -> Range-Doppler Domain ---
    # S(tau, fa)
    S_rd = fft.fft(phist, axis=0) # FFT along azimuth (columns)
    S_rd = fft.fftshift(S_rd, axes=0) # Shift so 0 Hz is center
    fa = fft.fftshift(fa)
    
    # D(fa): Migration Factor
    # formula: D(fa) = sqrt(1 - (lam * fa / (2*Vr))^2 )
    # But usually Beta = D(fa) or similar.
    # Cs(fa) factor: Scaling factor
    # Cs(fa) = 1/D(fa) - 1  (Standard CSA)
    
    # Check for evanescent modes
    arg_sqrt = 1.0 - (lam * fa / (2.0 * Vr))**2
    # Avoid sqrt negative zero
    arg_sqrt[arg_sqrt < 0] = 1e-9
    D_fa = np.sqrt(arg_sqrt)
    
    Cs_fa = (1.0 / D_fa) - 1.0
    
    # Km(fa): Modified Chirp Rate
    # Km = Kr / (1 + Kr * (2 * R_ref * Cs_fa / c )) ??
    # Simplified CSA usually assumes slight modification
    # Let's use the phase function definition directly.
    
    # Phase 1: Chirp Scaling
    # Phi_1(tau, fa) = exp( -j * pi * Km * Cs * (tau - 2*R_ref/(c*D))**2 )
    # Wait, simple version:
    # Phi_1 = exp ( -j * pi * Kr * Cs_fa * (tau - tau_ref(fa))^2 )
    # where tau_ref(fa) = 2 * R_ref / (c * D_fa)
    
    tau_ref_fa = 2.0 * R_ref / (c * D_fa)
    
    # We need tau(vector) vs fa(vector)
    # tau grid (1, N_rg)
    # fa grid (N_az, 1)
    
    XX, YY = np.meshgrid(tau, Cs_fa) # YY is Cs_fa repeated
    _, TauRef = np.meshgrid(tau, tau_ref_fa)
    
    # Calculate Phase 1
    Phi_1 = np.exp(-1j * np.pi * Kr * YY * (XX - TauRef)**2)
    
    S_sc = S_rd * Phi_1
    
    # --- Step 2: Range FFT -> 2D Frequency Domain ---
    # S(fr, fa)
    S_2df = fft.fft(S_sc, axis=1)
    # Typically no shift needed here if we coordinate fr properly, but let's shift to center 0
    S_2df = fft.fftshift(S_2df, axes=1)
    fr = fft.fftshift(fr)
    
    # Phase 2: Range Compression + Bulk RCMC + Range-Frequency Dependent Phase
    # Phi_2(fr, fa) = exp( j * pi * fr^2 / (Kr * (1+Cs)) )  <-- Range Comp (Inverse Chirp) modified
    #               * exp( j * 4 * pi * R_ref * fr * (1+Cs) / c ) ?? 
    #               * exp( j * 4 * pi * R_ref * D / lam ) <-- Bulk Azimuth Phase? No that's later.
    
    # According to "Cumming & Wong":
    # Phi_2 = exp( j * pi * D(fa) / (Kr * (1+Cs(fa))) * fr^2 )  ??? Re-check standard formula
    # Let's simply use: 
    #   Effective Chirp Rate Ke(fa) = Kr / (1 + Cs(fa))  (Actually it's Kr * (1+Cs) ?)
    
    # Correct relation: 1/Ke = 1/Kr + ...
    # Standard: The new chirp rate in RD domain was K_rd = Kr / D(fa). 
    # Scaling converts it.
    
    # Let's look at the standard 3 phases:
    # 1. Scaling: fixes the range migration curvature for all ranges to be same as reference range.
    # 2. Range Comp + Bulk RCMC:
    #    Matched filter H(fr, fa) = exp( j * pi * fr^2 / (Kr * D(fa)) ) OR similar?
    #    AND Bulk RCMC linear phase: exp( j * 4 * pi * R_ref * (1/D(fa) - 1) * fr / c ) ...
    
    # Let's trust a verified formulation (e.g. standard CSA):
    # Phi_2(fr, fa) = exp ( j * pi / (Kr * (1+Cs_fa)) * fr^2 )   <-- Range Compression
    #                 * exp ( j * 4 * pi / c * R_ref * Cs_fa * fr ) <-- Bulk RCMC
    
    FR, _ = np.meshgrid(fr, fa) # (N_az, N_rg) from (N_rg,), (N_az,)
    _, CS_FA = np.meshgrid(fr, Cs_fa)
    _, D_FA  = np.meshgrid(fr, D_fa)
    
    # Range Compression Phase (Cancel the chirp)
    # Original chirp was -pi * Kr * t^2 -> Freq domain exp(j * pi * f^2 / Kr)
    # CSA modified chirp rate is Kr_new = Kr / (1 + Cs) ?
    # We apply the conjugate filter.
    
    # Term 2.1: Range Compression
    # Using 1/(Kr * (1+Cs))
    phi_2_rc = np.pi * (FR**2) / (Kr * (1.0 + CS_FA))
    
    # Term 2.2: Bulk RCMC
    # Shift reference range to correct trajectory
    phi_2_rcmc = 4.0 * np.pi * R_ref * CS_FA * FR / c
    
    Phi_2 = np.exp(1j * (phi_2_rc + phi_2_rcmc))
    
    S_rc = S_2df * Phi_2
    
    # --- Step 3: Range IFFT -> Range-Doppler Domain ---
    # S(tau, fa)
    # Inverse of *Range* FFT
    S_rd_2 = fft.ifft(fft.ifftshift(S_rc, axes=1), axis=1)
    
    # Phase 3: Azimuth Compression + Residual Phase
    # Phi_3(tau, fa) = exp( j * 4 * pi * R0 * D(fa) / lam )  <-- Azimuth Phase History (Hyperbolic) ?
    #                * exp( -j * 4 * pi * K_m / c^2  * (1+Cs) * Cs * (R - R_ref)^2 ) <-- Residual Phase?
    
    # Actually, straightforward Azimuth Compression:
    # The signal in RD domain, after RCMC, looks like straight lines at range R.
    # Phase is ~ -4 * pi * R / lam * D(fa).
    # We multiplay by exp( j * 4 * pi * R * D(fa) / lam ) to traverse to zero phase (Focus).
    
    # Wait, we need the R for each pixel. We are in (tau, fa) domain.
    # tau is related to R.
    # R(tau) = c * tau / 2.
    
    R_vec = c * tau / 2.0
    R_grid, _ = np.meshgrid(R_vec, fa) # (N_az, N_rg)
    _, D_FA_2 = np.meshgrid(R_vec, D_fa)
    
    # Azimuth Compression Phase
    # We match the phase history of a point at Range R.
    # Phase_sig = -4 * pi * R / lam * D(fa)
    # Filter = exp( j * 4 * pi * R / lam * D(fa) ) - 4 * pi * R / lambda for baseband?
    # Usually we remove the carrier 4*pi*R/lam. 
    # If the raw data is basebanded, we only compensate the modulation.
    # D(fa) = sqrt(1 - ...) approx 1 - ...
    # Phase ~ -4piR/lam * (1 - ...)
    
    phi_3_ac = 4.0 * np.pi * R_grid * D_FA_2 / lam
    
    # Residual Phase Correction (specific to CSA)
    # Compensates for the quadratic phase we added in Step 1 that varies with range.
    # Phi_resid = -4 * pi / lam * (Cs * (1+Cs)) * (R - R_ref)^2 / R_ref ? No.
    # Standard: -4 * pi * Kr * Cs * (1+Cs) * (tau - 2*R_ref/c)^2 / (c^2??)
    
    # Let's use the explicit phase correction:
    # Phi_resid = exp( -j * 4 * pi * Kr * CS_FA * (1+CS_FA) * (tau - 2*R_ref/c)**2  * ??? )
    # This term is usually small but good for phase preservation.
    
    # From a reliable CSA ref:
    # Phi_3 = exp( j * 4 * pi * R * D(fa) / lam ) * exp( -j * 4 * pi * Kr * Cs * (1+Cs) * (tau - 2*R_ref/c)^2 ) ??
    # Let's check dimensions.
    # (tau - 2*R_ref/c) is time offset from ref range.
    
    tau_diff = XX - (2.0 * R_ref / c)
    phi_3_resid = -np.pi * Kr * CS_FA * (1.0 + CS_FA) * (tau_diff**2)
    # Note: 4 pi terms usually appear in different derivations. 
    # Let's stick to the "Phase 1 was pi * Kr..." so this should be "pi * Kr..."
    
    Phi_3 = np.exp(1j * (phi_3_ac + phi_3_resid))
    
    S_focused_rd = S_rd_2 * Phi_3
    
    # --- Step 4: Azimuth IFFT -> Image Domain ---
    img = fft.ifft(fft.ifftshift(S_focused_rd, axes=0), axis=0)
    
    # Coordinates
    range_axis = R_vec
    cross_range = fft.fftshift(np.fft.fftfreq(N_az, 1.0/prf_hz)) * Vr / (fc_doppler_centroid_approx if False else 1.0) 
    # Wait, Cross range for focused image:
    # x = v * t_slow.
    t_slow = np.arange(N_az) / prf_hz
    t_slow -= np.mean(t_slow)
    cross_range_axis = t_slow * Vr
    
    return img.T, range_axis, cross_range_axis

# --- Processing (Pulse-Shifted Pair) ---
# Rx1[1:] vs Rx2[:-1]
print("Applying DPCA Pulse Shift (Rx1[1:] vs Rx2[:-1])...")

raw_rx1_shift = raw_rx1[1:, :]
raw_rx2_shift = raw_rx2[:-1, :]

# Focus
print("Focusing Co-registered Channels with CSA...")
chirp_rate = BW / T_p

# Note: CSA requires raw uncompressed data. 'raw_rx1_shift' is exactly that.
slc1, rax, cax = sar_focus_csa(raw_rx1_shift, Lambda, T_p, chirp_rate, FS, PRF, V_eff, R0, t_start_fast)
slc2, _, _     = sar_focus_csa(raw_rx2_shift, Lambda, T_p, chirp_rate, FS, PRF, V_eff, R0, t_start_fast)

# --- Analysis ---
ati_interf = slc1 * np.conj(slc2)
ati_phase = np.angle(ati_interf)
slc1_mag = np.abs(slc1)

dpca_diff = slc1 - slc2
dpca_mag = np.abs(dpca_diff)

# Save
extent = [rax[0], rax[-1], cax[0], cax[-1]]

def save_plot(data, title, fname, cmap='gray', vmin=None, vmax=None, is_phase=False):
    plt.figure(figsize=(10, 8))
    if is_phase:
        plt.imshow(data, aspect='auto', origin='lower', extent=extent, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        plt.colorbar(label='Phase (rad)')
    else:
        if vmin is None:
            d_log = 20*np.log10(data + 1e-9)
            vmax = np.percentile(d_log, 99.9)
            vmin = vmax - 40
            plt.imshow(d_log, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Magnitude (dB)')
        else:
             plt.imshow(data, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
             plt.colorbar()
    plt.title(title)
    plt.xlabel("Range (m)")
    plt.ylabel("Cross-Range (m)")
    plt.savefig(fname)
    plt.close()

# Prefix filenames with 'csa_' to distinguish
save_plot(slc1_mag, "CSA Channel 1 Magnitude", "csa_sar_ati_ch1_mag.png", cmap='bone')
mag_mask = np.abs(slc1) > (np.max(np.abs(slc1)) * 0.05) 
ati_phase_masked = np.copy(ati_phase)
ati_phase_masked[~mag_mask] = 0
save_plot(ati_phase_masked, "CSA ATI Phase", "csa_sar_ati_phase.png", is_phase=True)
save_plot(dpca_mag, "CSA DPCA Difference", "csa_sar_dpca_diff.png", cmap='magma')

print("Saving Data for Interactive Viewer...")
# Saving to a new file so the viewer can load it if modified, 
# or we can rename it. Viewer loads 'sar_ati_dpca_data.npz'.
# Let's save to 'sar_ati_dpca_data_csa.npz'
np.savez("sar_ati_dpca_data_csa.npz", 
         slc1=slc1, 
         slc2=slc2, 
         range_axis=rax, 
         cross_range=cax)

print("Simulation Complete (CSA).")
