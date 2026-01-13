import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import os
import shutil
import gc
from tqdm import tqdm
from vehicle_targets import generate_fighter_jet, generate_destroyer, generate_f35

# --- GOLDEN REFERENCE CONSTANTS ---
C = 299792458.0
Re = 6371000.0
h = 350000.0
R_sat = Re + h
GM = 3.986004418e14
V_sat = np.sqrt(GM / R_sat)

FC = 9.65e9
BW = 500e6
Lambda = C / FC
T_P = 20e-6
K_RATE = BW / T_P
FS = 600e6 
PRF = 5000.0 

theta_look_deg = 45.0
theta_look_rad = np.radians(theta_look_deg)
theta_inc_rad = np.arcsin((R_sat / Re) * np.sin(theta_look_rad))
gamma_rad = theta_inc_rad - theta_look_rad

sin_g = np.sin(gamma_rad)
cos_g = np.cos(gamma_rad)
S0_from_C = np.array([0, -R_sat * sin_g, R_sat * cos_g])
V_unit = np.array([1.0, 0.0, 0.0])
C_offset = np.array([0, 0, -Re])
pos_sat_t0 = S0_from_C + C_offset 
R0 = np.linalg.norm(pos_sat_t0) 

P_TX = 1000.0           
ANT_WIDTH = 0.5         
T_SYS = 290.0           
NF_DB = 5.0             
LOSS_DB = 3.0           
K_BOLTZ = 1.380649e-23  
SCR_DB = 10.0           
K_NU = 1.0     
SNR_BOOST_DB = 26.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_raw_snr_db(r_slant, rcs, wavelength, bandwidth, ant_l, p_tx=P_TX, ant_w=ANT_WIDTH, t_sys=T_SYS, nf_db=NF_DB, loss_db=LOSS_DB):
    ant_area = ant_l * ant_w 
    effective_area = ant_area * 0.6
    gain = 4 * np.pi * effective_area / (wavelength ** 2)
    nf = 10 ** (nf_db / 10)
    loss = 10 ** (loss_db / 10)
    numerator = p_tx * (gain ** 2) * (wavelength ** 2) * rcs 
    denominator = ((4 * np.pi) ** 3) * (r_slant ** 4) * K_BOLTZ * t_sys * bandwidth * loss * nf
    snr_linear = numerator / denominator
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def generate_noise_tensor(shape, ref_power, snr_db, scr_db=SCR_DB, k_nu=K_NU):
    noise_power = ref_power / (10 ** (snr_db / 10))
    noise_std = torch.sqrt(noise_power / 2)
    thermal_noise = noise_std * (torch.randn(shape, device=device) + 1j * torch.randn(shape, device=device))
    
    clutter_power = ref_power / (10 ** (scr_db / 10))
    gamma_dist = torch.distributions.Gamma(torch.tensor(k_nu, device=device), torch.tensor(k_nu, device=device))
    texture = gamma_dist.sample(shape)
    exp_dist = torch.distributions.Exponential(torch.tensor(1.0, device=device))
    speckle = exp_dist.sample(shape)
    
    k_intensity = clutter_power * texture * speckle
    clutter_amp = torch.sqrt(k_intensity)
    clutter_phase = torch.rand(shape, device=device) * 2 * np.pi
    sea_clutter = clutter_amp * torch.exp(1j * clutter_phase)
    
    return thermal_noise + sea_clutter

@torch.no_grad()
def run_physics_spotlight(base_targets, t_vec, pos_sat, vel_sat, heading_deg, speed, l_ant):
    win_len = (2000.0 / C) + T_P + 10e-6 
    num_samples = int(np.ceil(win_len * FS))
    if num_samples % 2 != 0: num_samples += 1
    
    t_start = 2 * R0 / C - win_len / 2
    t_fast_abs = t_start + np.arange(num_samples) / FS
    
    phi = np.radians(heading_deg)
    v_tgt_np = np.array([speed * np.cos(phi), speed * np.sin(phi), 0])
    
    c, s = np.cos(phi), np.sin(phi)
    R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # 1. PRE-LOAD TARGETS TO GPU ONCE 
    all_t_pos_0 = torch.tensor(np.array([R_mat @ t['position'] for t in base_targets]), device=device, dtype=torch.float64)
    all_t_rcs = torch.tensor(np.array([t['rcs'] for t in base_targets]), device=device, dtype=torch.float64)
    
    t_vec_t = torch.tensor(t_vec, device=device, dtype=torch.float64)
    pos_sat_t = torch.tensor(pos_sat, device=device, dtype=torch.float64)
    v_tgt_t = torch.tensor(v_tgt_np, device=device, dtype=torch.float64)
    tf_t = torch.tensor(t_fast_abs, device=device, dtype=torch.float64).view(1, 1, -1)
    
    p_center = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64).view(1, 1, 3)
    
    raw_sig = torch.zeros((len(t_vec), num_samples), device=device, dtype=torch.complex128)
    
    # MEMORY OPT: Reduce batch size significantly
    # 100 targets * 500 pulses * 8000 samples * 16 bytes = 6.4 GB (Too large)
    # 20 targets = 1.28 GB (Safe)
    TARGET_BATCH_SIZE = 20
    num_targets = len(base_targets)
    use_batching = num_targets > TARGET_BATCH_SIZE 
    
    chunk_size = 500
    for i in range(0, len(t_vec), chunk_size):
        idx_e = min(i + chunk_size, len(t_vec))
        
        t = t_vec_t[i:idx_e].view(-1, 1, 1)
        p_sat = pos_sat_t[i:idx_e].view(-1, 1, 3)
        v_sat = torch.tensor(vel_sat[i:idx_e], device=device, dtype=torch.float64).view(-1, 1, 3)
        
        def compute_sig(pos_0, rcs):
            p_tgt = pos_0.view(1, -1, 3) + v_tgt_t.view(1, 1, 3) * t
            diff_tx = p_tgt - p_sat
            dist_tx = torch.norm(diff_tx, dim=2)
            tau_approx = 2 * dist_tx / C
            p_rx = p_sat + v_sat * tau_approx.unsqueeze(2)
            dist_rx = torch.norm(p_tgt - p_rx, dim=2)
            tau = (dist_tx + dist_rx) / C
            
            b_vec = p_center - p_sat
            look_vec = b_vec / torch.norm(b_vec, dim=2, keepdim=True)
            tgt_vec = diff_tx / dist_tx.unsqueeze(2)
            cos_off = torch.clamp(torch.sum(look_vec * tgt_vec, dim=2), -1, 1)
            angle_off = torch.acos(cos_off)
            
            x_val = np.pi * l_ant * torch.sin(angle_off) / Lambda
            gain = torch.ones_like(x_val)
            mask_g = torch.abs(x_val) > 1e-6
            gain[mask_g] = (torch.sin(x_val[mask_g]) / x_val[mask_g])**2
            
            t_local = tf_t - tau.unsqueeze(2)
            mask_p = torch.abs(t_local) <= (T_P / 2)
            phase = np.pi * K_RATE * (t_local**2) - 2 * np.pi * FC * tau.unsqueeze(2)
            
            return rcs.view(1, -1, 1) * gain.unsqueeze(2) * torch.exp(1j * phase) * mask_p

        if not use_batching:
            sig = compute_sig(all_t_pos_0, all_t_rcs)
            raw_sig[i:idx_e] = torch.sum(sig, dim=1)
            del sig
        else:
            sig_chunk = torch.zeros((idx_e - i, num_samples), device=device, dtype=torch.complex128)
            for tgt_idx in range(0, num_targets, TARGET_BATCH_SIZE):
                t_p0_batch = all_t_pos_0[tgt_idx : tgt_idx + TARGET_BATCH_SIZE]
                t_rcs_batch = all_t_rcs[tgt_idx : tgt_idx + TARGET_BATCH_SIZE]
                
                sig_batch = compute_sig(t_p0_batch, t_rcs_batch)
                sig_chunk += torch.sum(sig_batch, dim=1)
                del sig_batch, t_p0_batch, t_rcs_batch
            
            raw_sig[i:idx_e] = sig_chunk
            del sig_chunk
        
    return raw_sig, t_start, num_samples, v_tgt_np

@torch.no_grad()
def tdbp_gpu(raw_t, pos_plat, vel_plat, t_start, num_samples, vel_focus, t_pulses, 
             scene_size, nx=512, ny=512):
    x_axis = np.linspace(-scene_size/2, scene_size/2, nx)
    y_axis = np.linspace(-scene_size/2, scene_size/2, ny)
    
    pos_t = torch.tensor(pos_plat, device=device, dtype=torch.float64)
    vel_t = torch.tensor(vel_plat, device=device, dtype=torch.float64)
    
    t_ref = torch.linspace(-T_P/2, T_P/2, int(T_P * FS), device=device, dtype=torch.float64)
    ref_chirp = torch.exp(1j * np.pi * K_RATE * t_ref**2)
    ref_f = torch.fft.fft(torch.fft.fftshift(ref_chirp), n=num_samples)
    
    raw_f = torch.fft.fft(raw_t, n=num_samples, dim=1)
    rc_data = torch.fft.ifft(raw_f * torch.conj(ref_f).view(1, -1), dim=1)
    del raw_f, ref_f, t_ref
    
    gx, gy = torch.meshgrid(torch.tensor(x_axis, device=device, dtype=torch.float64), 
                            torch.tensor(y_axis, device=device, dtype=torch.float64), indexing='xy')
    grid_pts = torch.stack((gx.flatten(), gy.flatten(), torch.zeros_like(gx).flatten()), dim=1)
    n_pix = grid_pts.shape[0]
    
    final_img = torch.zeros(n_pix, device=device, dtype=torch.complex128)
    batch_size = 2048 
    num_batches = int(np.ceil(n_pix / batch_size))
    
    input_sig = torch.stack((rc_data.real, rc_data.imag), dim=1).unsqueeze(2)
    del rc_data
    
    v_f = torch.tensor(vel_focus, device=device, dtype=torch.float64).view(1, 1, 3)
    t_p = torch.tensor(t_pulses, device=device, dtype=torch.float64).view(-1, 1, 1)
    
    for b in range(num_batches):
        idx0, idx1 = b * batch_size, min((b+1)*batch_size, n_pix)
        g_batch = grid_pts[idx0:idx1].unsqueeze(0) 
        
        t_c = torch.mean(t_p)
        dt = t_p - t_c
        g_batch_expanded = g_batch + v_f * dt 

        diff_tx = g_batch_expanded - pos_t.unsqueeze(1)
        dist_tx = torch.norm(diff_tx, dim=2)
        
        r_unit = diff_tx / dist_tx.unsqueeze(2)
        v_rel = vel_t.unsqueeze(1) - v_f 
        v_rad = torch.sum(v_rel * r_unit, dim=2)
        t_shift = (-FC * (2 * v_rad / C)) / K_RATE
        
        tau_approx = 2 * dist_tx / C 
        pos_rx = pos_t.unsqueeze(1) + vel_t.unsqueeze(1) * tau_approx.unsqueeze(2)
        g_rx = g_batch_expanded + v_f * tau_approx.unsqueeze(2)
        dist_rx = torch.norm(g_rx - pos_rx, dim=2)
        tau_final = (dist_tx + dist_rx) / C
        
        idx_f = (tau_final - t_start + t_shift) * FS
        idx_norm = 2 * (idx_f / num_samples) - 1
        
        grid = torch.cat((idx_norm.unsqueeze(2), torch.zeros_like(idx_norm).unsqueeze(2)), dim=2).unsqueeze(2)
        sampled = torch.nn.functional.grid_sample(input_sig.float(), grid.float(), align_corners=False)
        sampled_c = torch.complex(sampled[:, 0, :, 0].double(), sampled[:, 1, :, 0].double())
        
        phi = 2 * np.pi * FC * tau_final
        phase_corr = torch.exp(1j * phi)
        
        final_img[idx0:idx1] = torch.sum(sampled_c * phase_corr, dim=0)
        del g_batch, g_batch_expanded, diff_tx, dist_tx, grid, sampled, sampled_c, tau_final, phase_corr, idx_f, idx_norm
        
    return final_img.reshape(ny, nx).cpu().numpy()

def main():
    if not os.path.exists("./batch_output"):
        os.makedirs("./batch_output")
        
    DURATION = 5.0
    FPS = 10 
    NUM_FRAMES = int(DURATION * FPS)
    
    T_START, T_END = -2.5, 2.5
    
    TOTAL_PULSES = int(np.ceil(DURATION * PRF))
    STEP_PULSES = int(PRF / FPS)
    CPI_PULSES = int(np.ceil(0.5 * PRF)) # 0.5s CPI
    
    t_vec_all = np.linspace(T_START, T_END, TOTAL_PULSES)
    pos_sat_all = np.zeros((TOTAL_PULSES, 3))
    vel_sat_all = np.zeros((TOTAL_PULSES, 3))
    
    omega = V_sat / R_sat
    for i, t in enumerate(t_vec_all):
        wt = omega * t 
        P_vec = S0_from_C * np.cos(wt) + (R_sat * V_unit) * np.sin(wt)
        V_vec = (V_sat * V_unit) * np.cos(wt) - (S0_from_C * omega) * np.sin(wt)
        pos_sat_all[i] = P_vec + C_offset
        vel_sat_all[i] = V_vec

    # --- BATCH DEFINITIONS (DESTROYER ONLY) ---
    vehicles = [
        # {"name": "PlaneCrus", "gen": generate_fighter_jet, "speed": 250.0, "swath": 2000.0},
        # {"name": "PlaneMach", "gen": generate_fighter_jet, "speed": 515.0, "swath": 2000.0},
        # {"name": "Stealth",   "gen": generate_f35,         "speed": 515.0, "swath": 2000.0},
        {"name": "Destroyer", "gen": generate_destroyer, "speed": 15.0, "swath": 500.0} 
    ]
    
    headings = [0, 90, 45, 135]
    
    algos = [
        {"name": "mBP",    "focus_tgt": True},
        {"name": "StdBP",  "focus_tgt": False}
    ]
    
    pbar = tqdm(total=len(vehicles)*len(headings)*len(algos))
    
    for v in vehicles:
        base_target = v["gen"](center_pos=(0,0,0))
        avg_rcs = 5.0
        if "Plane" in v["name"]: avg_rcs = 5.0
        elif "Stealth" in v["name"]: avg_rcs = 1.0
        elif "Destroyer" in v["name"]: avg_rcs = 5000.0
        
        L_ANT = Lambda * R0 / v["swath"]
        snr_db_raw = calculate_raw_snr_db(R0, avg_rcs, Lambda, BW, L_ANT)
        
        for h in headings:
            for algo in algos:
                desc = f"{v['name']} | H:{h} | {algo['name']}"
                pbar.set_description(desc)
                
                frames = []
                run_id = f"{v['name']}_{int(v['speed'])}_{h}_{algo['name']}"
                temp_dir = f"./temp_{run_id}"
                if not os.path.exists(temp_dir): os.makedirs(temp_dir)
                
                for f in range(NUM_FRAMES):
                    i0 = f * STEP_PULSES
                    i1 = i0 + CPI_PULSES
                    if i1 > TOTAL_PULSES: break
                    
                    t_cpi = t_vec_all[i0:i1]
                    p_cpi = pos_sat_all[i0:i1]
                    v_cpi = vel_sat_all[i0:i1]
                    
                    raw_sig, t_st, n_sp, v_tgt_vec = run_physics_spotlight(
                        base_target, t_cpi, p_cpi, v_cpi, 
                        heading_deg=h, speed=v["speed"], l_ant=L_ANT
                    )
                    
                    sig_p = torch.max(torch.abs(raw_sig)**2)
                    raw_sig = raw_sig + generate_noise_tensor(raw_sig.shape, sig_p, snr_db_raw + SNR_BOOST_DB)
                    
                    vf = v_tgt_vec if algo["focus_tgt"] else np.array([0.0, 0.0, 0.0])
                    
                    img = tdbp_gpu(
                        raw_sig, p_cpi, v_cpi, t_st, n_sp, 
                        vel_focus=vf, t_pulses=t_cpi, 
                        scene_size=v["swath"], nx=512, ny=512
                    )
                    
                    np.save(f"{temp_dir}/frame_{f:03d}.npy", img)
                    del raw_sig, img
                    torch.cuda.empty_cache()
                    # gc.collect() - Opt out for speed (batching solves memory)
                
                frame_files = sorted(os.listdir(temp_dir))
                if len(frame_files) > 0:
                    loaded_frames = [np.load(os.path.join(temp_dir, ff)) for ff in frame_files]
                    
                    g_max = max([np.max(np.abs(fr)) for fr in loaded_frames])
                    g_max = g_max if g_max > 0 else 1.0
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(np.abs(loaded_frames[0]), cmap='gray', vmin=0, vmax=g_max,
                                   extent=[-v["swath"]/2, v["swath"]/2, -v["swath"]/2, v["swath"]/2], 
                                   origin='lower')
                    ax.set_title(f"{v['name']} {v['speed']}m/s H:{h} {algo['name']}")
                    ax.set_xlabel("Along Track (m)")
                    ax.set_ylabel("Ground Range (m)")
                    
                    def update(idx):
                        im.set_data(np.abs(loaded_frames[idx]))
                        return [im]
                    
                    gif_name = f"batch_output/{run_id}.gif"
                    ani = animation.FuncAnimation(fig, update, frames=len(loaded_frames), blit=True)
                    ani.save(gif_name, writer='pillow', fps=FPS)
                    plt.close(fig)
                    
                shutil.rmtree(temp_dir)
                pbar.update(1)
                
    pbar.close()
    print("\nBatch Simulation Complete! Outputs in ./batch_output/")

if __name__ == "__main__":
    main()
