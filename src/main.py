# main.py
import pandas as pd
import physics
import bio
import decision
import constants as c

def run_prototype(csv_path):
    # 1. Initialization
    df = pd.read_csv(csv_path)
    
    t_blood = 4.0      # Assume cold start
    h_state = 0.001    # Assume 0.1% baseline hemolysis
    atp_state = c.C0_atp  # Start with full ATP concentration
    s_total = 0.0      # Start with zero damage
    
    history = []

    # 2. Iterative Processing (The "Loop")
    for i in range(len(df)):
        # Calculate time step
        dt = df['dt'].iloc[i] if 'dt' in df.columns else 60
        t_ext = df['t_surface'].iloc[i]
        
        # Mocking an accel buffer for this step
        accel_buffer = [(df['ax'].iloc[i], df['ay'].iloc[i], df['az'].iloc[i])]

        # --- PIPELINE START ---
        
        # STEP A: Physics (Signal Conditioning)
        t_blood = physics.update_blood_temp(t_ext, t_blood, dt)
        g_rms = physics.calculate_vibration_energy(accel_buffer)
        
        # STEP B: Biology (Life Limits)
        l_atp = bio.get_atp_life_limit(t_blood)
        h_dot = bio.get_hemolysis_rates(t_blood, g_rms)
        l_hem = bio.get_structural_life_limit(h_state, h_dot)
        
        # STEP C: Decision (Damage Accumulation)
        ds = decision.calculate_damage_increment(l_atp, l_hem, dt)
        s_total += ds
        
        # STEP D: Output Mapping
        score = decision.get_usability_score(s_total)
        status = decision.get_traffic_light(score, t_blood)
        
        # Update Virtual Assay States for logging
        h_state, atp_state = bio.update_assay_states(h_state, h_dot, atp_state, t_blood, dt)

        # --- PIPELINE END ---

        history.append({
            'T_Blood': round(t_blood, 2),
            'G_RMS': round(g_rms, 3),
            'ATP_Level': round(atp_state, 3),
            'Hemolysis': round(h_state, 4),
            'Damage_S': round(s_total, 4),
            'Score': score,
            'Status': status
        })

    return pd.DataFrame(history)
if __name__ == "__main__":
    results = run_prototype("../data/synthetic_transport_log.csv")
    print(results.tail())