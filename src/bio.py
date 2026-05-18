import numpy as np
import constants as c

def get_atp_life_limit(t_blood_c):
    """
    Calculates the metabolic life limit (L_atp) based on Arrhenius decay.
    Physically: "How long until ATP drops to the critical threshold?"
    """
    # 1. Convert to Kelvin for thermodynamic calculation
    t_kelvin = t_blood_c + 273.15
    
    # 2. Calculate Arrhenius Rate Constant (k)
    # k = A * exp(-Ea / (R * T))
    
    #piecewise adjustment for activation energy based on temperature (simplistic and not really defensible)
    if t_blood_c > 1 and t_blood_c < 25:
        Ea_ATP = 82e3 #(j/mol) for 1-25C
    elif t_blood_c >= 25:
        Ea_ATP = 120e3 #(j/mol) for 25-37C
    elif t_blood_c <= 0:
        Ea_ATP = np.inf #(j/mol) for 0-1C

    exponent = -Ea_ATP / (c.R * t_kelvin)
    k_atp = c.A_atp * np.exp(exponent)
    
    # 3. Calculate time to reach threshold (L)
    # Based on first-order decay: L = ln(C_initial / C_limit) / k
    # This represents the total seconds of life available at this state.
    numerator = np.log(c.C0_atp / c.C_lim)
    l_atp = numerator / k_atp
    
    return l_atp

#Add function for 2,3 DPG life limit 

def get_hemolysis_rates(t_blood_c, g_rms):
    """
    Calculates the instantaneous rate of membrane rupture (H_dot).
    Combines Thermal stress (Arrhenius) and Mechanical stress (Power Law).
    """
    t_kelvin = t_blood_c + 273.15
    
    # A. Thermal Hemolysis Rate
    # h_dot_thermal = A_h * exp(-Ea_h / (R * T))
    exp_h = -c.Ea_hem / (c.R * t_kelvin)
    h_dot_thermal = c.A_hem * np.exp(exp_h)
    
    # B. Vibrational Hemolysis Rate
    # h_dot_vibe = beta * (G_rms ^ n)
    h_dot_vibe = c.B * (g_rms ** c.n)
    
    # Total Hemolysis rate (change in % per second)
    h_dot_total = h_dot_thermal + h_dot_vibe
    
    return h_dot_total

def get_structural_life_limit(h_current, h_dot_total):
    """
    Calculates the structural life limit (L_hem).
    Physically: "Based on current bruising, how long until we hit the 0.8% limit?"
    """
    if h_dot_total <= 0:
        return float('inf') # Infinite life if there is no stress
    
    # Margin = Limit (0.8%) - Current (%)
    h_margin = c.H_lim - h_current
    
    # L = Margin / Rate
    l_hem = h_margin / h_dot_total
    
    return max(l_hem, 1.0) # Avoid division by zero, min 1 second

def update_assay_states(h_prev, h_dot_total, atp_prev, t_blood_c, dt):
    """
    Optional: Tracks the actual predicted biomarker values (Virtual Assay).
    Used for logging and 'receipt' generation.
    """
    # 1. Update Hemolysis (Additive)
    h_new = h_prev + (h_dot_total * dt)
    
    # 2. Update ATP (Exponential Decay)
    t_kelvin = t_blood_c + 273.15
    k_atp = c.A_atp * np.exp(-c.Ea_atp / (c.R * t_kelvin))
    atp_new = atp_prev * np.exp(-k_atp * dt)
    
    return h_new, atp_new