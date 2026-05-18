# decision.py
import numpy as np
import constants as c

def calculate_damage_increment(l_atp, l_hem, dt):
    """
    Miner's Rule Accumulation: dS = (w1/L1 + w2/L2) * dt
    Converts 'Time' into 'Fraction of Life Consumed'.
    """
    # Instantaneous damage rate from each assay
    # 1/L represents the % of life lost per second
    rate_atp = 1.0 / l_atp
    rate_hem = 1.0 / l_hem
    
    # Weighted sum of damage
    # w_atp and w_hem should sum to ~1.0 in constants.py
    ds = (c.W_atp * rate_atp + c.W_hem * rate_hem) * dt
    
    return ds

def get_usability_score(s_total):
    """
    Sigmoid Mapping: Transforms cumulative damage (S) into a 0-100 score.
    S = 1.0 is the theoretical failure point (Miner's Rule Limit).
    """
    # Sigmoid function centered at S_CRIT (usually 1.0)
    # U = 100 / (1 + exp(lambda * (S - S_crit)))
    score = 100 / (1 + np.exp(c.LAMBDA * (s_total - c.S_crit)))
    
    return round(score, 2)

def get_traffic_light(score, t_blood):
    """
    Decision Logic for the UI.
    Includes 'Hard Stops' for safety violations.
    """
    # 1. Immediate Safety Violations (Hard Stops)
    if t_blood > 25.0 or t_blood < 1.0:
        return "RED (Temp Violation)"
    
    # 2. Cumulative Damage Logic
    if score > 80:
        return "GREEN"
    elif score > 50:
        return "YELLOW"
    else:
        return "RED (Exhausted)"