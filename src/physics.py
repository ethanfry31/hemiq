# physics.py
import numpy as np
import constants as c

def update_blood_temp(t_ext, t_blood_prev, dt):
    """
    Lumped Capacitance Model using globally defined constants.
    
    Args:
        t_ext (float): Current external ambient temperature (°C).
        t_blood_prev (float): Blood temperature from the previous step (°C).
        dt (float): Time interval in seconds.
        
    Returns:
        float: New internal blood temperature (T_b).
    """
    # Calculate Thermal Inertia (Tau) from constants
    # Tau = R * C
    thermal_inertia = c.rtot * c.C
    
    # Physics: Change in temp is proportional to the gradient and time
    # T_new = T_prev + ((T_ext - T_prev) / Tau) * dt
    t_blood_new = t_blood_prev + ((t_ext - t_blood_prev) / thermal_inertia) * dt
    
    return t_blood_new


def calculate_vibration_energy(accel_buffer):
    """
    Processes raw 3-axis data into G_rms.
    """
    if not accel_buffer:
        return 0.0
    
    # Convert list of (ax, ay, az) to numpy array
    data = np.array(accel_buffer) 
    
    # 1. Vector Magnitude: |G| = sqrt(x^2 + y^2 + z^2)
    # We use axis=1 to calculate magnitude for each timestamped row
    magnitudes = np.sqrt(np.sum(np.square(data), axis=1))
    
    # 2. RMS calculation
    g_rms = np.sqrt(np.mean(np.square(magnitudes)))
    
    return g_rms


def to_kelvin(temp_celsius):
    """Helper for thermodynamic calculations."""
    return temp_celsius + 273.15