import numpy as np
import matplotlib.pyplot as plt
import constants as c
import physics as p

#creating the temperature profile for testing controls method
temperature_C = np.ones(len(c.time)) * 4
temperature_C[500:800] = 18

#convert to kelvin for testing
temperature_K = p.to_kelvin(temperature_C)

#state space variables intialization
ATP = np.zeros(len(c.time))
K = np.zeros(len(c.time))
H = np.zeros(len(c.time))
Q = np.zeros(len(c.time))

#intial conditions
ATP[0] = 1.0
K[0] = 0.0
H[0] = 0.0

#simulation loop
for k in range (len(c.time)-1):
    #ATP depletion
    kA = c.A1 * np.exp(-c.Ea1 / (c.R * temperature_K[k]))
    
    #Potassium leakage
    kP = c.A2 * np.exp(-c.Ea2 / (c.R * temperature_K[k]))
    
    #Hemolysis
    kH = c.A3 * np.exp(-c.Ea3 / (c.R * temperature_K[k]))
    
    #Calculate derivatives
    dATP_dt = -kA * ATP[k]
    dK_dt = kP * (1 - ATP[k]) #assuming potassium leakage is proportional to ATP depletion
    dH_dt = kH * (1 - ATP[k]) #assuming hemolysis is proportional to ATP depletion

    #Update states using Euler's method
    ATP[k+1] = ATP[k] + dATP_dt * c.dt
    K[k+1] = K[k] + dK_dt * c.dt
    H[k+1] = H[k] + dH_dt * c.dt

    #Bound the states between 0 and 1
    ATP[k+1] = np.clip(ATP[k+1], 0, 1)
    H[k+1] = np.clip(H[k+1], 0, 1)
    
    #Calculate cost function (Q)
    Q[k+1] = c.W_atp * (1 - ATP[k+1]) + c.W_potassium * K[k+1] + c.W_hem * H[k+1]

# Normalize quality score
Q = 100 * (Q - np.min(Q)) / (np.max(Q) - np.min(Q))


#plotting results
fig, axs = plt.subplots(4, 1, figsize=(10, 12))

# Temperature
axs[0].plot(c.time / 3600, temperature_C)
axs[0].set_ylabel("Temp (°C)")
axs[0].set_title("Temperature History")

# ATP
axs[1].plot(c.time / 3600, ATP)
axs[1].set_ylabel("ATP")

# Potassium + Hemolysis
axs[2].plot(c.time / 3600, K, label="Potassium")
axs[2].plot(c.time / 3600, H, label="Hemolysis")
axs[2].legend()
axs[2].set_ylabel("Damage States")

# Quality Score
axs[3].plot(c.time / 3600, Q)
axs[3].set_ylabel("Quality")
axs[3].set_xlabel("Time (hours)")
axs[3].set_title("Estimated Blood Quality")

plt.tight_layout()
plt.show()