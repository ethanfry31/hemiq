import numpy as np

#physics
Lwall = 0.0005 #thickness of bag wall in meters
kwall = 0.19 #thermal conductivity of bag wall
Ac = 0.024 #cross sectional area of bag in m2
kblood = 0.5 #thermal conductivity of blood
Lchar = 0.0125 #characteristic length for blood
rtot = Lwall/(kwall*Ac) + Lchar/(kblood*Ac) #total thermal resistance 

#thermal capacitance of blood mass:
p = 1060 #kg/m3
cp = 3600 #J/kgK
V = 450e-6 #m3
C = p*cp*V

#vibrational constants
n = 2.0 #power law exponent
B = 1e-7 #vibrational damage coefficient

#kinetics

#Metabolic assay constants (ATP)
A_atp = 1.2e8 #pre-exponential factor for ATP decay (1/s)
Ea_atp = 82e3 #activation energy for ATP decay (J/mol)
R = 8.314 #ideal gas constant
C0_atp = 4.5 #initial ATP concentration in uM
C_lim = 2.25 #critical ATP concentration in uM (50% viability threshold)\

#Structural Assay (Hemolysis)
Ea_hem = 115e3 #activation energy for hemolysis (J/mol)
H_lim = 0.8 #% hemolysis
H_init = 0.1 #initial hemolysis level in % (assumed)
A_hem = 1e-5 #pre-exponential factor for thermal hemolysis (1/s)

#tuning parameters
W_atp = 1 #weight for ATP in cost function
W_hem = 0.8 #weight for hemolysis in cost function
LAMBDA = 10 #sigmoid steepness for cost function
S_crit = 1 #critical stress

#controls method variables

dt =1 #timestep in seconds
total_hours = 20 
time = np.arange(0, total_hours * 3600, dt) #time vector in seconds

# ATP depletion
A1 = 1e-7
Ea1 = 50000

# Potassium leakage
A2 = 1e-6
Ea2 = 45000

# Hemolysis
A3 = 5e-5
Ea3 = 55000

W_potassium = 0.4
