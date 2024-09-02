import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.3    # Infection rate
sigma = 1/5.2  # Rate of progression from exposed to infectious
gamma = 1/12.39 # Recovery rate
N = 10000     # Total population

# Initial conditions
S0 = 9999    # Initial susceptible population
E0 = 0       # Initial exposed population
I0 = 1       # Initial infectious population
R0 = 0       # Initial recovered population
days = 160    # Simulation period in days

# Time array
t = np.linspace(0, days, days)

# SEIR differential equations
def deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, E0, I0, R0

# Integrate the SEIR equations over the time grid, t
ret = np.zeros((days, 4))
ret[0] = y0
for i in range(1, days):
    ret[i] = ret[i-1] + np.array(deriv(ret[i-1], t[i], N, beta, sigma, gamma)) * (t[i] - t[i-1])

S, E, I, R = ret.T

# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.title('SEIR Model')
plt.legend()
plt.grid()
plt.show()
