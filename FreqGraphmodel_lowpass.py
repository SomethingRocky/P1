import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Læser csv'en
data = pd.read_csv("frequency analysis2.0.csv", sep=",", skiprows=20, encoding="utf-8", index_col=0)

# navngiver "columns'ne" så de kan arbejdes med
data.columns = ["Channel_1", "Channel_2", "Faze_channel"]

Res = 1000
Cap = 0.22 * 10**-6
omega = 2 * np.pi * np.array(data.index, dtype=float)

squareroot = np.sqrt(1 + (omega * Cap * Res) ** 2)
model_frequency = 20 * np.log10((1  / squareroot))
model_faze = np.degrees(-np.arctan(omega*Res*Cap))

max_squareroot = np.sqrt(1 + (omega * Cap * 1050) ** 2)
min_squareroot = np.sqrt(1 + (omega * Cap * 950) ** 2)
max_freq_deviation = 20 * np.log10((1 / max_squareroot))
min_freq_deviation = 20 * np.log10((1 / min_squareroot))

max_phase_deviation =np.degrees(-np.arctan(omega*1050*Cap))
min_phase_deviation =np.degrees(-np.arctan(omega*950*Cap))

# Plot the two channels against the index
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Channel_1"], color='b', label='Channel 1')
plt.plot(data.index, data["Channel_2"], color='r', label='Channel 2')
plt.axhline(y=-3,linestyle = "--",label ="Cutoff frequency")
plt.axvline(723.43,linestyle = "--")
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Decibel")
plt.title("Channel 1 and Channel 2")
plt.ylim(-10, 10)
plt.legend()
plt.grid(True)
plt.savefig("Freq_analysis_standard")
plt.clf()

# Against the model
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Channel_1"], color='b', label='Channel 1')
plt.plot(data.index, data["Channel_2"], color='r', label='Channel 2')
plt.plot(data.index, model_frequency, label="Model", color='g')
plt.axhline(y=-3,linestyle = "--",label ="Cutoff frequency")
plt.xscale('log')
plt.ylim(-10,10)
plt.xlabel("Frequency")
plt.ylabel("Decibel")
plt.title("Channel 1, Channel 2, and Model")
plt.legend()
plt.grid(True)
plt.savefig("Freq_analysis_model")
plt.clf()

# Error plot (model - data)
error = model_frequency - data["Channel_2"]
plt.figure(figsize=(10, 6))
plt.plot(data.index, error, color='m', label='Error (Model - Channel 2)')
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Error (Decibel)")
plt.title("Error Plot (Model - Channel 2)")
plt.legend()
plt.grid(True)
plt.savefig("Freq_analysis_error")
plt.clf()

# Fill between plot
plt.figure(figsize=(10, 6))
plt.fill_between(data.index, min_freq_deviation, max_freq_deviation, color='green', alpha=0.3, label="Room for error")
plt.plot(data.index, data["Channel_2"], color='r', alpha=0.4, label='Channel 2')
plt.axhline(y=-3,linestyle = "--",label ="Cutoff frequency")
plt.axvline(x=726,linestyle = "--")
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Decibel")
plt.title("Error Plot")
plt.legend()
plt.grid(True)
plt.savefig("Freq_analysis_allowable_error")
plt.clf()

# Plot max_deviation - min_deviation
deviation_difference = max_freq_deviation - min_freq_deviation
plt.figure(figsize=(10, 6))
plt.plot(data.index, deviation_difference, color='b', label='Max Deviation - Min Deviation')
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Deviation Difference (Decibel)")
plt.title("Max Deviation - Min Deviation")
plt.legend()
plt.grid(True)
plt.savefig("Freq_analysis_deviation_difference")
plt.clf()

# Plot against model phase channel
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Faze_channel"], color='r', label='Experiment')
plt.plot(data.index, model_faze, color="b", label="Model")
plt.axhline(y=-45,linestyle = "--",label ="Cutoff frequency")
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Degrees (°)")
plt.title("Phase analysis")
plt.ylim(-90, 90)
plt.legend()
plt.grid(True)
plt.savefig("Faze_analysis_model")
plt.clf()

# Plot phase error (model - data)
phase_error = model_faze - data["Faze_channel"]
plt.figure(figsize=(10, 6))
plt.plot(data.index, phase_error, label='Phase Error (Model - Experiment)')
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Phase Error (Degrees)")
plt.title("Phase Error Plot (Model - Experiment)")
plt.legend()
plt.grid(True)
plt.savefig("Faze_analysis_error")
plt.clf()

#plot fill between
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Faze_channel"], color='r', label='Experiment')
plt.fill_between(data.index,min_phase_deviation,max_phase_deviation,alpha=0.4, color="g",label = "Allowable error")
plt.axhline(y=-45,linestyle = "--",label ="Cutoff frequency")
plt.xscale('log')
plt.xlabel("Frequency")
plt.ylabel("Degrees (°)")
plt.title("Phase analysis")
plt.ylim(-90, 10)
plt.axvline(723.43,linestyle = "--")
plt.legend()
plt.grid(True)
plt.savefig("Faze_analysis_allowable_error")
plt.clf()