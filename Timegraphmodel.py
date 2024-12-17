import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Læser csv'en
filepath = r"time analysis2.0.csv"
data = pd.read_csv(filepath, sep=",", skiprows=19, encoding="utf-8")

# navngiver "columns'ne" så de kan arbejdes med
data.columns = ["Time", "Channel_1", "Channel_2"]

# Filtrere dataen, så det kun er det vigtigte data
data = data[(data["Time"] > -0.0025) & (data["Time"] < 0.0025)]


# Så splitter vi dataen i charge og discharge
# og fjerner det første data punkt, fordi den er fucked
# Det er et mærkeligt transition data punkt
data_discharge = data[data["Time"] < 0].copy()
data_charge = data[data["Time"] >= 0].copy()
data_discharge = data_discharge.iloc[1:]
data_charge = data_charge.iloc[1:]

# Så justere vi lige discharge så den starter ved 0
data_discharge["Time"] = data_discharge["Time"] - data_discharge["Time"].min()

# Modelling at the same time points as the experimental data
Res = 1000
Cap = 0.22*10**-6

model_discharge = 2 * np.exp(-data_discharge["Time"] / (Res*Cap)) - 1
max_model_discharge = 2 * np.exp(-data_discharge["Time"] / (Res*1.05*Cap*1.05)) - 1
min_model_discharge = 2 * np.exp(-data_discharge["Time"] / (Res*0.95*Cap*0.95)) - 1
model_charge = 2 - 2 * np.exp(-data_charge["Time"] / (Res*Cap)) - 1
max_model_charge = 2 - 2 * np.exp(-data_charge["Time"] / (Res*1.05*Cap*1.05)) - 1
min_model_charge = 2 - 2 * np.exp(-data_charge["Time"] / (Res*0.95*Cap*0.95)) - 1


# Beregner deviation fra teoretisk (error)
error_discharge = model_discharge - data_discharge["Channel_2"]
error_charge = model_charge - data_charge["Channel_2"]

# Beregner noise 
noise = np.abs(data["Channel_1"]) - 1

# Beregner noise sepereret
noise_discharge = np.abs(data_discharge["Channel_1"]) - 1
noise_charge = np.abs(data_charge["Channel_1"]) - 1

# Beregner deviation (error) med noise taget fra
error_discharge_with_noise = error_discharge - noise_discharge
error_charge_with_noise = error_charge - noise_charge

# Calculate the mean of noise_charge and noise_discharge
mean_noise_charge = np.mean(noise_charge)
mean_noise_discharge = np.mean(noise_discharge)

# Subtract the mean noise from the errors
error_charge_mean_subtracted = error_charge - mean_noise_charge
error_discharge_mean_subtracted = error_discharge - mean_noise_discharge

# Plottelse af data (combined)
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], data_discharge["Channel_2"], label="Experiment (Discharge)")
plt.plot(data_discharge["Time"], model_discharge, label="Model (Discharge)")
plt.plot(data_charge["Time"], data_charge["Channel_2"], label="Experiment (Charge)")
plt.plot(data_charge["Time"], model_charge, label="Model (Charge)")
plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
plt.savefig("Time_analysis_both.png")
plt.clf()

# Plottelse af data (discharge)
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], data_discharge["Channel_2"], label="Experiment (Discharge)")
plt.plot(data_discharge["Time"], model_discharge, label="Model (Discharge)")
plt.xlabel('Time (s)')
plt.ylabel('Voltage over capacitor(V)')
plt.title('Time analysis (Discharge)')
plt.legend()
plt.grid(True)
plt.savefig("Time_analysis_discharge.png")
plt.clf()

# Plottelse af data (charge)
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], data_charge["Channel_2"], label="Experiment (Charge)", color='green')
plt.plot(data_charge["Time"], model_charge, label="Model (Charge)", color='red')
plt.xlabel('Time (s)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Charge)')
plt.legend()
plt.grid(True)
plt.savefig("Time_analysis_charge.png")
plt.clf()

# Plottelse af square waven as a point style plot
plt.figure(figsize=(10, 6))
plt.scatter(data["Time"], data["Channel_1"], label="The square wave", marker='o')
plt.xlabel('Time (s)')
plt.ylabel('Voltage over circuit (V)')
plt.title('Square Wave analysis (Charge)')
plt.legend()
plt.grid(True)
plt.savefig("Square_Wave_analysis.png")
plt.clf()

# Plottelse af deviation (discharge)
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], error_discharge, label="Deviation (Discharge)")
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation analysis (Discharge)')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_analysis_discharge.png")
plt.clf()

# Plottelse af deviation (charge)
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], error_charge, label="Deviation (Charge)", color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation analysis (Charge)')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_analysis_charge.png")
plt.clf()

# Plottelse af noise (combined)
plt.figure(figsize=(10, 6))
plt.plot(data["Time"], noise, label="Noise (Combined)")
plt.axvline(x=-0.0025, color='blue', linestyle='--', label='Discharge Start', ymin=0.85, ymax=1)
plt.axvline(x=0, color='black', linestyle='--', label='Discharge End / Charge Start', ymin=0.85, ymax=1)
plt.axvline(x=0.0025, color='red', linestyle='--', label='Charge End', ymin=0.85, ymax=1)
plt.xlabel('Time (s)')
plt.ylabel('Noise (V)')
plt.title('Noise analysis (Combined)')
plt.legend()
plt.grid(True)
plt.savefig("Noise_analysis_combined_limited.png")
plt.ylim(-0.01, max(noise) + 0.01)  # Limit y-axis to start from -0.01
plt.savefig("Noise_analysis_combined.png")
plt.clf()

# Plottelse af deviation (discharge) med noise taget fra
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], error_discharge_with_noise, label="Deviation (Discharge) with Noise Subtracted")
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation analysis (Discharge) with Noise Subtracted')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_analysis_discharge_with_noise.png")
plt.clf()

# Plottelse af deviation (charge) med noise taget fra
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], error_charge_with_noise, label="Deviation (Charge) with Noise Subtracted", color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation analysis (Charge) with Noise Subtracted')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_analysis_charge_with_noise.png")
plt.clf()


# Plottelse af deviation analysis for charge med mean noise subtracted
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], error_charge_mean_subtracted, label="Deviation (Charge) with Mean Noise Subtracted", color='purple')
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation Analysis (Charge) with Mean Noise Subtracted')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_Analysis_Charge_Mean_Noise_Subtracted.png")
plt.clf()


# Plottelse af deviation analysis for discharge med mean noise subtracted
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], error_discharge_mean_subtracted, label="Deviation (Discharge) with Mean Noise Subtracted")
plt.xlabel('Time (s)')
plt.ylabel('Deviation (V)')
plt.title('Deviation Analysis (Discharge) with Mean Noise Subtracted')
plt.legend()
plt.grid(True)
plt.savefig("Deviation_Analysis_Discharge_Mean_Noise_Subtracted.png")
plt.clf()


#Plottelse af noise, DE 2 punkter fjernet
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"]-0.0025,np.abs(data_discharge["Channel_1"])-1, label = "Noise discharge")
plt.plot(data_charge["Time"],np.abs(data_charge["Channel_1"])-1, label = "Noise charge")
plt.xlabel('Time (s)')
plt.ylabel('Noise (V)')
plt.title('Noise analysis (Combined)')
plt.legend()
plt.grid(True)
plt.savefig("Noise_analysis_removed")
plt.clf()


#Histogram noise
plt.figure(figsize=(10, 6))
plt.hist(noise_discharge, bins=50, alpha=0.5, label='Noise discharge')
plt.hist(noise_charge, bins=50, alpha=0.5, label='Noise charge')
plt.xlabel('Noise (V)')
plt.ylabel('Occurances')
plt.xlim((-0.001,0.005))
plt.title('Noise Histogram')
plt.legend()
plt.grid(True)
plt.savefig("Noise_histogram.png")
plt.clf()


#Tolerance shown
# Plottelse af data (combined)
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], data_discharge["Channel_2"], label="Experiment (Discharge)")
plt.plot(data_charge["Time"], model_charge, label="Experiment (Charge)")
plt.fill_between(data_discharge["Time"],
                 min_model_discharge,
                 max_model_discharge,
                 color = "green",
                 alpha = 0.3,
                 label = "Allowable error (Discharge)")
plt.fill_between(data_charge["Time"],
                 min_model_charge,
                 max_model_charge,
                 color = "green", 
                 alpha = 0.3,
                 label = "Allowable error (Charge)")

plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
plt.savefig("Time_analysis_both_minmaxed.png")
plt.xlim(0,0.0011)
plt.savefig("Time_analysis_both_minmaxed_limited.png")
plt.clf()


# Calculate theoretical deviation
theoretical_deviation_charge = max_model_charge - min_model_charge
theoretical_deviation_discharge = max_model_discharge - min_model_discharge


#Theoretical deviation  discharge
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], theoretical_deviation_discharge, label="Theoretical Deviation (Discharge)")
plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
#plt.xlim(0,0.0011)
plt.savefig("Theoretical_deviation_discharge.png")
plt.clf()


#Theoretical deviation charge
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], theoretical_deviation_charge, label="Theoretical Deviation (charge)")
plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
#plt.xlim(0,0.0011)
plt.savefig("Theoretical_deviation_charge.png")
plt.clf()


#Theoretical deviation  discharge plotted against experiment
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], theoretical_deviation_discharge, label="Theoretical Deviation (Discharge)")
plt.plot(data_discharge["Time"], error_discharge, label="Deviation (Discharge)")
plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
#plt.xlim(0,0.0011)
plt.savefig("Theoretical_and_exp_deviation_discharge.png")
plt.clf()

#Theoretical deviation charge plotted against experiment
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], theoretical_deviation_charge, label="Theoretical Deviation (charge)")
plt.plot(data_charge["Time"], error_charge, label="Deviation (Charge)", color='purple')
plt.xlabel('Time (S)')
plt.ylabel('Voltage over capacitor (V)')
plt.title('Time analysis (Combined)')
plt.legend()
plt.grid(True)
#plt.xlim(0,0.0011)
plt.savefig("Theoretical_and_exp_deviation_charge.png")
plt.clf()



# Normalize the deviations and errors

normalized_deviation_charge = theoretical_deviation_charge / np.min(theoretical_deviation_charge)
normalized_deviation_discharge = theoretical_deviation_discharge / np.max(theoretical_deviation_discharge)
normalized_error_charge = error_charge / np.min(error_charge)
normalized_error_discharge = error_discharge / np.max(error_discharge)



# Plot normalized theoretical deviation and error charge against experiment
plt.figure(figsize=(10, 6))
plt.plot(data_charge["Time"], normalized_deviation_charge, label="Normalized Theoretical Deviation (Charge)")
plt.plot(data_charge["Time"], normalized_error_charge, label="Normalized Deviation (Charge)", color='purple')
plt.xlabel('Time (S)')
plt.ylabel('Normalized Voltage over capacitor (V)')
plt.title('Normalized Theoretical Deviation vs Experiment (Charge)')
plt.legend()
plt.grid(True)
plt.savefig("Normalized_Theoretical_and_exp_deviation_charge.png")
plt.clf()

# Plot normalized theoretical deviation and error discharge against experiment
plt.figure(figsize=(10, 6))
plt.plot(data_discharge["Time"], normalized_deviation_discharge, label="Normalized Theoretical Deviation (Discharge)")
plt.plot(data_discharge["Time"], normalized_error_discharge, label="Normalized Deviation (Discharge)", color='orange')
plt.xlabel('Time (S)')
plt.ylabel('Normalized Voltage over capacitor (V)')
plt.title('Normalized Theoretical Deviation vs Experiment (Discharge)')
plt.legend()
plt.grid(True)
plt.savefig("Normalized_Theoretical_and_exp_deviation_discharge.png")
plt.clf()

