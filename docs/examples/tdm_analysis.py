import sys, os
sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")

from utility_functions import loadFileData, plotMaps
from scipy.constants import speed_of_light as c0
import scipy.signal.windows as win
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skradar.detection import cfar
plt.ioff()

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleRight20251210_09-32-51_1.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleRight20251210_09-32-51_2.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleClose20251210_09-29-37_1.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleClose20251210_09-29-37_2.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleFar20251210_09-31-47_1.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleFar20251210_09-31-47_2.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\TDM_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\TDM_rx_ip12.h5"
# RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\TDM_tx0_poleRight20251212_09-31-47_1.h5"
# RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\TDM_tx0_poleRight20251212_09-31-47_2.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_TDM_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_TDM_rx_ip12.h5"


frame_idx = 15
zp_ft = 4
zp_st = 4

# load data
all_data_N1, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_1)
all_data_N2, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_2)
print(f"all_data shape: {all_data_N1.shape}")
tx0_data_N1 = all_data_N1[frame_idx,:,::2]
tx1_data_N1 = all_data_N1[frame_idx,:,1::2]
tx_split_data_N1 = np.vstack((tx0_data_N1[None], tx1_data_N1[None]))
tx0_data_N2 = all_data_N2[frame_idx,:,::2]
tx1_data_N2 = all_data_N2[frame_idx,:,1::2]
tx_split_data_N2 = np.vstack((tx0_data_N2[None], tx1_data_N2[None]))
all_data = np.vstack((tx_split_data_N1[None],tx_split_data_N2[None]))

# calculate range spectrum
N_ft = all_data.shape[4]
win_ft = win.hann(N_ft)
win_ft_scaled = win_ft / (np.sum(win_ft) / N_ft)
N_ft_fft = N_ft*zp_ft
rp = np.fft.fftshift(np.fft.fft(all_data*win_ft_scaled,n=N_ft_fft,axis=4),axes=4)[:,:,:,:,N_ft_fft//2:]
rp_scaled = rp * 2 / N_ft
rp_mean = np.mean(np.abs(rp_scaled),axis=(2,3))
rp_peak_search = np.copy(rp_mean)

ranges = np.linspace(0, 1 - 1 / N_ft_fft, N_ft_fft) * (N_ft - 1) * c0 / measurement_parameters['B']
ranges = ranges[:ranges.shape[0]//2]

max_velocity = c0 / (2 * measurement_parameters["Ts_s"]/2 * (2 * measurement_parameters["fc"]))

min_val = np.min(rp_peak_search)
rp_peak_search[:,:,ranges<2] = min_val

max_indices = np.argmax(rp_peak_search,axis=2)

max_values = np.empty(shape=rp_scaled.shape[:-1],dtype=complex)
for tx_idx in range(rp_scaled.shape[0]):
    for rx_idx in range(rp_scaled.shape[1]):
        max_values[tx_idx,rx_idx,:,:] = rp_scaled[tx_idx,rx_idx,:,:,max_indices[tx_idx,rx_idx]]


# calculated rd map
N_st = all_data.shape[3]
win_st = win.hann(N_st)
win_st_scaled = win_st / (np.sum(win_st) / N_st)
N_st_fft = N_st*zp_st
rd = np.fft.fftshift(np.fft.fft(rp_scaled*win_st_scaled[:,None],n=N_st_fft,axis=3),axes=3)
rd_scaled = rd / N_st

# Your data prep
data_dB = np.mean(np.abs(rd),axis=2)
data_dB = 20 * np.log10(data_dB/np.max(data_dB))

plotMaps(np.reshape(data_dB,(-1,data_dB.shape[2],data_dB.shape[3])),
         titles=["N1tx0 --> N1", "N2tx0 --> N1","N1tx0 --> N2","N2tx0 --> N2"],
         cbarLabel = 'Magnitude (dB)',
         x_axis={'min_value': -max_velocity, 'max_value' : max_velocity, 'label' : 'Velocity (m/s)'},
         y_axis={'min_value': 0, 'max_value' : ranges[-1], 'label' : 'Range (m)'})

to_plot = np.angle(max_values)*180/np.pi

fig_phase, ax_phase = plt.subplots(2,2,num="phase progressions",constrained_layout=True,figsize=[2*5.3, 2*5.3 *0.8*1])
for row in range(ax_phase.shape[0]):
    for col in range(ax_phase.shape[0]):
        ax_phase[row,col].plot(np.angle(max_values[row,col,0])*180/np.pi,label="rx0")
        ax_phase[row,col].plot(np.angle(max_values[row,col,1])*180/np.pi,label="rx1")
        ax_phase[row,col].plot(np.angle(max_values[row,col,2])*180/np.pi,label="rx2")
        ax_phase[row,col].plot(np.angle(max_values[row,col,3])*180/np.pi,label="rx3")
        ax_phase[row,col].set_xlim([0,ranges[-1]])
        ax_phase[row,col].set_ylim([-180,180])
        ax_phase[row,col].legend()
        ax_phase[row,col].grid()
        ax_phase[row,col].set_xlabel("Ramp Index (1)")
        ax_phase[row,col].set_ylabel("Phase (°)")

ax_phase[0,0].set_title("N1tx0 -> N1")
ax_phase[0,1].set_title("N1tx0 -> N1")
ax_phase[1,0].set_title("N1tx0 -> N2")
ax_phase[1,1].set_title("N2tx0 -> N2")

max_mag = 20*np.log10(np.max(rp_mean))
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles")
ax_rp.plot(ranges,20*np.log10(rp_mean[0,0])-max_mag,label="N1tx0 -> N1")
ax_rp.plot(ranges,20*np.log10(rp_mean[0,1])-max_mag,label="N2tx0 -> N1")
ax_rp.plot(ranges,20*np.log10(rp_mean[1,0])-max_mag,label="N1tx0 -> N2")
ax_rp.plot(ranges,20*np.log10(rp_mean[1,1])-max_mag,label="N2tx0 -> N2")
# ax_rp.plot(ranges,20*np.log10(rp_peak_search[0,0])-max_mag,label="psTest")
ax_rp.set_xlim([0,ranges[-1]])
ax_rp.legend()
ax_rp.grid()
ax_rp.set_xlabel("Round Trip Range (m)")
ax_rp.set_ylabel("Normalized Power (dB)")
plt.show()

