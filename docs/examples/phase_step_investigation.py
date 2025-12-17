import sys, os
sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")

from utility_functions import loadFileData
from scipy.constants import speed_of_light as c0
import scipy.signal.windows as win
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

# phase steps#
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx0_20251212_09-15-31_1.h5" # .12
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx0_20251212_09-15-32_2.h5" # .11
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip12_rx_ip12.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip12_rx_ip11.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip11_rx_ip12.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip11_rx_ip11.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_phaseStep_right_tx_ip11_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_phaseStep_right_tx_ip11_rx_ip12.h5"

zp_ft = 4
zp_st = 4
nr_of_frames=200
rx=3
phase_step = 90 #2.8125

# load data
all_data_N1, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_1)
all_data_N2, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_2)
nr_of_frames = np.min([nr_of_frames,all_data_N1.shape[0],all_data_N2.shape[0]])
all_data = np.concatenate((all_data_N1[:nr_of_frames,None],all_data_N2[:nr_of_frames,None]),axis=1)
print(f"all_data shape: {all_data.shape}")

angle_results = np.empty(shape=all_data.shape[:4])

for frame_idx in range(nr_of_frames):
    frame_data = all_data[frame_idx]

    # calculate range spectrum
    N_ft = frame_data.shape[3]
    win_ft = win.hann(N_ft)
    win_ft_scaled = win_ft / (np.sum(win_ft) / N_ft)
    N_ft_fft = N_ft*zp_ft
    rp = np.fft.fftshift(np.fft.fft(frame_data*win_ft_scaled,n=N_ft_fft,axis=3),axes=3)[:,:,:,N_ft_fft//2:]
    rp_scaled = rp * 2 / N_ft
    rp_mean = np.mean(np.abs(rp_scaled),axis=(1,2))
    rp_peak_search = np.copy(rp_mean)

    ranges = np.linspace(0, 1 - 1 / N_ft_fft, N_ft_fft) * (N_ft - 1) * c0 / measurement_parameters['B']
    ranges = ranges[:ranges.shape[0]//2]

    rp_peak_search[:,ranges<2] = np.min(rp_peak_search)
    max_indices = np.argmax(rp_peak_search,axis=1)

    max_values = np.empty(shape=rp_scaled.shape[:-1],dtype=complex)
    for rx_idx in range(rp_scaled.shape[0]):
        max_values[rx_idx,:,:] = rp_scaled[rx_idx,:,:,max_indices[rx_idx]]

    max_values_angles = np.angle(max_values)
    max_values_angles = np.unwrap(max_values_angles,axis=2)
    max_values_angles = max_values_angles*180/np.pi
    angle_results[frame_idx] = max_values_angles

    max_values_amplitudes = np.abs(max_values)
    

max_values = max_values * np.exp(-1j*np.angle(max_values[:,:,0]))[:,:,None]

fig_scatter, ax_scatter = plt.subplots(1,2,num="phase progressions")
ax_scatter[0].scatter(max_values[0].real, max_values[0].imag,marker="x")
ax_scatter[0].set_aspect("equal")
ax_scatter[0].set_xlim([-500,500])
ax_scatter[0].set_ylim([-500,500])
ax_scatter[0].set_title("N1")
ax_scatter[1].scatter(max_values[1].real, max_values[1].imag,marker="x")
ax_scatter[1].set_aspect("equal")
ax_scatter[1].set_xlim([-500,500])
ax_scatter[1].set_ylim([-500,500])
ax_scatter[1].set_title("N2")
ax_scatter[0].grid()
ax_scatter[1].grid()
ax_scatter[0].set_xlabel("Im")
ax_scatter[0].set_ylabel("Re")
ax_scatter[1].set_xlabel("Im")
ax_scatter[1].set_ylabel("Re")
plt.show()

set_angle = np.linspace(0,phase_step*(max_values_angles.shape[2] - 1), max_values_angles.shape[2])
angle_results = angle_results - np.mean(angle_results[:,:,:,0][:,:,:,None],axis=0,keepdims=True) 
error = angle_results - set_angle[None,None,None,:]

fig_phase, ax_phase = plt.subplots(3,1,num="phase progressions")
ax_phase[0].plot(set_angle,np.mean(angle_results[:,0],axis=(0,1)),label="N1")
ax_phase[0].plot(set_angle,np.mean(angle_results[:,1],axis=(0,1)),label="N2")
ax_phase[0].set_xlim([0,360])
ax_phase[0].legend()
ax_phase[0].set_title("N1 Tx0 transmitting")
ax_phase[0].grid()
ax_phase[0].set_ylabel("phase (°)")
ax_phase[1].plot(set_angle,np.var(angle_results[:,0],axis=(0,1)),label="N1")
ax_phase[1].plot(set_angle,np.var(angle_results[:,1],axis=(0,1)),label="N2")
ax_phase[1].set_xlim([0,360])
ax_phase[1].set_xlabel("Set angle (°)")
ax_phase[1].set_ylabel("var (°)")
ax_phase[1].legend()
ax_phase[2].plot(set_angle,np.mean(error[:,0]**2,axis=(0,1))**0.5,label="N1")
ax_phase[2].plot(set_angle,np.mean(error[:,1]**2,axis=(0,1))**0.5,label="N2")
ax_phase[2].set_xlim([0,360])
ax_phase[2].set_xlabel("Set angle (°)")
ax_phase[2].set_ylabel("rmse (°)")
ax_phase[2].legend()
ax_phase[2].grid()
ax_phase[1].grid()


max_mag = 20*np.log10(np.max(rp_mean))
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles")
ax_rp.plot(ranges,20*np.log10(rp_mean[0])-max_mag,label="N1 tx0")
ax_rp.plot(ranges,20*np.log10(rp_mean[1])-max_mag,label="N2 tx0")
# ax_rp.plot(ranges,20*np.log10(rp_peak_search[0])-max_mag,label="psTest")
ax_rp.set_xlim([0,ranges[-1]])
ax_rp.legend()
ax_rp.grid()
ax_rp.set_xlabel("Round Trip Range (m)")
ax_rp.set_ylabel("Normalized Power (dB)")
plt.show()

# calculated rd map
N_st = all_data.shape[3]
win_st = win.hann(N_st)
win_st_scaled = win_st / (np.sum(win_st) / N_st)
N_st_fft = N_st*zp_st
rd = np.fft.fftshift(np.fft.fft(rp_scaled*win_st_scaled[:,None],n=N_st_fft,axis=2),axes=2)
rd_scaled = rd / N_st

rd_rxMean = np.mean(np.abs(rd),1)
fig_rd, ax_rd = plt.subplots(2,1)
ax_rd[0].imshow(20*np.log10(rd_rxMean[0]),aspect="auto")
ax_rd[0].set_title("N1 tx0")
ax_rd[1].imshow(20*np.log10(rd_rxMean[1]),aspect="auto")
ax_rd[1].set_title("N2 tx0")
plt.show()