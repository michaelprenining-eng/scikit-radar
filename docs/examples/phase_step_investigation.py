import sys, os
sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")

from utility_functions import loadFileData, plotMaps
from scipy.constants import speed_of_light as c0
import scipy.signal.windows as win
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

# phase steps#
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx0_20251212_09-15-31_1.h5" # .12
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx0_20251212_09-15-32_2.h5" # .11
# RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip12_rx_ip12.h5"
# RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip12_rx_ip11.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip11_rx_ip12.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\phaseStep_tx_ip11_rx_ip11.h5"

# RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_phaseStep_right_tx_ip11_rx_ip11.h5"
# RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_phaseStep_right_tx_ip11_rx_ip12.h5"

# 251218 calib
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_tx_ip11_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_tx_ip11_rx_ip12.h5"
RADAR_FILENAME_HDF5_3 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_tx_ip12_rx_ip11.h5"
RADAR_FILENAME_HDF5_4 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_tx_ip12_rx_ip12.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_spTarget2_tx_ip11_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_spTarget2_tx_ip11_rx_ip12.h5"
RADAR_FILENAME_HDF5_3 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_spTarget2_tx_ip12_rx_ip11.h5"
RADAR_FILENAME_HDF5_4 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\phaseStepping_spTarget2_tx_ip12_rx_ip12.h5"


zp_ft = 4
zp_st = 4
nr_of_frames=200
phase_step = 2.8125 # phase step per chirp info for evaluation

# load data
all_data_N11, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_1)
all_data_N12, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_2)
all_data_N21, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_3)
all_data_N22, measurement_parameters = loadFileData(RADAR_FILENAME_HDF5_4)
nr_of_frames = np.min([nr_of_frames,all_data_N11.shape[0],all_data_N12.shape[0],nr_of_frames,all_data_N21.shape[0],all_data_N22.shape[0]])
all_data = np.concatenate((all_data_N11[:nr_of_frames,None],all_data_N12[:nr_of_frames,None],all_data_N21[:nr_of_frames,None],all_data_N22[:nr_of_frames,None]),axis=1)[:,:,:,:int(360/phase_step),:]
# nr_of_frames = np.min([nr_of_frames,all_data_N11.shape[0],all_data_N12.shape[0]])
# all_data = np.concatenate((all_data_N11[:nr_of_frames,None],all_data_N12[:nr_of_frames,None]),axis=1)[:,:,:,:int(360/phase_step),:]
print(f"all_data shape: {all_data.shape}")

angle_results = np.empty(shape=all_data.shape[:4])
max_values_all_frames = np.empty(shape=all_data.shape[:4],dtype=complex)
N_ft = all_data.shape[4]
win_ft = win.hann(N_ft)
win_ft_scaled = win_ft / (np.sum(win_ft) / N_ft)
N_ft_fft = N_ft*zp_ft
ranges = np.linspace(0, 1 - 1 / N_ft_fft, N_ft_fft) * (N_ft - 1) * c0 / measurement_parameters['B']
ranges = ranges[:ranges.shape[0]//2]

for frame_idx in range(nr_of_frames):
    frame_data = all_data[frame_idx]

    # calculate range spectrum
    rp = np.fft.fftshift(np.fft.fft(frame_data*win_ft_scaled,n=N_ft_fft,axis=3),axes=3)[:,:,:,N_ft_fft//2:]
    rp_scaled = rp * 2 / N_ft
    rp_mean = np.mean(np.abs(rp_scaled),axis=(1,2)) # mean over chirps and rx 

    # find target peak, for single target just take maximum value
    rp_peak_search = np.copy(rp_mean)
    rp_peak_search[:,ranges<2] = np.min(rp_peak_search) # ensure that leakage is not detected as maximum
    # rp_peak_search[:,ranges>6] = np.min(rp_peak_search)
    max_indices = np.argmax(rp_peak_search,axis=1)

    max_values = np.empty(shape=rp_scaled.shape[:-1],dtype=complex)
    for rx_idx in range(rp_scaled.shape[0]):
        max_values[rx_idx,:,:] = rp_scaled[rx_idx,:,:,max_indices[rx_idx]]

    max_values_all_frames[frame_idx] = max_values
    max_values_amplitudes = np.abs(max_values)
    

# for last frame plot constellation
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

set_angle = np.linspace(0,phase_step*(angle_results.shape[3] - 1), angle_results.shape[3])

angle_results = np.angle(max_values_all_frames)
angle_results = np.unwrap(angle_results,axis=3)*180/np.pi
angle_results = angle_results - np.mean(angle_results,3)[:,:,:,None] + 180
# angle_results = angle_results - angle_results[:,:,:,0][:,:,:,None]
# angle_results = angle_results - np.mean(angle_results[:,:,:,0][:,:,:,None],axis=0,keepdims=True)
error = angle_results - set_angle[None,None,None,:]

angle_results = angle_results[0,:,0]
# angle_results = np.mean(angle_results,axis=(0,2))
error = error[0,:,0]
# error = np.mean(error,axis=(0,2))

fig_phase, ax_phase = plt.subplots(2,1,num="phase progressions")
ax_phase[0].plot(set_angle,np.diff(angle_results[0],prepend=0),label="N1 --> N1")
ax_phase[0].plot(set_angle,np.diff(angle_results[1],prepend=0),label="N1 --> N2")
ax_phase[0].plot(set_angle,np.diff(angle_results[2],prepend=0),label="N2 --> N1")
ax_phase[0].plot(set_angle,np.diff(angle_results[3],prepend=0),label="N2 --> N2")
# ax_phase[0].plot(set_angle,angle_results[0],label="N1 --> N1")
# ax_phase[0].plot(set_angle,angle_results[1],label="N1 --> N2")
# ax_phase[0].plot(set_angle,angle_results[2],label="N2 --> N1")
# ax_phase[0].plot(set_angle,angle_results[3],label="N2 --> N2")
ax_phase[0].set_xlim([0,360-phase_step])
ax_phase[0].legend()
ax_phase[0].set_title("N1 Tx0 transmitting")
ax_phase[0].grid()
ax_phase[0].set_ylabel("phase (°)")
ax_phase[1].plot(set_angle,error[0],label="N1 --> N1")
ax_phase[1].plot(set_angle,error[1],label="N1 --> N2")
ax_phase[1].plot(set_angle,error[2],label="N2 --> N1")
ax_phase[1].plot(set_angle,error[3],label="N2 --> N2")
ax_phase[1].set_xlim([0,360-phase_step])
ax_phase[1].set_xlabel("Set angle (°)")
ax_phase[1].set_ylabel("error (°)")
ax_phase[1].legend()
ax_phase[1].grid()


max_mag = 20*np.log10(np.max(rp_mean))
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles")
ax_rp.plot(ranges,20*np.log10(rp_mean[0])-max_mag,label="N1 --> N1")
ax_rp.plot(ranges,20*np.log10(rp_mean[1])-max_mag,label="N1 --> N2")
ax_rp.plot(ranges,20*np.log10(rp_mean[2])-max_mag,label="N2 --> N1")
ax_rp.plot(ranges,20*np.log10(rp_mean[3])-max_mag,label="N2 --> N2")
ax_rp.plot(ranges,20*np.log10(rp_peak_search[0])-max_mag,label="psTest")
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
plotMaps(20*np.log10(rd_rxMean),y_axis={'min_value': 0, 'max_value' : ranges[-1], 'label' : 'Range (m)'})