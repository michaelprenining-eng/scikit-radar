import sys, os
sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")

import utility_functions
from scipy.constants import speed_of_light as c0
import scipy.signal.windows as win
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skradar.detection import cfar
plt.ioff()

folder_path = r"C:/Users/Preining/Documents/CD_Lab/antenna_chamber/measurement_data/fullTDM/"

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

# 251218 calib
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\TDM_ip11_0deg_2m3_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\TDM_ip11_0deg_2m3_rx_ip12.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\TDM_ip12_0deg_2m3_rx_ip11.h5"
RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\251218_calib\TDM_ip12_0deg_2m3_rx_ip12.h5"

# fullTDM
RADAR_FILENAME_HDF5_1 = os.path.join(folder_path, "TDM_right_moreChirps_allTx_ip11.h5")
RADAR_FILENAME_HDF5_2 = os.path.join(folder_path, "TDM_right_moreChirps_allTx_ip12.h5")
RADAR_FILENAME_HDF5_1 = os.path.join(folder_path, "TDM_calib11_allTx_ip11.h5") # N1 = ip11
RADAR_FILENAME_HDF5_2 = os.path.join(folder_path, "TDM_calib11_allTx_ip12.h5")
# RADAR_FILENAME_HDF5_1 = os.path.join(folder_path, "TDM_calib12_allTx_ip11.h5") # N2 = ip 12
# RADAR_FILENAME_HDF5_2 = os.path.join(folder_path, "TDM_calib12_allTx_ip12.h5")

calib_data = utility_functions.generate_calibration_data(os.path.join(folder_path, "N1_p00deg_target_rd_cell_values.npy"), os.path.join(folder_path, "N2_p00deg_target_rd_cell_values.npy"))


frame_idx = 15
min_range = 5/2
max_range = 6/2
zp_ft = 4
zp_st = 4
nr_chirps = int(128//6)*6
# load data
all_data_N1, measurement_parameters = utility_functions.loadFileData(RADAR_FILENAME_HDF5_1)
all_data_N2, measurement_parameters = utility_functions.loadFileData(RADAR_FILENAME_HDF5_2)
print(f"all_data shape: {all_data_N1.shape}")
tx0_data_N1 = all_data_N1[frame_idx,:,:nr_chirps:6]
tx1_data_N1 = all_data_N1[frame_idx,:,1:nr_chirps:6]
tx2_data_N1 = all_data_N1[frame_idx,:,2:nr_chirps:6]
tx3_data_N1 = all_data_N1[frame_idx,:,3:nr_chirps:6]
tx4_data_N1 = all_data_N1[frame_idx,:,4:nr_chirps:6]
tx5_data_N1 = all_data_N1[frame_idx,:,5:nr_chirps:6]
tx_split_data_N1 = np.vstack((tx0_data_N1[None], tx1_data_N1[None], tx2_data_N1[None], tx3_data_N1[None], tx4_data_N1[None], tx5_data_N1[None]))
tx0_data_N2 = all_data_N2[frame_idx,:,:nr_chirps:6]
tx1_data_N2 = all_data_N2[frame_idx,:,1:nr_chirps:6]
tx2_data_N2 = all_data_N2[frame_idx,:,2:nr_chirps:6]
tx3_data_N2 = all_data_N2[frame_idx,:,3:nr_chirps:6]
tx4_data_N2 = all_data_N2[frame_idx,:,4:nr_chirps:6]
tx5_data_N2 = all_data_N2[frame_idx,:,5:nr_chirps:6]
tx_split_data_N2 = np.vstack((tx0_data_N2[None], tx1_data_N2[None], tx2_data_N2[None], tx3_data_N2[None], tx4_data_N2[None], tx5_data_N2[None]))
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
N_st_fft = int(2**np.ceil(np.log2(N_st*zp_st)))
rd = np.fft.fftshift(np.fft.fft(rp_scaled*win_st_scaled[:,None],n=N_st_fft,axis=3),axes=3)
rd_scaled = rd / N_st

# data prep
rd_abs = np.mean(np.abs(rd),axis=2) # mean over rx per node
rd_abs = np.vstack((np.mean(rd_abs[0,0:3],axis=0,keepdims=True), np.mean(np.vstack((rd_abs[0,3:6],rd_abs[1,0:3])),axis=0,keepdims=True),
                                np.mean(rd_abs[1,3:6],axis=0,keepdims=True))) # mean over tx-rx combination with same round-trip-range

max_dB = 20 * np.log10(np.max(rd_abs))
data_dB = 20 * np.log10(rd_abs) - max_dB

# for static target scenarios do cfar detection on RDM range cross section at v=0
range_slice_dB = data_dB[:,data_dB.shape[1]//2] # shape: [rx-nodes, tx, range bins]


CFARConfig = cfar.CFARConfig(train_cells=8*zp_st, guard_cells=4*zp_st,
                                               pfa=1e-2, mode=cfar.CFARMode.CA)

range_slice_threshold = np.empty_like(range_slice_dB, dtype=float)
for idx in range(data_dB.shape[0]):
    range_slice_threshold[idx] = cfar.cfar_threshold(rd_abs[idx, data_dB.shape[1]//2], cfg=CFARConfig)

range_slice_threshold_dB = 20*np.log10(range_slice_threshold) - max_dB

data_max = np.nanmax(range_slice_dB)
data_min = np.nanmin(range_slice_dB)

# find peaks in allowed intervals
# peak_finiding_mask = (range_slice_dB > range_slice_threshold_dB) & (ranges > 2*min_range)[None,:] & (ranges < 2*max_range)[None,:]

peak_finiding_mask = (ranges > 2*min_range)[None,:] & (ranges < 2*max_range)[None,:]
peak_finiding_mask = np.repeat(peak_finiding_mask, range_slice_dB.shape[0], axis=0)

peak_idx_dict = utility_functions.findPeaksInIntervals(range_slice_dB,peak_finiding_mask)

print(peak_idx_dict)
highest_peak_values = {
    "N1_tx0_to_N1_rx0" : rd[0,0,0,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx0_to_N1_rx1" : rd[0,0,1,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx0_to_N1_rx2" : rd[0,0,2,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx0_to_N1_rx3" : rd[0,0,3,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx1_to_N1_rx0" : rd[0,1,0,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx1_to_N1_rx1" : rd[0,1,1,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx1_to_N1_rx2" : rd[0,1,2,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx1_to_N1_rx3" : rd[0,1,3,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx2_to_N1_rx0" : rd[0,2,0,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx2_to_N1_rx1" : rd[0,2,1,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx2_to_N1_rx2" : rd[0,2,2,data_dB.shape[1]//2,peak_idx_dict[0]],
    "N1_tx2_to_N1_rx3" : rd[0,2,3,data_dB.shape[1]//2,peak_idx_dict[0]],

    "N2_tx0_to_N1_rx0" : rd[0,3,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx0_to_N1_rx1" : rd[0,3,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx0_to_N1_rx2" : rd[0,3,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx0_to_N1_rx3" : rd[0,3,3,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx1_to_N1_rx0" : rd[0,4,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx1_to_N1_rx1" : rd[0,4,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx1_to_N1_rx2" : rd[0,4,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx1_to_N1_rx3" : rd[0,4,3,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx2_to_N1_rx0" : rd[0,5,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx2_to_N1_rx1" : rd[0,5,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx2_to_N1_rx2" : rd[0,5,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N2_tx2_to_N1_rx3" : rd[0,5,3,data_dB.shape[1]//2,peak_idx_dict[1]],

    "N1_tx0_to_N2_rx0" : rd[1,0,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx0_to_N2_rx1" : rd[1,0,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx0_to_N2_rx2" : rd[1,0,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx0_to_N2_rx3" : rd[1,0,3,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx1_to_N2_rx0" : rd[1,1,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx1_to_N2_rx1" : rd[1,1,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx1_to_N2_rx2" : rd[1,1,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx1_to_N2_rx3" : rd[1,1,3,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx2_to_N2_rx0" : rd[1,2,0,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx2_to_N2_rx1" : rd[1,2,1,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx2_to_N2_rx2" : rd[1,2,2,data_dB.shape[1]//2,peak_idx_dict[1]],
    "N1_tx2_to_N2_rx3" : rd[1,2,3,data_dB.shape[1]//2,peak_idx_dict[1]],

    "N2_tx0_to_N2_rx0" : rd[1,3,0,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx0_to_N2_rx1" : rd[1,3,1,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx0_to_N2_rx2" : rd[1,3,2,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx0_to_N2_rx3" : rd[1,3,3,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx1_to_N2_rx0" : rd[1,4,0,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx1_to_N2_rx1" : rd[1,4,1,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx1_to_N2_rx2" : rd[1,4,2,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx1_to_N2_rx3" : rd[1,4,3,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx2_to_N2_rx0" : rd[1,5,0,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx2_to_N2_rx1" : rd[1,5,1,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx2_to_N2_rx2" : rd[1,5,2,data_dB.shape[1]//2,peak_idx_dict[2]],
    "N2_tx2_to_N2_rx3" : rd[1,5,3,data_dB.shape[1]//2,peak_idx_dict[2]],
}

print("Measured phase values of all signal paths:")
rd_cell_values = np.empty([48], dtype=complex)
iteration_idx = 0
for signal_path, rd_cell_value in highest_peak_values.items():
    print(signal_path+f": {np.angle(rd_cell_value,deg=True)}°")

    rd_cell_values[iteration_idx] = rd_cell_value[0]
    iteration_idx = iteration_idx + 1

rd_cell_values = np.reshape(rd_cell_values,(4,3,4))
np.save("N2_p00deg_target_rd_cell_values.npy", rd_cell_values)

calibrated_rd_cell_values = rd_cell_values/calib_data
print(np.angle(calibrated_rd_cell_values, deg = True))

rd_cell_values_to_bf_N1 = calibrated_rd_cell_values[0].flatten()
rd_cell_values_to_bf_N2 = calibrated_rd_cell_values[3].flatten()
rd_cell_values_monostatic_to_bf = np.vstack((rd_cell_values_to_bf_N1,rd_cell_values_to_bf_N2))

# perform beamforming on 4rx subarrays
rd_cell_values_to_bf = np.reshape(calibrated_rd_cell_values,(12,4))
fig, ax = plt.subplots(1,1,num="bf_angle_profile")
for i in range(rd_cell_values_to_bf.shape[0]):
    angle_profile, incident_angle = utility_functions.beamformer(rd_cell_values_to_bf[i,:])
    ax.plot(incident_angle, angle_profile-np.max(angle_profile))

ax.set_xlim([-90,90])
ax.set_xlabel("Incident angle (°)")
ax.set_ylabel("Normalized Power in dB")
ax.grid()
ax.legend()

# perform beamforming on monostatic 3tx4rx virtual arrays
fig_va_mono, ax_va_mono = plt.subplots(1,1,num="bf_angle_profile_va")
for i in range(rd_cell_values_monostatic_to_bf.shape[0]):
    angle_profile, incident_angle = utility_functions.beamformer(rd_cell_values_monostatic_to_bf[i,:])
    ax_va_mono.plot(incident_angle, angle_profile-np.max(angle_profile), label=f"N{i+1}")

ax_va_mono.set_xlim([-90,90])
ax_va_mono.set_xlabel("Incident angle (°)")
ax_va_mono.set_ylabel("Normalized Power in dB")
ax_va_mono.grid()
ax_va_mono.legend()
plt.show()


# do plots
plot_profiles = np.concatenate((np.expand_dims(range_slice_dB, axis=1), np.expand_dims(range_slice_threshold_dB, axis=1), np.expand_dims(1*peak_finiding_mask-10,axis=1)),axis=1)

# range profiles
utility_functions.plotProfiles(ranges,plot_profiles,
             titles=["N1", "N1 <--> N2","N2"],
             x_axis={'min_value': 0, 'max_value' : ranges[-1], 'label' : 'Range (m)'},
             y_axis={'min_value': data_min, 'max_value' : data_max, 'label' : 'Normalized Power (dB)'})

# rd maps
utility_functions.plotMaps(np.reshape(data_dB,(-1,data_dB.shape[2],data_dB.shape[3])),
         titles=["N1tx0 --> N1", "N1tx1 --> N1", "N1tx2 --> N1", "N2tx0 --> N1", "N2tx1 --> N1", "N2tx2 --> N1",
                 "N1tx0 --> N2", "N1tx1 --> N2", "N1tx2 --> N2", "N2tx0 --> N2", "N2tx1 --> N2", "N2tx2 --> N2"],
         cbarLabel = 'Magnitude (dB)',
         x_axis={'min_value': -max_velocity, 'max_value' : max_velocity, 'label' : 'Velocity (m/s)'},
         y_axis={'min_value': 0, 'max_value' : ranges[-1], 'label' : 'Range (m)'})

to_plot = np.angle(max_values)*180/np.pi

fig_phase, ax_phase = plt.subplots(2,2,num="phase_progressions",constrained_layout=True,figsize=[2*5.3, 2*5.3 *0.8*1])
for row in range(ax_phase.shape[0]):
    for col in range(ax_phase.shape[1]):
        ax_phase[row,col].plot(to_plot[row,col,0],label="rx0")
        ax_phase[row,col].plot(to_plot[row,col,1],label="rx1")
        ax_phase[row,col].plot(to_plot[row,col,2],label="rx2")
        ax_phase[row,col].plot(to_plot[row,col,3],label="rx3")
        ax_phase[row,col].set_xlim([0,ranges[-1]])
        ax_phase[row,col].set_ylim([-180,180])
        ax_phase[row,col].legend()
        ax_phase[row,col].grid()
        ax_phase[row,col].set_xlabel("Ramp Index (1)")
        ax_phase[row,col].set_ylabel("Phase (°)")

ax_phase[0,0].set_title("N1tx0 --> N1")
ax_phase[0,1].set_title("N2tx0 --> N1")
ax_phase[1,0].set_title("N1tx0 --> N2")
ax_phase[1,1].set_title("N2tx0 --> N2")

max_mag = 20*np.log10(np.max(rp_mean))
rp_mean_flatter = np.reshape(rp_mean,(-1,rp_mean.shape[2]))
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles")
for i in range(rp_mean_flatter.shape[0]):
    ax_rp.plot(ranges,20*np.log10(rp_mean_flatter[i])-max_mag)

ax_rp.set_xlim([0,ranges[-1]])
#ax_rp.legend()
ax_rp.grid()
ax_rp.set_xlabel("Round Trip Range (m)")
ax_rp.set_ylabel("Normalized Power (dB)")
plt.show()

