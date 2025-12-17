# interpreter skradarMod
import sys, os  # add path where skradar is located, relative import of skradar does not work
import copy

import spkit as sp

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import skradar
from scipy.constants import speed_of_light as c0
from scipy.signal import ShortTimeFFT
import h5py
import time
import scipy

from utility_functions import detect_target_and_spurs, get_info, plot_RD_maps

from matplotlib import rcParams
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 

plt.ioff()
np.random.seed(1)
#compare_profiles()
# reordered by ipaddr, testing chirp mod
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\testing\ip12_tx1_000_1.h5"



RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\qpsk_qmono_tx_1.h5"

# measurements 27.11 single transmit 90 degree steps
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\ip11_tx1_90deg_corr_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx1_88dB_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx1_first_iteration_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx2_84dB.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx2_first_iteration.h5"

RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx0_95dB_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx0_first_iteration.h5"

# prog_file = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx1_const_progress.npy"
# prog_file = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx0_const_progress.npy"
# prog_file = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_imbalance\tx2_const_progress.npy"
# phase_progression = np.load(prog_file)
# phase_progression[:,1]=phase_progression[:,1]-90
# phase_progression[:,2]=phase_progression[:,2]-180
# phase_progression[:,3]=phase_progression[:,3]-270
# fig,ax = plt.subplots(1,1,num="estimated_error")
# ax.plot(-(phase_progression-phase_progression[:,0][:,None]))
# ax.set_xlabel("Iteration idx")
# ax.set_ylabel("Applied correction offset (°)")
# ax.legend(["0 deg","90 deg","180 deg","270 deg"])
# ax.set_xlim([0,10])
# ax.grid()
# plt.show()

# test bistatic pole
# RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\qpsk_tx0_pole4mp_driving_wood20251203_08-10-30_1.h5" # unf only one sensor activated
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\setup_demo\TDM_tx0_poleRight20251210_09-32-51_1.h5"

# RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_TDM_rx_ip11.h5"
# RADAR_FILENAME_HDF5_2 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\new_TDM_rx_ip12.h5"

# newer iterations
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\test_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\phase_step_investigation\test_firstIt_1.h5"


evaluate_measurement = True
frame_idx =0 #10
min_target_range = 1 # omit detection of leakage peak
nr_of_targets = 1 # number of peak positions to detect
range_zoom = 2#5
axis_frequency = False  # show IF frequency on range profile
calib_target_dist = 4 # 2.93 #  5.3 # used also for simulation
calib_range_peak_tolerance = 2 # 0.5 # in m, depends on target position accuracy and expected frequency offsets
second_target_dist = 20

# general processing settings
zp_fact_doppler = 4*2
zp_fact_range = 4*2
chirp_idx = 255//2

# chirp z transform parameters
N_chirp_transform_bins = 2*128
block_size = 128

# calibration/static target
calib_target_pos = np.array([[0], [calib_target_dist], [0]])
calib_target = skradar.Target(rcs=10, pos=calib_target_pos, name="Calib target, 10 sqm")

#moving target
target_pos2 = np.array([[0], [second_target_dist], [0]])
target_v2 = np.array([[0], [2], [0]])
target2 = skradar.Target(rcs=10, pos=target_pos2,vel=target_v2, name="Moving target, 10 sqm")

# load radar data from file
with h5py.File(RADAR_FILENAME_HDF5_1, "r") as f:
    f0 = f.attrs["fStrt"][0]  # start freqeuncy
    f1 = f.attrs["fStop"][0]  # stop freqeuncy
    fs_f = f.attrs["fs"][0]  # fast-time sampling rate
    N_f = int(f.attrs["Cfg_N"][0])  # number of fast-time samples
    N_s = int(f.attrs["Cfg_Np"][0])  # number of slow-time samples
    T_sw = f.attrs["Cfg_TRampUp"][0]
    TRampTot = (
        f.attrs["Cfg_Tp0"]
        + f.attrs["Cfg_TRampUp"][0]
        + f.attrs["Cfg_Tp1"]
        + f.attrs["Cfg_TRampDo"]
        + f.attrs["Cfg_Tp2"]
    )
    Tframe_min = TRampTot * N_s
    NrChn = int(f.attrs["NrChn"][0])

    data = np.empty([NrChn, N_s, N_f])

    print(f"Loaded measurement with {int(f["Chn1"].shape[0]/N_s)} frames")
    print(f"N_f = {N_f}, N_s = {N_s}")
    # get slow slow and fast time data for every channel of specified frame
    all_data = np.empty([int(f["Chn1"].shape[0]/N_s),NrChn, N_s, N_f])
    for Na in range(1,NrChn+1):
        ch_data = f['Chn%d'%Na]
        ch_data_reshaped = np.reshape(ch_data, (-1,ch_data.chunks[0],ch_data.chunks[1]))
        all_data[:,Na-1,:,:] = ch_data_reshaped

    
    # signal parameters
    B_sw = f1 - f0  # chirp bandwidth
    k = B_sw / T_sw
    B_samp = k * (N_f - 1) / fs_f  # signal bandwidth while sampling
    fc = f0 + B_sw / 2  # center frequency
    lbd = c0 / fc
    Ts_s = TRampTot  # slow-time sampling interval



lambd = c0 / fc
v_max = lambd / (4 * Ts_s)

# simulated radar sensor (MIMO with 1/3Tx-1Rx)
tx_pos = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) # for one spurs-only velocity cell
tx_pos = np.array([[0], [0], [0]]) # for single transmitter spurs
rx_pos = np.array([[0], [0], [0]])
radar_pos = np.array([[0], [0], [0]])

# three transmitter with qpsk
tx_lo = np.array(
    [[0, 0, 0], [0, np.pi/2, 2*np.pi/2]]
)  # used LO and chirp modulation (lo idx, phase shift)

# # two transmitter with bpsk
tx_lo = np.array(
    [[0], [np.pi/2]]
)  # used LO and chirp modulation (lo idx, phase shift)
for shift in tx_lo[1]:
    pass


rx_lo = np.array([0])  # used LO

# phase noise definition
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) #/ 2
L_dB_vec = np.array([-65, -65, -85, -115, -115]) * 10

# generate radar with parameters from measurement for simulation
radar_simulated = skradar.FMCWRadar(
    B=B_samp,
    fc=fc,
    N_f=N_f,
    T_f=1 / fs_f,
    T_s=Ts_s,
    L_freqs_vec=L_freqs_vec,
    L_dB_vec=L_dB_vec,
    N_s=N_s,
    tx_pos=tx_pos,
    rx_pos=rx_pos,
    tx_lo=tx_lo,
    rx_lo=rx_lo,
    win_range='hann',
    win_doppler='hann',
    tx_ant_gains=np.array([15]),#, 15
    rx_ant_gains=np.array([10,10]),
    pos=radar_pos,
    name="simulated radar",
    if_real=True,
    coherent_pn=True,    # for comparsion with measurement where no phase noise is added in simulation
    chirp_modulation = "bpsk",
)



# theoretical maximum range
max_range = (radar_simulated.N_f - 1) * c0 / (4 * radar_simulated.B)

# peak position in range profile where calibration target peak is expected, incorporating expected frequency errors
d_0 = calib_target_dist - calib_range_peak_tolerance
d_1 = calib_target_dist + calib_range_peak_tolerance
calib_target_freq = (calib_target_dist / max_range) / (2 * radar_simulated.T_f)
calib_frequency_peak_tolerance = (calib_range_peak_tolerance / max_range) / (2 * radar_simulated.T_f)
f_0 = calib_target_freq - calib_frequency_peak_tolerance
f_1 = calib_target_freq + calib_frequency_peak_tolerance

# copy to store and process measurement data in parallel
radar_measured = copy.deepcopy(radar_simulated)


data = all_data[frame_idx]
if data.ndim == 3:  # add axis that scikit processing can be used
    radar_measured.s_if = data[None,:,:,:]
    radar_measured.s_if_noisy = data[None,:,:,:]
else:
    print("Invalid measurement data shape")

# generate scene
# scene = skradar.Scene([radar_simulated, radar_measured], [calib_target, target2])
scene = skradar.Scene([radar_simulated, radar_measured], [calib_target])

# do simulation
radar_simulated.sim_chirps()

# define imbalance vector
imbalance_error_vector = np.array([0.0, 7.0, 16.0, -8.0])
remaining_imbalance_error_vector = np.copy(imbalance_error_vector)
remaining_imbalance_error_vector_all = np.copy(remaining_imbalance_error_vector)

# assign to simulated signal
radar_simulated.s_if_noisy[0,:,0::4,:] = radar_simulated.s_if_noisy[0,:,0::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[0]/180)
radar_simulated.s_if_noisy[0,:,1::4,:] = radar_simulated.s_if_noisy[0,:,1::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[1]/180)
radar_simulated.s_if_noisy[0,:,2::4,:] = radar_simulated.s_if_noisy[0,:,2::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[2]/180)
radar_simulated.s_if_noisy[0,:,3::4,:] = radar_simulated.s_if_noisy[0,:,3::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[3]/180)
radar_simulated.merge_mimo()


# calculate rd map 
radar_simulated.range_compression(zp_fact=zp_fact_range)
radar_simulated.doppler_processing(zp_fact=zp_fact_doppler)
radar_measured.range_compression(zp_fact=zp_fact_range)
radar_measured.doppler_processing(zp_fact=zp_fact_doppler)

# estimate phase error
estimated_angle_error = detect_target_and_spurs(radar_simulated.rd_noisy[0], True)
estimated_angle_error_progression = np.zeros_like(estimated_angle_error)
estimated_angle_error_progression = np.vstack((estimated_angle_error_progression,estimated_angle_error))
detect_target_and_spurs(radar_measured.rd[0], True)

# correct error in simulated signal
nr_of_iterations = 10
for iteration_idx in range(nr_of_iterations):

    if estimated_angle_error.shape[0]==4:
        
        # integrator
        remaining_imbalance_error_vector = remaining_imbalance_error_vector - 0.7 * estimated_angle_error

    elif estimated_angle_error.shape[0]==1:
        print(f"Estimated phase imbalance: {imbalance_error_vector-remaining_imbalance_error_vector}°")
        break

    remaining_imbalance_error_vector_all = np.vstack((remaining_imbalance_error_vector_all,remaining_imbalance_error_vector))
    # create new measurement data
    radar_simulated.sim_chirps()
    radar_simulated.merge_mimo()

    # add updated error
    radar_simulated.s_if_noisy[0,:,0::4,:] = radar_simulated.s_if_noisy[0,:,0::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[0]/180)
    radar_simulated.s_if_noisy[0,:,1::4,:] = radar_simulated.s_if_noisy[0,:,1::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[1]/180)
    radar_simulated.s_if_noisy[0,:,2::4,:] = radar_simulated.s_if_noisy[0,:,2::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[2]/180)
    radar_simulated.s_if_noisy[0,:,3::4,:] = radar_simulated.s_if_noisy[0,:,3::4,:]*np.exp(1j*np.pi*remaining_imbalance_error_vector[3]/180)

    # calculate rd map 
    radar_simulated.range_compression(zp_fact=zp_fact_range)
    radar_simulated.doppler_processing(zp_fact=zp_fact_doppler)

    if iteration_idx == 9:
        estimated_angle_error = detect_target_and_spurs(radar_simulated.rd_noisy[0], True)
    else:
        estimated_angle_error = detect_target_and_spurs(radar_simulated.rd_noisy[0], False)

fig,ax = plt.subplots(1,1,num="remaining_error")
ax.plot(remaining_imbalance_error_vector_all)
ax.set_xlabel("Iteration idx")
ax.set_ylabel("Remaining error (°)")
ax.legend(["0 deg","90 deg","180 deg","270 deg"])
ax.set_xlim([0,remaining_imbalance_error_vector_all.shape[0]-1])
ax.grid()
plt.savefig("remaining_error.pdf",
            bbox_inches = 'tight'
            )

fig,ax = plt.subplots(1,1,num="estimated_imbalance_error")
ax.plot(remaining_imbalance_error_vector_all-imbalance_error_vector[None,:])
ax.set_xlabel("Iteration idx")
ax.set_ylabel("Applied correction offset (°)")
ax.legend(["0 deg","90 deg","180 deg","270 deg"])
ax.set_xlim([0,remaining_imbalance_error_vector_all.shape[0]-1])
ax.grid()
plt.savefig("estimated_imbalance_error.pdf",
            bbox_inches = 'tight'
            )
plt.show()

# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_simulated_noisy_scaled = 1 / (np.sqrt(2)) * radar_simulated.rp_noisy
rp_simulated_scaled = 1 / (np.sqrt(2)) * radar_simulated.rp
radar_measured.range_compression(zp_fact=zp_fact_range)
radar_measured.doppler_processing(zp_fact=zp_fact_doppler)
rp_measured_noisy_scaled = 1 / (np.sqrt(2)) * radar_measured.rp_noisy
rp_measured_scaled = 1 / (np.sqrt(2)) * radar_measured.rp
frequency_axis = np.linspace(0,fs_f/2,radar_simulated.N_f*zp_fact_range//2,endpoint=False)

# range axis and peak search range
target_dists = radar_simulated.ranges / 2  # halve values to account for round-trip ranges
resolution = target_dists[1] * zp_fact_range
x_axis_plot = target_dists[: len(radar_simulated.ranges) // 2]
if axis_frequency:
    x_axis_plot = frequency_axis*1e-6

calib_peak_search_idx = np.where(np.abs(target_dists - calib_target_dist) < calib_range_peak_tolerance)[0]
min_peak_search_idx = np.where(target_dists >= min_target_range)[0][0]
# find calibration target peak


rp_simulated_noisy_abs_chirp_mean = np.abs(rp_simulated_noisy_scaled)
rp_simulated_abs_chirp_mean = np.abs(rp_simulated_scaled)
rp_measured_noisy_abs_chirp_mean = np.abs(rp_measured_noisy_scaled)
rp_measured_abs_chirp_mean = np.abs(rp_measured_scaled)

if evaluate_measurement:
    peak_pos_idx = np.argmax(rp_measured_noisy_abs_chirp_mean[:,:,:,calib_peak_search_idx],axis=3) + calib_peak_search_idx[0]
else:
    peak_pos_idx = np.argmax(rp_simulated_noisy_abs_chirp_mean[:,:,:,calib_peak_search_idx],axis=3) + calib_peak_search_idx[0]


#################################################################################################################################################################
# plots

# range profile result
chirp_idx = -1
if chirp_idx < 0:
    rp_simulated_noisy_plot_dB = 20 * np.log10(np.mean(rp_simulated_noisy_abs_chirp_mean[:,:,:,: len(radar_simulated.ranges) // 2],axis=2,keepdims=False))
    rp_simulated_plot_dB = 20 * np.log10(np.mean(rp_simulated_abs_chirp_mean[:,:,:,: len(radar_simulated.ranges) // 2],axis=2,keepdims=False))
    rp_measured_noisy_plot_dB = 20 * np.log10(np.mean(rp_measured_noisy_abs_chirp_mean[:,:,:,: len(radar_simulated.ranges) // 2],axis=2,keepdims=False))
    rp_measured_plot_dB = 20 * np.log10(np.mean(rp_measured_abs_chirp_mean[:,:,:,: len(radar_simulated.ranges) // 2],axis=2,keepdims=False))
else:
    rp_simulated_noisy_plot_dB = 20 * np.log10(rp_simulated_noisy_abs_chirp_mean)[:,:,chirp_idx,: len(radar_simulated.ranges) // 2]
    rp_simulated_plot_dB = 20 * np.log10(rp_simulated_abs_chirp_mean)[:,:,chirp_idx,: len(radar_simulated.ranges) // 2]
    rp_measured_noisy_plot_dB = 20 * np.log10(rp_measured_noisy_abs_chirp_mean)[:,:,chirp_idx,: len(radar_simulated.ranges) // 2]
    rp_measured_plot_dB = 20 * np.log10(rp_measured_abs_chirp_mean)[:,:,chirp_idx,: len(radar_simulated.ranges) // 2]

fig_rp, (ax_rp_noisy_simulated, ax_rp_noisy_measured) = plt.subplots(2,1,num="range_profiles",figsize=[8,6],layout="compressed")
fig_rp_sim, ax_rp_noisy_sim = plt.subplots(1,1,num="range_profile_sim",figsize=[8,4],layout="compressed")
fig_rp_meas, ax_rp_noisy_meas = plt.subplots(1,1,num="range_profile_meas",figsize=[8,4],layout="compressed")
for tx_idx in range(rp_simulated_noisy_scaled.shape[0]):
    for rx_idx in range(rp_simulated_noisy_scaled.shape[1]):
        
        # peak detection
        peak_indices_simulated, peak_dict_simulated = scipy.signal.find_peaks(rp_simulated_noisy_plot_dB[tx_idx, rx_idx],height=(None, None))
        peak_heights_simulated = peak_dict_simulated['peak_heights'][peak_indices_simulated>=min_peak_search_idx]
        peak_indices_simulated = peak_indices_simulated[peak_indices_simulated>=min_peak_search_idx]
        highest_peak_indices_simulated = peak_indices_simulated[np.argsort(peak_heights_simulated)[-nr_of_targets:]]
        peak_indices_measured, peak_dict_measured = scipy.signal.find_peaks(rp_measured_noisy_plot_dB[tx_idx, rx_idx],height=(None, None))
        peak_heights_measured = peak_dict_measured['peak_heights'][peak_indices_measured>=min_peak_search_idx]
        peak_indices_measured = peak_indices_measured[peak_indices_measured>=min_peak_search_idx]
        highest_peak_indices_measured = peak_indices_measured[np.argsort(peak_heights_measured)[-nr_of_targets:]]

        # plot
        if rp_simulated_noisy_scaled.shape[0] == 1:
            label = f"rx{rx_idx}"
        else:
            label = f"tx{tx_idx}, rx{rx_idx}"
        ax_rp_noisy_simulated.plot(
            x_axis_plot, rp_simulated_noisy_plot_dB[tx_idx, rx_idx]- np.max(rp_simulated_noisy_plot_dB), label=label
        )
        ax_rp_noisy_simulated.plot(
            x_axis_plot[highest_peak_indices_simulated], rp_simulated_noisy_plot_dB[tx_idx,rx_idx][highest_peak_indices_simulated] - np.max(rp_simulated_noisy_plot_dB), "xk"
        )
        ax_rp_noisy_measured.plot(
            x_axis_plot, rp_measured_noisy_plot_dB[tx_idx,rx_idx] - np.max(rp_measured_noisy_plot_dB), label=label
        )
        ax_rp_noisy_measured.plot(
            x_axis_plot[highest_peak_indices_measured], rp_measured_noisy_plot_dB[tx_idx,rx_idx][highest_peak_indices_measured] - np.max(rp_measured_noisy_plot_dB), "xk"
        )

        ax_rp_noisy_meas.plot(
            x_axis_plot, rp_measured_noisy_plot_dB[tx_idx, rx_idx]- np.max(rp_measured_noisy_plot_dB), label=label
        )
        ax_rp_noisy_meas.plot(
            x_axis_plot[highest_peak_indices_measured], rp_measured_noisy_plot_dB[tx_idx,rx_idx][highest_peak_indices_measured] - np.max(rp_measured_noisy_plot_dB), "xk"
        )
        ax_rp_noisy_sim.plot(
            x_axis_plot, rp_simulated_noisy_plot_dB[tx_idx, rx_idx]- np.max(rp_simulated_noisy_plot_dB), label=label
        )
        ax_rp_noisy_sim.plot(
            x_axis_plot[highest_peak_indices_simulated], rp_simulated_noisy_plot_dB[tx_idx,rx_idx][highest_peak_indices_simulated] - np.max(rp_simulated_noisy_plot_dB), "xk"
        )

np.save("error", np.append(x_axis_plot, rp_simulated_noisy_plot_dB))

ax_rp_noisy_simulated.legend(loc='upper right')
ax_rp_noisy_simulated.grid(True)
ax_rp_noisy_simulated.set_title("Simulated")
ax_rp_noisy_simulated.set_ylabel("Normalized Power (dB)")
ax_rp_noisy_simulated.set_xlim([0,x_axis_plot[-1]/range_zoom])
ax_rp_noisy_measured.legend(loc='upper right')
ax_rp_noisy_measured.grid(True)
ax_rp_noisy_measured.set_title("Measured")
ax_rp_noisy_measured.set_ylabel("Normalized Power (dB)")
ax_rp_noisy_measured.set_xlim([0,x_axis_plot[-1]/range_zoom])

ax_rp_noisy_meas.legend(loc='upper right')
ax_rp_noisy_meas.grid(True)
ax_rp_noisy_meas.set_title("Measured")
ax_rp_noisy_meas.set_ylabel("Normalized Power (dB)")
ax_rp_noisy_meas.set_xlim([0,x_axis_plot[-1]/range_zoom])
ax_rp_noisy_sim.legend(loc='upper right')
ax_rp_noisy_sim.grid(True)
ax_rp_noisy_sim.set_title("Simulated")
ax_rp_noisy_sim.set_ylabel("Normalized Power (dB)")
ax_rp_noisy_sim.set_xlim([0,x_axis_plot[-1]/range_zoom])

if axis_frequency:
    ax_rp_noisy_simulated.set_xlabel("Frequency (MHz)")
    ax_rp_noisy_meas.set_xlabel("Frequency (MHz)")
    ax_rp_noisy_sim.set_xlabel("Frequency (MHz)")
    ax_rp_noisy_measured.set_xlabel("Frequency (MHz)")
else:
    ax_rp_noisy_simulated.set_xlabel("Range (m)")
    ax_rp_noisy_meas.set_xlabel("Range (m)")
    ax_rp_noisy_sim.set_xlabel("Range (m)")
    ax_rp_noisy_measured.set_xlabel("Range (m)")

plt.savefig("range_profile_sim.eps",
            papertype = 'a4',
            bbox_inches = 'tight'
            )

plt.figure("range_profile_meas")
plt.savefig("range_profile_meas.eps",
            papertype = 'a4',
            bbox_inches = 'tight'
            )

plt.figure("range_profiles")
plt.savefig("range_profiles.eps",
            papertype = 'a4',
            bbox_inches = 'tight'
            )
plt.show()


rd_fig, (rd_sim_ax, rd_meas_ax) = plt.subplots(1,2,figsize=[10,4],num="rd_maps")
rd_sim_ax.imshow(
    20 * np.log10(np.abs(radar_simulated.rd_noisy[0, 0, :, : N_f * zp_fact_range // 2])).T,
    aspect="auto",origin="lower",extent=[-v_max,v_max,0,x_axis_plot[-1]]
)
rd_sim_ax.set_xlabel("v in m/s")
rd_sim_ax.set_ylabel("range in m")
rd_sim_ax.set_title("Simulation")
rd_meas_ax.imshow(
    20 * np.log10(np.abs(radar_measured.rd_noisy[0, 0, :, : N_f * zp_fact_range // 2])).T,
    aspect="auto",origin="lower",extent=[-v_max,v_max,0,x_axis_plot[-1]]
)
rd_meas_ax.set_xlabel("v in m/s")
rd_meas_ax.set_ylabel("range in m")
rd_meas_ax.set_title("Measurement")

plt.savefig("rd_maps.pdf",
            bbox_inches = 'tight'
            )
plt.show()