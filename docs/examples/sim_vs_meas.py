# interpreter skradarMod
import sys, os  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import skradar
from scipy.constants import speed_of_light as c0
import h5py
import time
import scipy

from matplotlib import rcParams
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params) 

plt.ioff()
np.random.seed(1)

# load radar configs
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_single_static_20250930_13-33-19_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\single_static_20250930_13-35-59_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_282mm_20250924_13-25-49_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_two_targets_20250924_13-38-41_1.h5"
process_measurement = True
frame_idx =20# 50
min_target_range = 1

# simulation parameters
target_dist = 20
calib_target_dist = 3.8

# processing settings
zp_fact_range = 4*2
zp_fact_doppler = 4*2
calib_peak_tolerance = 0.1

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
    NrChn = 2#int(f.attrs["NrChn"][0])

    if process_measurement:
        data = np.empty([NrChn, N_s, N_f])

        print(f"Loaded measurement with {int(f["Chn1"].shape[0]/N_s)} frames")

        # get slow slow and fast time data for every channel of specified frame
        all_data = np.empty([int(f["Chn1"].shape[0]/N_s),NrChn, N_s, N_f])
        for Na in range(1,NrChn+1):
            ch_data = f['Chn%d'%Na]
            ch_data_reshaped = np.reshape(ch_data, (-1,ch_data.chunks[0],ch_data.chunks[1]))
            all_data[:,Na-1,:,:] = ch_data_reshaped

        data = all_data[frame_idx]

    # signal parameters
    B_sw = f1 - f0  # chirp bandwidth
    k = B_sw / T_sw
    B_samp = k * (N_f - 1) / fs_f  # signal bandwidth while sampling
    fc = f0 + B_sw / 2  # center frequency
    lbd = c0 / fc
    Ts_s = TRampTot  # slow-time sampling interval

lambd = c0 / fc
v_max = lambd / (4 * Ts_s)

# simulated radar sensor (MIMO with 2x 1Tx-1Rx)
tx_pos = np.array([[0, 0], [0, 0], [0, 0]])
rx_pos = np.array([[0, 0], [0, 0], [0, 0]])
radar_pos = np.array([[0], [0], [0]])

# lo definitions
lo_error_spec = np.array(
    [[0,fs_f / 500], [0,10e-5 * B_sw]]
)  # specifications of available LOs (delta fc, delta B)
tx_lo = np.array(
    [[1, 0], [0, np.pi]]
)  # used LO and chirp modulation (lo idx, phase shift)
rx_lo = np.array([0, 1])  # used LO

# phase noise definition
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) / 2
L_dB_vec = np.array([-65, -65, -85, -115, -115]) #* 10

# setup test scene
radar = skradar.FMCWRadar(
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
    lo_spec=lo_error_spec,
    tx_lo=tx_lo,
    rx_lo=rx_lo,
    tx_ant_gains=np.array([15, 15]),
    rx_ant_gains=np.array([10, 10]),
    pos=radar_pos,
    name="First radar",
    if_real=False,
)

# calibration/static target
calib_target_pos = np.array([[0], [calib_target_dist], [0]])
calib_target = skradar.Target(rcs=10, pos=calib_target_pos, name="Calib target, 10 sqm")

#moving target
target_pos2 = np.array([[0], [target_dist], [0]])
target_v2 = np.array([[0], [2], [0]])
target2 = skradar.Target(rcs=10, pos=target_pos2,vel=target_v2, name="Moving target, 10 sqm")

scene = skradar.Scene([radar], [calib_target, target2])

radar.sim_chirps()

if process_measurement:
    if data.ndim == 3:  # add axis that scikit processing can be used
        radar.s_if = data[None,:,:,:]
        radar.s_if_noisy = data[None,:,:,:]
    else:
        print("Invalid measurement data shape")
else:
    # radar.apply_errors_unmerged()
    radar.merge_mimo()

# extract mimo into multiple transmitters here before errors can be applied
# split signal into slow time frequency bands

# apply errors
radar.apply_errors()

# processing
radar.range_compression(zp_fact=zp_fact_range)
radar.doppler_processing(zp_fact=zp_fact_doppler, win_doppler="hann")

target_dists = radar.ranges / 2  # halve values to account for round-trip ranges
target_dists_plot = target_dists[: len(radar.ranges) // 2]
calib_peak_search_idx = np.where(np.abs(target_dists - calib_target_dist) < calib_peak_tolerance)[0]

# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_noisy_scaled = 1 / (np.sqrt(2)) * radar.rp_noisy
rp_noisy_abs_chirp_mean = np.mean(np.abs(rp_noisy_scaled), axis=2)
rp_scaled = 1 / (np.sqrt(2)) * radar.rp
rp_abs_chirp_mean = np.mean(np.abs(rp_scaled), axis=2)

# find clibration target peak
peak_pos_idx = np.argmax(rp_noisy_abs_chirp_mean[:,:,calib_peak_search_idx],axis=2) + calib_peak_search_idx[0]
print(f"Calib target pos = {np.round(target_dists[peak_pos_idx], 2)}m")

CFARConfig = skradar.detection.cfar.CFARConfig(train_cells=8*zp_fact_range, guard_cells=4*zp_fact_range,
                                               pfa=1e-4, mode=skradar.detection.cfar.CFARMode.CAGO)

rp_noisy_plot_dB = 20 * np.log10(rp_noisy_abs_chirp_mean)[:,:,: len(radar.ranges) // 2]
rp_plot_dB = 20 * np.log10(rp_abs_chirp_mean)[:,:,: len(radar.ranges) // 2]
threshold = np.zeros_like(rp_noisy_plot_dB)
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles",figsize=[10,5])
for tx_idx in range(rp_noisy_scaled.shape[0]):
    for rx_idx in range(1):#rp_noisy_scaled.shape[1]):
        threshold[tx_idx,rx_idx] = 20 * np.log10(skradar.detection.cfar.cfar_threshold(rp_noisy_abs_chirp_mean[tx_idx,rx_idx,: len(radar.ranges) // 2], cfg = CFARConfig))
        target_peaks = scipy.signal.find_peaks(rp_noisy_plot_dB[tx_idx, rx_idx], height = threshold[tx_idx,rx_idx], distance=2*zp_fact_range)[0]
        target_peaks = target_peaks[target_dists_plot[target_peaks]>min_target_range]
        ax_rp.plot(
            target_dists_plot, rp_plot_dB[tx_idx, rx_idx], label=f"tx{tx_idx}, rx{rx_idx}"
        )
        ax_rp.plot(
            target_dists_plot, rp_noisy_plot_dB[tx_idx, rx_idx], label=f"tx{tx_idx}, rx{rx_idx} (erroneous)"
        )
        # ax_rp.plot(
        #     target_dists_plot, threshold[tx_idx,rx_idx] - np.max(rp_plot_dB), label=f"tx{tx_idx}, rx{rx_idx}"
        # )
        # ax_rp.vlines(target_dists_plot[target_peaks],ymin=-10,ymax=5)

np.save("error", np.append(target_dists_plot, rp_noisy_plot_dB))

ax_rp.legend()
ax_rp.grid(True)
ax_rp.set_xlabel("Range (m)")
ax_rp.set_ylabel("Magnitude (dBV)")
ax_rp.set_xlim([0, target_dists_plot[-1]/5])

figure_name ="test_rp_comparison"
plt.savefig("python_plots/"+figure_name+".eps",
            papertype = 'a4',
            bbox_inches = 'tight'
            )

# plt.show()

print(radar.rd_noisy.shape)
plt.figure("rd_map")
plt.imshow(
    20 * np.log(np.abs(radar.rd_noisy[0, 0, :, : N_f * zp_fact_range // 2])),
    aspect="auto",origin="lower",extent=[0,target_dists_plot[-1],-v_max,v_max]
)
plt.ylabel("v in m/s")
plt.xlabel("range in m")
plt.show()




# # process all frames
# peak_range = np.empty([all_data.shape[0]])
# start_time = time.time()
# for frame_idx in range(all_data.shape[0]):
#     if all_data.ndim == 4:
#         radar.s_if = all_data[frame_idx,None,:,:,:]
#         radar.s_if_noisy = all_data[frame_idx,None,:,:,:]
#     elif all_data.ndim == 5:
#         radar.s_if = all_data[frame_idx]
#         radar.s_if_noisy = all_data[frame_idx]

#     radar.range_compression(zp_fact=zp_fact_range)
#     start_idx = np.where(radar.ranges > 2)[0][0]
#     range_profile = np.mean(np.abs(radar.rp),axis=(1,2))[0]
#     peak_range[frame_idx] = radar.ranges[np.argmax(range_profile[start_idx:radar.rp.shape[-1]//2]+start_idx, axis=-1)]/2
#     print(f"Range = {peak_range[frame_idx]}m, time since start = {np.round(time.time()-start_time,3)}")
    
# plt.figure("peak_range")
# plt.plot(peak_range)
# plt.show()