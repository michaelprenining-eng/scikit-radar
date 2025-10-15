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
from scipy.signal.windows import hann

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
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_single_static_20250930_13-33-19_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\hpf_18_single_static_20251002_18-18-46_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\single_static_20250930_13-35-59_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_two_targets_20251002_18-49-20_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\hpf_18_far_single_static_20251002_18-40-31_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_282mm_20250924_13-25-49_1.h5"
#RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_two_targets_20250924_13-38-41_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\bpsk_far_single_static_20251006_14-20-25_1.h5"

frame_idx =20 # 39 same range bin
min_target_range = 1 # omit detection of leakage peak
nr_of_targets = 1 # number of peak positions to detect
range_zoom = 1#5
axis_frequency = True  # show IF frequency on range profile
calib_target_dist = 5.3 # used also for simulation
calib_range_peak_tolerance = 2 # 0.5 # in m, depends on target position accuracy and expected frequency offsets
second_target_dist = 20

# general processing settings
zp_fact_range = 4*2
zp_fact_doppler = 4*2
single_chirp_analysis = True
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
    NrChn = 2#int(f.attrs["NrChn"][0])

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
    [[0,1 * fs_f / 200], [0,1*10e-5 * B_sw]]
)  # specifications of available LOs (delta fc, delta B)

# two transmitter with bpsk
tx_lo = np.array(
    [[1, 0], [0, np.pi]]
)  # used LO and chirp modulation (lo idx, phase shift)

rx_lo = np.array([0, 1])  # used LO

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
    lo_spec=lo_error_spec,
    tx_lo=tx_lo,
    rx_lo=rx_lo,
    win_range='hann',
    win_doppler='hann',
    tx_ant_gains=np.array([15,15]),#, 15
    rx_ant_gains=np.array([10,10]),
    pos=radar_pos,
    name="simulated radar",
    if_real=True,
    coherent_pn=True    # for comparsion with measurement where no phase noise is added in simulation
)

# copy to store and process measurement data in parallel
radar_measured = copy.deepcopy(radar_simulated)
if data.ndim == 3:  # add axis that scikit processing can be used
    radar_measured.s_if = data[None,:,:,:]
    radar_measured.s_if_noisy = data[None,:,:,:]
else:
    print("Invalid measurement data shape")

# theoretical maximum range
max_range = (radar_simulated.N_f - 1) * c0 / (4 * radar_simulated.B)

# peak position in range profile where calibration target peak is expected, incorporating expected frequency errors
d_0 = calib_target_dist - calib_range_peak_tolerance
d_1 = calib_target_dist + calib_range_peak_tolerance
calib_target_freq = (calib_target_dist / max_range) / (2 * radar_simulated.T_f)
calib_frequency_peak_tolerance = (calib_range_peak_tolerance / max_range) / (2 * radar_simulated.T_f)
f_0 = calib_target_freq - calib_frequency_peak_tolerance
f_1 = calib_target_freq + calib_frequency_peak_tolerance

# generate scene
# scene = skradar.Scene([radar_simulated, radar_measured], [calib_target, target2])
scene = skradar.Scene([radar_simulated, radar_measured], [calib_target])

# do simulation
radar_simulated.sim_chirps()
radar_simulated.merge_mimo()

# apply errors
radar_simulated.apply_errors_bpsk()
radar_measured.apply_errors_bpsk()

# extract mimio for tx dependent error estimation
radar_simulated.extract_mimo()
radar_measured.extract_mimo()

############################################################################################################
# per chirp frequency evaluation
f_axis_chirp_transform = np.linspace(f_0,f_1,N_chirp_transform_bins)
f_bin_size_chirp_transform = f_axis_chirp_transform[1]-f_axis_chirp_transform[0]

# bandwitdth error estimation
start_time = time.time()
blocks_per_chirp, transformed, transformed_noisy = radar_simulated.chirp_transform(f_0,f_1,N_chirp_transform_bins,block_size=block_size,win="hann")
nn, chirp_transformed, chirp_transformed_noisy = radar_simulated.chirp_transform(f_0,f_1,N_chirp_transform_bins,block_size=radar_measured.N_f,win="hann")
print(f"chirp z transform range: f0 = {np.round(f_0*1e-6,2)}MHz, f1 = {np.round(f_1*1e-6,2)}MHz, {blocks_per_chirp} blocks of size {block_size} per " + \
      f"chirp. Frequency bin size: {f_bin_size_chirp_transform*1e-3}kHz. Duration: {time.time() - start_time}")

transformed_noisy_abs = np.abs(transformed_noisy)
chirp_transformed_noisy_abs = np.abs(chirp_transformed_noisy)
if single_chirp_analysis:
    transformed_noisy_abs = transformed_noisy_abs[:,:,chirp_idx,1:-1]
    chirp_transformed_noisy_abs = chirp_transformed_noisy_abs[:,:,chirp_idx,0]
else:
    transformed_noisy_abs = np.mean(transformed_noisy_abs[:,:,:,1:-1],axis=2,keepdims=False)
    chirp_transformed_noisy_abs = np.mean(chirp_transformed_noisy_abs[:,:,:,0],axis=2,keepdims=False)

chirp_transform_max_idx = np.argmax(transformed_noisy_abs,axis=-1)
chirp_transform_max_frequency = f_axis_chirp_transform[chirp_transform_max_idx]
chirp_transform_max_frequency_diff = np.diff(chirp_transform_max_frequency,axis=-1)
chirp_transform_max_frequency_diff_mean = np.mean(chirp_transform_max_frequency_diff,axis=-1)
chirp_transform_delta_B = chirp_transform_max_frequency_diff_mean * (transformed_noisy.shape[-2] + 1)
print(chirp_transform_delta_B)

chirp_transform_frequency_offset = f_axis_chirp_transform[np.argmax(chirp_transformed_noisy_abs,axis=-1)] - calib_target_freq - chirp_transform_delta_B/2
print(chirp_transform_frequency_offset)

# short-time fft
mfft_equal_res = int((2*N_chirp_transform_bins)*(1/(2*radar_simulated.T_f))/(f_1-f_0))
f_axis_stft = np.linspace(0,1/(2*radar_simulated.T_f),mfft_equal_res//2)
w = hann(block_size, sym=False)
SFT = ShortTimeFFT(
    win=w,
    hop=block_size // 2,
    fs=1 / radar_measured.T_f,
    mfft=mfft_equal_res,
    scale_to="magnitude",
    fft_mode="centered",
)
f_bin_size_stft = SFT.delta_f

# prepare signal for stft
if_shape = radar_measured.s_if_noisy.shape
to_stft = np.reshape(radar_simulated.s_if_noisy, (if_shape[0] * if_shape[1] * if_shape[2], -1))
sft_extent = SFT.extent(to_stft.shape[-1])

# perform 
start_time = time.time()
stft_calc = SFT.stft(to_stft, axis=-1)
print(f"STFT: {blocks_per_chirp} blocks of size {block_size} per chirp." \
      + f" Frequency bin size: {f_bin_size_stft*1e-3}kHz.  Duration: {time.time() - start_time}")

# reshape stft signal
stft_result = np.reshape(
    stft_calc, (if_shape[0], if_shape[1], if_shape[2], -1, stft_calc.shape[-1])
)
stft_result = np.moveaxis(stft_result, -1, -2)[
    :, :, :, :, mfft_equal_res // 2 :
]

stft_result_abs = np.abs(stft_result)
if single_chirp_analysis:
    stft_result_abs = stft_result_abs[:,:,chirp_idx,2:-2]
else:
    stft_result_abs = np.mean(stft_result_abs[:,:,:,2:-2],axis=2,keepdims=False)


# calculate shift of maxima
stft_calib_peak_search_idx = np.where(np.abs(f_axis_stft - calib_target_freq) < calib_frequency_peak_tolerance)[0]
stft_result_abs = stft_result_abs / np.max(stft_result_abs[:,:,:,stft_calib_peak_search_idx], axis = -1)[:,:,:,None]
stft_max_idx = np.argmax(stft_result_abs[:,:,:,stft_calib_peak_search_idx], -1)+stft_calib_peak_search_idx[0]
stft_max_frequency = f_axis_stft[stft_max_idx]
stft_max_frequency_diff = np.diff(stft_max_frequency,axis=-1)
stft_max_frequency_diff_mean = np.mean(stft_max_frequency_diff,axis=-1)
stft_delta_B = (
    stft_max_frequency_diff_mean
    * (stft_result.shape[-2] - 1)
)
print(stft_delta_B)
stft_fc = np.mean(stft_max_idx[0, 0]) / (2 * stft_result.shape[-1] * radar_measured.T_f)
# error estimation
zp_frft = 4
alpha_res = 0.25e-3
alpha_start = 0.95  # only evaluate alphas for minor slope errors, depends on zp_frft
alpha_points = int((1 - alpha_start) / alpha_res)
# perform Fractional Fourier Transform for multiple alphas (additional approach to stft)
alpha_test = np.linspace(alpha_start, 1, alpha_points, endpoint=True)

# for testing select single chirp
wf = hann(radar_measured.s_if_noisy.shape[-1], sym=False)
to_frft = radar_simulated.s_if_noisy[0, 0, 0, :].real * wf
to_frft = np.pad(
    to_frft,
    (
        to_frft.shape[0] * (zp_frft - 1) // 2,
        to_frft.shape[0] * (zp_frft - 1) // 2,
    ),
    "constant",
    constant_values=(0, 0),
)
frft_result = np.empty([alpha_points, to_frft.size], dtype=complex)
for idx, alpha in enumerate(alpha_test):
    frft_result[idx, :] = sp.frft(to_frft, alpha=alpha)

frft_result_one_sided = frft_result[:, : frft_result.shape[1] // 2]
print(f"frft bin size = {1/(frft_result.shape[1]*radar_measured.T_f)}Hz")

# calculate center frequency and slope
max_idx = np.unravel_index(
    np.argmax(np.abs(frft_result_one_sided), axis=None), frft_result_one_sided.shape
)

# estimate slope error
print(f"alpha est = {alpha_test[max_idx[0]]}")
frft_k = (1 / np.tan(np.pi / 2 * alpha_test[max_idx[0]])) / (
    (radar_measured.T_f * radar_measured.T_f) * ((radar_measured.N_f - 1) * zp_frft)
)
frft_delta_B = frft_k * radar_measured.T_f * (radar_measured.N_f - 1)

# estimate frequency offset
frft_fc = (
    (
        (frft_result_one_sided.shape[1] - max_idx[1] - 1)
        / (frft_result_one_sided.shape[1] - 1)
    )
    / np.sin(np.pi / 2 * alpha_test[max_idx[0]])
) / (2 * radar_measured.T_f)

f0_error = lo_error_spec[0,1]
slope_error = lo_error_spec[1,1]
# theoretical if-frequency of calibration target
calib_target_frequency = 2 * np.linalg.norm(radar_pos - calib_target_pos) * k / c0
frft_delta_f = frft_fc - (calib_target_frequency + slope_error / 2)


stft_delta_f = stft_fc - (calib_target_frequency + slope_error / 2)
print("Results:")
print(
    f"frft_delta_B_est = {np.round(frft_delta_B*1e-3,2)}kHz,"
    + f" stft_delta_B_est = {np.round(stft_delta_B[0,0]*1e-3,2)}kHz, delta_B = {np.round(slope_error*1e-3)}kHz"
)
print(
    f"frft_delta_f_est = {np.round(frft_delta_f*1e-3,2)}kHz,"
    + f" stft_delta_f_est = {np.round(stft_delta_f*1e-3,2)}kHz, delta_f = {np.round(f0_error*1e-3)}kHz"
)

# processing
radar_simulated.range_compression(zp_fact=zp_fact_range)
radar_simulated.doppler_processing(zp_fact=zp_fact_doppler)
rp_simulated_noisy_scaled = 1 / (np.sqrt(2)) * radar_simulated.rp_noisy
rp_simulated_scaled = 1 / (np.sqrt(2)) * radar_simulated.rp
radar_measured.range_compression(zp_fact=zp_fact_range)
radar_measured.doppler_processing(zp_fact=zp_fact_doppler)
frequency_axis = np.linspace(0,fs_f/2,radar_simulated.N_f//2,endpoint=False)

if single_chirp_analysis:
    rp_simulated_noisy_abs_chirp_mean = np.abs(rp_simulated_noisy_scaled)[:,:,chirp_idx]
    rp_simulated_abs_chirp_mean = np.abs(rp_simulated_scaled)[:,:,chirp_idx]
    rp_measured_noisy_abs_chirp_mean = np.abs(radar_measured.rp_noisy)[:,:,chirp_idx]
    rp_measured_abs_chirp_mean = np.abs(radar_measured.rp)[:,:,chirp_idx]
else:
    rp_simulated_noisy_abs_chirp_mean = np.mean(np.abs(rp_simulated_noisy_scaled), axis=2)
    rp_simulated_abs_chirp_mean = np.mean(np.abs(rp_simulated_scaled), axis=2)
    rp_measured_noisy_abs_chirp_mean = np.mean(np.abs(radar_measured.rp_noisy), axis=2)
    rp_measured_abs_chirp_mean = np.mean(np.abs(radar_measured.rp), axis=2)

# range axis and peak search range
target_dists = radar_simulated.ranges / 2  # halve values to account for round-trip ranges
resolution = target_dists[1] * zp_fact_range
print(f"Resolution = {resolution}")
x_axis_plot = target_dists[: len(radar_simulated.ranges) // 2]
if axis_frequency:
    x_axis_plot = frequency_axis*1e-6

calib_peak_search_idx = np.where(np.abs(target_dists - calib_target_dist) < calib_range_peak_tolerance)[0]
min_peak_search_idx = np.where(target_dists >= min_target_range)[0][0]
# find calibration target peak
peak_pos_idx = np.argmax(rp_simulated_noisy_abs_chirp_mean[:,:,calib_peak_search_idx],axis=2) + calib_peak_search_idx[0]
print(f"Calib target pos = {np.round(target_dists[peak_pos_idx], 2)}m")
#################################################################################################################################################################
# plots

# chirp z transform result
chirp_trans_fig, (chirp_trans_ax, chirp_trans_im) = plt.subplots(1,2,figsize=[10,4],num="chirp_transform")
for i in range(transformed_noisy_abs.shape[-2]):
    chirp_trans_ax.plot(f_axis_chirp_transform, 20*np.log(transformed_noisy_abs[1,1,i]),label=i)

chirp_trans_ax.legend()
chirp_trans_ax.set_xlim([f_0,f_1])
chirp_trans_ax.set_ylim([-110,10])
chirp_trans_ax.set_xlabel('Frequency (Hz)')
chirp_trans_ax.set_ylabel('Magnitude')
chirp_trans_ax.set_title('Frequency Response')
chirp_trans_ax.grid()
chirp_trans_im.imshow(20*np.log(transformed_noisy_abs[1,1]),aspect="auto",extent=[f_0,f_1,0,blocks_per_chirp],origin="lower")
chirp_trans_im.set_xlabel("Frequency in Hz")
chirp_trans_im.set_ylabel("Block idx")

# stft result
stft_fig, (stft_ax, stft_im) = plt.subplots(1,2,figsize=[10,4],num="stft")
for i in range(stft_result_abs.shape[-2]):
    stft_ax.plot(f_axis_stft, 20*np.log(stft_result_abs[1,1,i]),label=i)

stft_ax.legend()
stft_ax.set_xlim([f_0,f_1])
stft_ax.set_ylim([-110,10])
stft_ax.set_xlabel('Frequency (Hz)')
stft_ax.set_ylabel('Magnitude')
stft_ax.set_title('Frequency Response')
stft_ax.grid()
stft_im.imshow(20*np.log(stft_result_abs[1,1]),aspect="auto",extent=[0,1/(2*radar_simulated.T_f),0,blocks_per_chirp],origin="lower")
stft_im.set_xlabel("Frequency in Hz")
stft_im.set_ylabel("Block idx")
# stft_im.set_xlim([f_0,f_1])


# range profile result
rp_simulated_noisy_plot_dB = 20 * np.log10(rp_simulated_noisy_abs_chirp_mean)[:,:,: len(radar_simulated.ranges) // 2]
rp_simulated_plot_dB = 20 * np.log10(rp_simulated_abs_chirp_mean)[:,:,: len(radar_simulated.ranges) // 2]
rp_measured_noisy_plot_dB = 20 * np.log10(rp_measured_noisy_abs_chirp_mean)[:,:,: len(radar_simulated.ranges) // 2]
rp_measured_plot_dB = 20 * np.log10(rp_measured_abs_chirp_mean)[:,:,: len(radar_simulated.ranges) // 2]

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

plt.figure("rd_map_simulated")
plt.imshow(
    20 * np.log(np.abs(radar_simulated.rd_noisy[0, 0, :, : N_f * zp_fact_range // 2])),
    aspect="auto",origin="lower",extent=[0,x_axis_plot[-1],-v_max,v_max]
)
plt.ylabel("v in m/s")
plt.xlabel("range in m")

plt.figure("rd_map_measured")
plt.imshow(
    20 * np.log(np.abs(radar_measured.rd_noisy[0, 0, :, : N_f * zp_fact_range // 2])),
    aspect="auto",origin="lower",extent=[0,x_axis_plot[-1],-v_max,v_max]
)
plt.ylabel("v in m/s")
plt.xlabel("range in m")

plt.show()