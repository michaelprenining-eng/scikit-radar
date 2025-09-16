import sys  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")

import skradar
import numpy as np
import matplotlib.pyplot as plt
import h5py
from utility_functions import get_info, plot_three_RD_maps
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

import spkit as sp

plt.ioff()
import sys
import copy
from scipy.constants import speed_of_light as c0

rng = np.random.default_rng(2)

savefig = False
debugPlot = True
with_noise = True
plot_noisy = True
single_target = False

# load radar config only to get radar paramters
RADAR_FILENAME_HDF5_1 = (
    r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\3m_calib_1.h5"
)
target_dist = 20

# errors to add
slope_error = 2e-4  # fraction of reference chirp bandwidth
f0_error = 3e-4
trigger_jitter_std = 100e-12  # standard deviation of sequence trigger
start_phase_add_flag = 1
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) * 20  # phase noise
L_dB_vec = np.array([-65, -65, -85, -115, -115])  # dBc/Hz

# processing settings
zp_stft = 8 * 4
zp_frft = 4
alpha_res = 0.25e-3
alpha_start = 0.95  # only evaluate alphas for minor slope errors, depends on zp_frft
alpha_points = int((1 - alpha_start) / alpha_res)
zp_fact_range = 8
zp_fact_doppler = 8

###########################################################################################################
# signal generation part

# load radar parameters from file
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

    # signal parameters
    B_sw = f1 - f0  # chirp bandwidth
    k = B_sw / T_sw
    B_samp = k * (N_f - 1) / fs_f  # signal bandwidth while sampling
    fc = f0 + B_sw / 2  # center frequency
    lbd = c0 / fc
    Ts_s = TRampTot  # slow-time sampling interval


# define radar
radar_pos = np.array([[0], [0], [0]])
tx_pos = np.array([[0], [0], [0]])
rx_pos = np.array([[0, 0.5 * lbd, 1 * lbd, 1.5 * lbd], [0, 0, 0, 0], [0, 0, 0, 0]])
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
    pos=radar_pos,
    if_real=False,
    name="Radar",
)
radar_info = get_info(radar)
print(radar_info)

# target position
target0_pos = np.array([[0], [target_dist], [0]])
target0_vel = np.array([[0], [0], [0]])
target1_pos = np.array([[0], [20], [0]])
target1_vel = np.array([[0], [10], [0]])
target0 = skradar.Target(
    rcs=1000, pos=target0_pos, vel=target0_vel, name="Calibraion target"
)
target1 = skradar.Target(
    rcs=150, pos=target1_pos, vel=target1_vel, name="Moving target"
)

# generate scene
if single_target:
    scene = skradar.Scene([radar], [target0])
else:
    scene = skradar.Scene(
        [radar], [target0, target1]
    )  # with noise only single target possible, but why?

# theoretical if-frequency of calibration target
calib_target_frequency = 2 * np.linalg.norm(radar_pos - target0_pos) * k / c0

# generate IF-signal
radar.sim_chirps()

# second radar object for comparison with non-error signal
radar_no_correction = copy.copy(radar)

###########################################################################################################
# error introduction part

# sampling time instances for individual chirp
t_chirp = np.arange(0, radar.N_f * radar.T_f, radar.T_f)

# additional phase due to start frequency offset
delta_f = radar.B * f0_error
phase_course_f0_offset = 2 * np.pi * delta_f * t_chirp

# additional phase due to trigger jitter (linear change with time within chirp)
f0_offset_trigger = k * rng.normal(
    loc=0,
    scale=trigger_jitter_std,
    size=(radar.tx_pos.shape[1], radar.rx_pos.shape[1]),
)
phase_course_f0_offset_trigger = (
    2 * np.pi * f0_offset_trigger[:, :, None, None] * t_chirp[None, None, None, :]
)

# additional phase due to chirp slope error (quadratic change with time within chirp)
delta_B = radar.B * slope_error
phase_course_delta_B = np.pi * delta_B / t_chirp[-1] * t_chirp**2

# additional random starting phase per chirp (constant within chirp)
start_phase = (
    2 * np.pi * np.random.rand(radar.tx_pos.shape[1], radar.rx_pos.shape[1], radar.N_s)
) * start_phase_add_flag

# phase noise
chirp_cntr = 0  # assuming movement not relevant (currently)
dists, tx_dist, rx_dist = radar.calc_dists(chirp_cntr * radar.T_s)
tofs = dists / c0


plt.figure(1)
plt.clf()
plt.semilogx(L_freqs_vec, L_dB_vec - 3, "ro-")  # specified spectrum 3dB L vs. S?
plt.grid(True)
plt.title("Phase noise - power density spectrum of $\\varphi$")
plt.xlabel("$f \\mathrm{{(Hz)}}$")
plt.ylabel("$S_{{\\varphi}}(f) \\mathrm{{(dBrad^2/Hz)}}$")
# plt.legend(("Specification (shifted by 3 dB)", "Interpolated values"))
plt.show()

# sum all errors
phase_course_rdm = (
    phase_course_f0_offset[None, None, None:]
    + phase_course_f0_offset_trigger
    + phase_course_delta_B[None, None, None:]
    + start_phase[:, :, :, None]
)

# if-signal with non-coherent effects
if with_noise:
    if plot_noisy:
        radar.s_if = radar.s_if_noisy
    radar.s_if_noisy = radar.s_if_noisy * np.exp(1j * phase_course_rdm)
else:
    radar.s_if_noisy = radar.s_if * np.exp(1j * phase_course_rdm)

radar_no_correction.s_if_noisy = radar.s_if_noisy

###########################################################################################################
# error correction part

# short-time ffts to estimate slope alteration
block_length = radar.N_f // 8

# perform short time fft
# get calib target position for every Tx-Rx combination for each chirp
w = hann(block_length, sym=False)
SFT = ShortTimeFFT(
    win=w,
    hop=block_length // 2,
    fs=1 / radar.T_f,
    mfft=block_length * zp_stft,
    scale_to="magnitude",
    fft_mode="centered",
)

# prepare signal for stft
if_shape = radar.s_if_noisy.shape
to_stft = np.reshape(radar.s_if_noisy, (if_shape[0] * if_shape[1] * if_shape[2], -1))
sft_extent = SFT.extent(to_stft.shape[-1])

# perform stft
stft_calc = SFT.stft(to_stft, axis=-1)

# reshape stft signal
stft_result = np.reshape(
    stft_calc, (if_shape[0], if_shape[1], if_shape[2], -1, stft_calc.shape[-1])
)
stft_result = np.moveaxis(stft_result, -1, -2)[
    :, :, :, :, zp_stft * block_length // 2 :
]

# calculate shift of maxima
stft_max_idx = np.argmax(np.abs(stft_result), -1)
stft_max_idx_diff = np.diff(stft_max_idx, axis=-1)
stft_delta_B = (
    SFT.delta_f
    * np.mean(
        stft_max_idx_diff[0, 0, 0, 1:-1]
    )  # np.mean(stft_max_idx_diff[:, :, :, 1:-1])
    * stft_max_idx_diff.shape[-1]
)
# stft_fc = np.mean(stft_max_idx[:, :, :, 1:-1]) / (2 * stft_result.shape[-1] * radar.T_f)

# for comparison with frft only evaluate one chirp
stft_fc = np.mean(stft_max_idx[0, 0, 0, 1:-1]) / (2 * stft_result.shape[-1] * radar.T_f)

s_if_noisy_to_plot = np.real(radar.s_if_noisy)

# perform Fractional Fourier Transform for multiple alphas (additional approach to stft)
alpha_test = np.linspace(alpha_start, 1, alpha_points, endpoint=True)

# for testing select single chirp
wf = hann(radar.s_if_noisy.shape[-1], sym=False)

to_frft = radar.s_if_noisy[0, 0, 0, :].real * wf
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
print(f"frft bin size = {1/(frft_result.shape[1]*radar.T_f)}Hz")

# calculate center frequency and slope
max_idx = np.unravel_index(
    np.argmax(np.abs(frft_result_one_sided), axis=None), frft_result_one_sided.shape
)

# estimate slope error
print(f"alpha est = {alpha_test[max_idx[0]]}")
frft_k = (1 / np.tan(np.pi / 2 * alpha_test[max_idx[0]])) / (
    (radar.T_f * radar.T_f) * ((radar.N_f - 1) * zp_frft)
)
frft_delta_B = frft_k * radar.T_f * (radar.N_f - 1)

# estimate frequency offset
frft_fc = (
    (
        (frft_result_one_sided.shape[1] - max_idx[1] - 1)
        / (frft_result_one_sided.shape[1] - 1)
    )
    / np.sin(np.pi / 2 * alpha_test[max_idx[0]])
) / (2 * radar.T_f)

stft_delta_f = stft_fc - (calib_target_frequency + delta_B / 2)
frft_delta_f = frft_fc - (calib_target_frequency + delta_B / 2)

print("Results:")
print(
    f"frft_delta_B_est = {np.round(frft_delta_B*1e-3,2)}kHz ({np.round((frft_delta_B/delta_B-1)*100,2)}%),"
    + f" stft_delta_B_est = {np.round(stft_delta_B*1e-3,2)}kHz ({np.round((stft_delta_B/delta_B-1)*100,2)}%), delta_B = {np.round(delta_B*1e-3)}kHz"
)
print(
    f"frft_delta_f_est = {np.round(frft_delta_f*1e-3,2)}kHz ({np.round((frft_delta_f/delta_f-1)*100,2)}%),"
    + f" stft_delta_f_est = {np.round(stft_delta_f*1e-3,2)}kHz ({np.round((stft_delta_f/delta_f-1)*100,2)}%), delta_f = {np.round(radar.B * f0_error*1e-3)}kHz"
)

# correct the erroneous signal
# phase_course_delta_B_correction = np.pi * stft_delta_B / t_chirp[-1] * t_chirp**2
# phase_course_delta_f_correction = 2 * np.pi * stft_delta_f * t_chirp
phase_course_delta_B_correction = np.pi * frft_delta_B / t_chirp[-1] * t_chirp**2
phase_course_delta_f_correction = 2 * np.pi * frft_delta_f * t_chirp
radar.s_if_noisy = (
    radar.s_if_noisy
    * np.exp(-1j * phase_course_delta_B_correction)
    * np.exp(-1j * phase_course_delta_f_correction)
)

# process IF-signal
radar.range_compression(zp_fact=zp_fact_range)
radar_no_correction.range_compression(zp_fact=zp_fact_range)
rp_simulated = np.squeeze(np.sum(np.abs(radar.rp), (1, 2)))
rp_measured = np.squeeze(np.sum(np.abs(radar.rp_noisy), (1, 2)))
rp_measured_no_correction = np.squeeze(
    np.sum(np.abs(radar_no_correction.rp_noisy), (1, 2))
)

# find the range index of the calibration target
max_idx = np.argmax(np.abs(radar.rp_noisy)[:, :, :, : radar.rp_noisy.shape[3] // 2], 3)

# get the measured phase of the calibration target for each chirp
dim0, dim1, dim2 = np.meshgrid(
    np.arange(radar.rp_noisy.shape[0]),
    np.arange(radar.rp_noisy.shape[1]),
    np.arange(radar.rp_noisy.shape[2]),
    indexing="ij",
)
phase_corr = radar.rp_noisy[dim0, dim1, dim2, max_idx] / np.abs(
    radar.rp_noisy[dim0, dim1, dim2, max_idx]
)

# correct phase value
radar.rp_noisy = radar.rp_noisy / phase_corr[:, :, :, None]

# perform doppler processing
radar.doppler_processing(zp_fact=zp_fact_doppler, win_doppler="hann")
radar_no_correction.doppler_processing(zp_fact=zp_fact_doppler, win_doppler="hann")

# plots
range_axis = np.linspace(0, radar_info["max_range"], N_f * zp_fact_range // 2)

# plot rd maps
plot_three_RD_maps(
    20 * np.log(np.sum(np.abs(np.squeeze(radar.rd)), 0)[:, : N_f * zp_fact_range // 2]),
    20
    * np.log(
        np.sum(np.abs(np.squeeze(radar_no_correction.rd_noisy)), 0)[
            :, : N_f * zp_fact_range // 2
        ]
    ),
    20
    * np.log(
        np.sum(np.abs(np.squeeze(radar.rd_noisy)), 0)[:, : N_f * zp_fact_range // 2]
    ),
    radar_info,
    left_title="no phase error",
    center_title="with phase error",
    right_title="corrected phase error",
    left_cbar_label="Power in dB",
    center_cbar_label="Power in dB",
    right_cbar_label="Power in dB",
    figure_name="RD_maps_all",
)
# plt.savefig(
#     "measurement_data/figures/" + "RD_maps_all" + ".pdf",
#     bbox_inches="tight",
# )

# plot range spectrum
rp_simulated_plot = 20 * np.log(rp_simulated)[: N_f * zp_fact_range // 2]
rp_measured_plot = 20 * np.log(rp_measured)[: N_f * zp_fact_range // 2]
rp_measured_no_correction_plot = (
    20 * np.log(rp_measured_no_correction)[: N_f * zp_fact_range // 2]
)
peak_val = np.max(
    np.vstack((rp_simulated_plot, rp_measured_plot, rp_measured_no_correction_plot))
)
rp_fig, rp_ax = plt.subplots(
    1, 1, figsize=[4, 3], num="range_spectrum", layout="compressed"  # figsize=[5.3, 3]
)
rp_ax.plot(
    range_axis[: 2 * range_axis.size // 4],
    rp_simulated_plot[: 2 * range_axis.size // 4] - peak_val,
    label="no phase error",
)
# rp_ax.plot(
#     range_axis[: 2 * range_axis.size // 4],
#     rp_measured_no_correction_plot[: 2 * range_axis.size // 4] - peak_val,
#     label="with phase error",
# )
rp_ax.plot(
    range_axis[: 2 * range_axis.size // 4],
    rp_measured_plot[: 2 * range_axis.size // 4] - peak_val,
    label="corrected phase error",
)
# rp_ax.plot(range_axis, 20*np.log(threshold),label="threshold")
rp_ax.set_xlim([0, 2 * radar_info["max_range"] / 4])
rp_ax.set_xlabel("Range in m")
rp_ax.set_ylabel("Power in dB")
rp_ax.grid(True)
rp_ax.legend()
# plt.savefig(
#     "measurement_data/figures/" + "range_spectrum" + ".pdf",
#     bbox_inches="tight",
# )

# plot exemplary stft result for one chirp
fig_stft, ax_stft = plt.subplots(
    1, 1, num="stft_result", figsize=[5.3, 5], layout="compressed"
)
ax_stft.imshow(
    np.abs(stft_result[0, 0, 0]),
    aspect="auto",
    extent=[0, radar_info["max_range"], sft_extent[0] * 1e6, sft_extent[1] * 1e6],
    origin="lower",
)
ax_stft.set_xlabel("Range in m")
ax_stft.set_ylabel("Time in us")

# plot exemplary frft result for one chirp
fig_frft, ax_frft = plt.subplots(1, 1, figsize=[8, 5], num="frft_result")
ax_frft.imshow(
    np.abs(frft_result_one_sided)[:, ::-1],
    aspect="auto",
    origin="lower",
    extent=[0, radar_info["max_range"], alpha_start, 1],
)
ax_frft.set_ylabel("$\\alpha$")
ax_frft.xaxis.set_label_position("top")
ax_frft.xaxis.tick_top()

# plot simulated signal
if_fig, if_ax = plt.subplots(1, num="IF_signal")
if_ax.plot(t_chirp * 1e6, np.squeeze(np.real(radar.s_if))[0, 0], label="no phase error")
if_ax.plot(
    t_chirp * 1e6, np.squeeze(s_if_noisy_to_plot)[0, 0], label="with phase error"
)
if_ax.plot(
    t_chirp * 1e6,
    np.squeeze(np.real(radar.s_if_noisy))[0, 0],
    label="corrected phase error",
)
if_ax.set_xlabel("t in us")
if_ax.set_ylabel("P in dB")
if_ax.set_xlim([t_chirp[0] * 1e6, t_chirp[-1] * 1e6])
if_ax.grid()
if_ax.legend()

if debugPlot:
    if savefig:
        pass
        # figure_name = "test"
        # plt.savefig(
        #     "files/python_plots/" + figure_name + ".pdf",
        #     papertype="a4",
        #     bbox_inches="tight",
        # )
    else:
        plt.show()

# to_frft = np.pad(
#     radar.s_if_noisy * wf[None, None, None, :],
#     (
#         (0, 0),
#         (0, 0),
#         (0, 0),
#         (
#             radar.s_if_noisy.shape[-1] * (zp_frft - 1) // 2,
#             radar.s_if_noisy.shape[-1] * (zp_frft - 1) // 2,
#         ),
#     ),
#     "constant",
#     constant_values=((0, 0), (0, 0), (0, 0), (0, 0)),
# )
# frft_result = np.empty(np.append(alpha_points, to_frft.shape), dtype=complex)
# frft_max_idx = np.empty(np.append(to_frft.shape[:-1], 2), dtype=int)
# for tx_idx in range(frft_result.shape[1]):
#     for rx_idx in range(frft_result.shape[2]):
#         for ch_idx in range(frft_result.shape[3]):
#             for idx, alpha in enumerate(alpha_test):
#                 frft_result[idx, tx_idx, rx_idx, ch_idx, :] = sp.frft(
#                     to_frft[tx_idx, rx_idx, ch_idx, :], alpha=alpha
#                 )

#             frft_chirp_result = frft_result[
#                 :, tx_idx, rx_idx, ch_idx, : frft_result.shape[4] // 2
#             ]
#             frft_max_idx[tx_idx, rx_idx, ch_idx] = np.unravel_index(
#                 np.argmax(np.abs(frft_chirp_result), axis=None),
#                 frft_chirp_result.shape,
#             )
