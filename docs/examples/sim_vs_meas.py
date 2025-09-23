import sys, os  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import skradar
from scipy.constants import speed_of_light as c0
import h5py

plt.ioff()
np.random.seed(1)

# load radar configs
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\RadServe_Windows_64bit\bpsk_test_20250918_15-47-39_1.h5"
RADAR_FILENAME_HDF5_1 = r"C:\Users\Preining\Documents\CD_Lab\antenna_chamber\measurement_data\single_sensor\movement_test_1tx_20250923_13-10-02_1.h5"
target_dist = 20
calib_target_dist = 3
frame_idx = 50
process_measurement = True

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

    print(f"Loaded measurement with {f["Chn1"].shape[0]/N_s} frames")

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


# simulated radar sensor
tx_pos = np.array([[0, 0], [0, 0], [0, 0]])
rx_pos = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

lambd = c0 / fc
v_max = lambd / (4 * Ts_s)

# lo definitions
lo_spec = np.array(
    [[0, fs_f / 100], [0,10e-5 * B_sw]]
)  # specifications of available LOs (delta fc, delta B)
tx_lo = np.array(
    [[0, 1], [0, np.pi]]
)  # used LO and chirp modulation (lo idx, phase shift)
rx_lo = np.array([0, 1, 0, 1])  # used LO

radar_pos = np.array([[0], [0], [0]])
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) / 2
L_dB_vec = np.array([-65, -65, -85, -115, -115]) #  * 10
radar = skradar.FMCWRadar(  # add chirp phase modulation as parameter
    B=B_sw,
    fc=fc,
    N_f=N_f,
    T_f=1 / fs_f,
    T_s=Ts_s,
    L_freqs_vec=L_freqs_vec,
    L_dB_vec=L_dB_vec,
    N_s=N_s,
    tx_pos=tx_pos,
    rx_pos=rx_pos,
    lo_spec=lo_spec,
    tx_lo=tx_lo,
    rx_lo=rx_lo,
    tx_ant_gains=np.array([15, 15]),
    rx_ant_gains=np.array([10, 10, 10, 10]),
    pos=radar_pos,
    name="First radar",
    if_real=False,
)

# target_pos = np.array([[0], [11.3], [0]])
target_pos1 = np.array([[0], [target_dist], [0]])
target_pos2 = np.array([[0], [calib_target_dist], [0]])
target_v1 = np.array([[0], [2], [0]])
# target = skradar.Target(rcs=10, pos=target_pos, name="Static target, 10 sqm")
target1 = skradar.Target(rcs=10, pos=target_pos1,vel=target_v1, name="Moving target, 10 sqm")
target2 = skradar.Target(rcs=10, pos=target_pos2, name="Calib target, 10 sqm")

scene = skradar.Scene([radar], [target1,target2])

radar.sim_chirps()
radar.merge_mimo()

if process_measurement:
    if data.ndim == 3:
        radar.s_if = data[None,:,:,:]
        radar.s_if_noisy = data[None,:,:,:]
    elif data.ndim == 4:
        radar.s_if = data
        radar.s_if_noisy = data
    else:
        print("Invalid measurement data shape")


#radar.apply_errors()

# processing
zp_fact_range = 4
radar.range_compression(zp_fact=zp_fact_range)
radar.doppler_processing(zp_fact=4, win_doppler="hann")

target_dists = radar.ranges / 2  # halve values to account for round-trip ranges
target_dists_plot = target_dists[: len(radar.ranges) // 2]
# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_plot = 1 / (np.sqrt(2)) * radar.rp[:,:,0, : len(radar.ranges) // 2]
rp_plot_noisy = 1 / (np.sqrt(2)) * radar.rp_noisy[:,:,0, : len(radar.ranges) // 2]

rp_plot_dB = 20 * np.log10(np.abs(rp_plot))
fig_rp, ax_rp = plt.subplots(1,1,num="range_profiles",figsize=[10,5])
for tx_idx in range(rp_plot.shape[0]):
    for rx_idx in range(rp_plot.shape[1]):
        ax_rp.plot(
            target_dists_plot, rp_plot_dB[tx_idx, rx_idx] - np.max(rp_plot_dB), label=f"tx{tx_idx}, rx{rx_idx}"
        )

ax_rp.legend()
ax_rp.grid(True)
ax_rp.set_xlabel("Range (m)")
ax_rp.set_ylabel("Normalized power (dB)")
ax_rp.set_xlim([0, target_dists_plot[-1]])
plt.show()

print(radar.rd_noisy.shape)
plt.figure("rd_map")
plt.imshow(
    20 * np.log(np.abs(radar.rd[0, 0, :, : N_f * zp_fact_range // 2])),
    aspect="auto",origin="lower",extent=[0,target_dists_plot[-1],-v_max,v_max]
)
plt.ylabel("v in m/s")
plt.xlabel("range in m")
plt.show()