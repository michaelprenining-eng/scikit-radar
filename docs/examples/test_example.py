import sys  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
import numpy as np
import matplotlib.pyplot as plt
import skradar

plt.ioff()
np.random.seed(1)
B = 1e9
fc = 76.5e9
N_f = 4 * 512  # number of fast-time samples
fs_f = 4 * 1e6  # fast-time sampling rate
Ts_s = (N_f - 1) / fs_f  # slow-time sampling interval
N_s = 256  # number of slow-time samples

tx_pos = np.array([[0, 0], [0, 0], [0, 0]])
rx_pos = np.array([[0, 0], [0, 0], [0, 0]])
tx_lo = np.array([[0, 1], [0, np.pi]])
rx_lo = np.array([0, 1])

radar_pos = np.array([[0], [0], [0]])
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) / 2
L_dB_vec = np.array([-65, -65, -85, -115, -115])
radar = skradar.FMCWRadar(  # add chirp phase modulation as parameter
    B=B,
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
    tx_ant_gains=np.array([15, 15]),
    rx_ant_gains=np.array([10, 10]),
    pos=radar_pos,
    name="First radar",
    if_real=False,
)

# target_pos = np.array([[0], [11.3], [0]])
target_pos1 = np.array([[0], [40], [0]])
# target = skradar.Target(rcs=10, pos=target_pos, name="Static target, 10 sqm")
target1 = skradar.Target(rcs=10, pos=target_pos1, name="Static target, 10 sqm")

scene = skradar.Scene([radar], [target1])

radar.sim_chirps()

zp_fact_range = 4
radar.range_compression(zp_fact=zp_fact_range)
radar.doppler_processing(zp_fact=4, win_doppler="hann")

target_dists = radar.ranges / 2  # halve values to account for round-trip ranges
target_dists_plot = target_dists[: len(radar.ranges) // 2]
# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_plot = 1 / (np.sqrt(2)) * radar.rp[0, :, 0, : len(radar.ranges) // 2]
rp_plot_noisy = 1 / (np.sqrt(2)) * radar.rp_noisy[0, :, 0, : len(radar.ranges) // 2]
peak_idx = np.argmax(np.abs(rp_plot[0]))

plt.figure(1, figsize=(12, 8))
plt.clf()
plt.subplot(1, 1, 1)
plt.plot(
    target_dists_plot, 20 * np.log10(np.abs(rp_plot[0])), label="Coherent receiver"
)
plt.plot(
    target_dists_plot, 20 * np.log10(np.abs(rp_plot[1])), label="Incoherent receiver"
)
# plt.plot(target_dists_plot, 20 * np.log10(np.abs(rp_plot_noisy[0])), label="Noisy")
# plt.plot(
#     target_dists_plot, 20 * np.log10(np.abs(rp_plot_noisy[1])), label="Noisy incoherent"
# )
plt.plot(
    [target_dists_plot[peak_idx], target_dists_plot[peak_idx]],
    [
        20 * np.log10(np.abs(rp_plot[0, peak_idx])) - 40,
        20 * np.log10(np.abs(rp_plot[0, peak_idx])) + 10,
    ],
    "--r",
)
plt.legend()
plt.grid(True)
plt.xlabel("Range (m)")
plt.ylabel("RMS power (dBV)")
plt.show()
print(radar.rd_noisy.shape)
plt.figure("rd_map")
plt.imshow(
    20 * np.log(np.abs(radar.rd[0, 0, :, : N_f * zp_fact_range // 2])),
    aspect="auto",
)
# plt.imshow(
#     np.abs(radar.rd[0, 1, :,: N_f * zp_fact_range // 2]),
#     aspect="auto",
# )
plt.show()
