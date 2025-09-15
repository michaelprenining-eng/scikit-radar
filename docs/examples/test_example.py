import sys  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
import numpy as np
import matplotlib.pyplot as plt
import skradar

plt.ioff()
# np.random.seed(1)
B = 1e9
fc = 76.5e9
N_f = 4 * 512  # number of fast-time samples
fs_f = 4 * 1e6  # fast-time sampling rate
Ts_s = (N_f - 1) / fs_f  # slow-time sampling interval
N_s = 1  # number of slow-time samples

tx_pos = np.array([[0, 0], [0, 0], [0, 0]])
rx_pos = np.array([[0], [0], [0]])
tx_lo = np.array([0, 1])
rx_lo = np.array([0])

radar_pos = np.array([[0], [0], [0]])
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8])
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
    rx_ant_gains=np.array([10]),
    pos=radar_pos,
    name="First radar",
    if_real=True,
)

target_pos = np.array([[0], [11.3], [0]])
target = skradar.Target(rcs=10, pos=target_pos, name="Static target, 10 sqm")

scene = skradar.Scene([radar], [target])

fig = plt.figure(1, figsize=(12, 8))
plt.clf()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim((-7.5, 7.5))
ax.set_ylim((0, 15))
ax.set_zlim((-7.5, 7.5))
scene.visualize("world", ax, coord_len=2)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

fig = plt.figure(2, figsize=(12, 8))
plt.clf()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim((-7.5, 7.5))
ax.set_ylim((-15, 0))
ax.set_zlim((-7.5, 7.5))
scene.visualize(target, ax, coord_len=2)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

radar.sim_chirps()

radar.range_compression(zp_fact=32)

target_dists = radar.ranges / 2  # halve values to account for round-trip ranges
target_dists_plot = target_dists[: len(radar.ranges) // 2]
# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_plot = 1 / (np.sqrt(2)) * radar.rp[0, 0, 0, : len(radar.ranges) // 2]
rp_plot_noisy = 1 / (np.sqrt(2)) * radar.rp_noisy[0, 0, 0, : len(radar.ranges) // 2]
peak_idx = np.argmax(np.abs(rp_plot))

plt.figure(2, figsize=(12, 8))
plt.clf()
plt.subplot(2, 1, 1)
plt.plot(target_dists_plot, 20 * np.log10(np.abs(rp_plot)), label="Noiseless")
plt.plot(target_dists_plot, 20 * np.log10(np.abs(rp_plot_noisy)), label="Noisy")
plt.plot(
    [target_dists_plot[peak_idx], target_dists_plot[peak_idx]],
    [
        20 * np.log10(np.abs(rp_plot[peak_idx])) - 40,
        20 * np.log10(np.abs(rp_plot[peak_idx])) + 10,
    ],
    "--r",
)
plt.legend()
plt.grid(True)
plt.xlabel("Range (m)")
plt.ylabel("RMS power (dBV)")
plt.subplot(2, 1, 2)
plt.plot(
    target_dists_plot[peak_idx - 10 : peak_idx + 10],
    20 * np.log10(np.abs(rp_plot[peak_idx - 10 : peak_idx + 10])),
    "-x",
    label="Noiseless",
)
plt.plot(
    target_dists_plot[peak_idx - 10 : peak_idx + 10],
    20 * np.log10(np.abs(rp_plot_noisy[peak_idx - 10 : peak_idx + 10])),
    "-o",
    label="Noisy",
)
plt.plot(target_dists_plot[peak_idx], 20 * np.log10(np.abs(rp_plot[peak_idx])), ".r")
plt.plot(
    [target_dists_plot[peak_idx], target_dists_plot[peak_idx]],
    [
        20 * np.log10(np.abs(rp_plot[peak_idx])) - 1,
        20 * np.log10(np.abs(rp_plot[peak_idx])) + 1,
    ],
    "--r",
)
plt.legend()
plt.grid(True)
plt.xlabel("Range (m)")
plt.ylabel("RMS power (dBV)")
plt.show()
print(f"Estimated target distance: {radar.ranges[peak_idx]/2:.3f} m")
