import sys, os  # add path where skradar is located, relative import of skradar does not work

sys.path.append(r"C:\Users\Preining\Documents\CD_Lab\skradar_modify\scikit_radar")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import skradar
from scipy.constants import speed_of_light as c0

plt.ioff()
np.random.seed(1)
B = 0.9e9
fc = 77e9 # 76.5e9
N_f = 2 * 512 # 4 * 512  # number of fast-time samples
fs_f = N_f / 1e-4 # 4 * 1e6  # fast-time sampling rate
Ts_s = (N_f - 1) / fs_f  # slow-time sampling interval
N_s = 256  # number of slow-time samples

tx_pos = np.array([[0, 0], [0, 0], [0, 0]])
rx_pos = np.array([[0, 0], [0, 0], [0, 0]])

lambd = c0 / fc
v_max = lambd / (4 * Ts_s)

# lo definitions
lo_spec = np.array(
    [[0, fs_f / 100], [0,10e-5 * B]]
)  # specifications of available LOs (delta fc, delta B)
tx_lo = np.array(
    [[0, 1], [0, np.pi]]
)  # used LO and chirp modulation (lo idx, phase shift)
rx_lo = np.array([0, 1])  # used LO

radar_pos = np.array([[0], [0], [0]])
L_freqs_vec = np.array([10, 100e3, 300e3, 5000e3, 1e8]) / 2
L_dB_vec = np.array([-65, -65, -85, -115, -115]) #  * 10
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
    lo_spec=lo_spec,
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
target_pos2 = np.array([[0], [5], [0]])
target_v1 = np.array([[0], [2], [0]])
# target = skradar.Target(rcs=10, pos=target_pos, name="Static target, 10 sqm")
target1 = skradar.Target(rcs=10, pos=target_pos1,vel=target_v1, name="Moving target, 10 sqm")
target2 = skradar.Target(rcs=10, pos=target_pos2, name="Calib target, 10 sqm")

scene = skradar.Scene([radar], [target1,target2])

radar.sim_chirps()
radar.apply_errors()
radar.merge_mimo()





# processing
zp_fact_range = 4
radar.range_compression(zp_fact=zp_fact_range)
radar.doppler_processing(zp_fact=4, win_doppler="hann")

target_dists = radar.ranges / 2  # halve values to account for round-trip ranges
target_dists_plot = target_dists[: len(radar.ranges) // 2]
# sqrt(2) to convert to RMS power from sinusoidal peak value
rp_plot = 1 / (np.sqrt(2)) * radar.rp[:,:,0, : len(radar.ranges) // 2]
rp_plot_noisy = 1 / (np.sqrt(2)) * radar.rp_noisy[:,:,0, : len(radar.ranges) // 2]

fig_rp, (ax_rp, ax_rp_noisy) = plt.subplots(2,1,num="range_profiles",figsize=[10,8])
for tx_idx in range(rp_plot.shape[0]):
    for rx_idx in range(rp_plot.shape[1]):
        ax_rp.plot(
            target_dists_plot, 20 * np.log10(np.abs(rp_plot[tx_idx,rx_idx])), label=f"tx{tx_idx}, rx{rx_idx}"
        )
        ax_rp_noisy.plot(
            target_dists_plot, 20 * np.log10(np.abs(rp_plot_noisy[tx_idx,rx_idx])), label=f"tx{tx_idx}, rx{rx_idx}"
        )

ax_rp.legend()
ax_rp.grid(True)
ax_rp.set_xlabel("Range (m)")
ax_rp.set_ylabel("RMS power (dBV)")
ax_rp.set_xlim([0, target_dists_plot[-1]])
ax_rp_noisy.legend()
ax_rp_noisy.grid(True)
ax_rp_noisy.set_xlabel("Range (m)")
ax_rp_noisy.set_ylabel("RMS power (dBV)")
ax_rp_noisy.set_xlim([0, target_dists_plot[-1]])
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