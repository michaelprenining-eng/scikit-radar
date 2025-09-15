import skradar
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
# params = {'text.usetex' : True,
#           'font.size' : 11,
#           'font.family' : 'lmodern',
#           }
# plt.rcParams.update(params)

from scipy.constants import speed_of_light as c0

FIGURE_WIDTH = 5.3


def generate_pn(L_freqs_vec: np.array, L_dB_vec: np.array, N_s: int, T_s: float, tau):

    N_s_pn_ss = N_s // 2  # TODO to be checked
    T_s_pn_ss = T_s * 2  # due to doubled fs and Ns by stacking

    # interpolate phase noise to IF bin frequencies
    f_fft_SSB_vec = np.arange(N_s_pn_ss) / (
        T_s_pn_ss * N_s_pn_ss
    )  # frequencies without ZP
    f_fft_IF_tmp = f_fft_SSB_vec.copy()
    # set first FFT frequency point to 1 Hz
    # to avoid problems with the interpolation in semilogx-form
    f_fft_IF_tmp[0] = 1

    L_vec = 10 ** (L_dB_vec / 10)  # PN psd specification (linear, rel. to carrier)
    # scale to bin width
    # factor 2 depends on type of PN spec (SSB vs. DSB) and conversion of L to S
    # smaini uses factor 1/2 when coming from L
    O_vec_bin = np.sqrt(
        (L_vec / 2) * (1 / (N_s_pn_ss * T_s_pn_ss))
    )  # phase noise spectrum wtihout phase spectrum added yet
    # interpolated array
    S_f_vec_bin_dB = np.interp(
        np.log10(f_fft_IF_tmp), np.log10(L_freqs_vec), 20 * np.log10(O_vec_bin)
    )
    S_f_vec_bin = 10 ** (S_f_vec_bin_dB / 20)

    S_f = S_f_vec_bin.copy()
    S_f[0] = 0  # dc
    S_f = np.hstack((S_f, 0, S_f[:0:-1]))  # now we cover 0...2*fs_SSB
    fs_PN = 2 / T_s_pn_ss  # stacking increases sampling frequency for PN
    N_PN = 2 * N_s_pn_ss  # stacking increases number of PN samples

    PN_phi_seed = np.pi * (1 - 2 * np.random.rand(tau.shape[0], int(N_PN / 2) - 1))

    # add zero tau for transmitter phase noise
    tau = np.concatenate((np.zeros(tau.shape[0:-1])[:, :, None], tau), axis=2)

    phi_shift = (
        PN_phi_seed[:, None, None, :]
        - 2
        * np.pi
        * f_fft_SSB_vec[None, None, None, 1 : int(N_s_pn_ss)]
        * tau[:, :, :, None]
    )
    phi_shift = np.concatenate(
        (
            np.zeros(np.append(phi_shift.shape[0:-1], 1)),
            phi_shift,
            np.zeros(np.append(phi_shift.shape[0:-1], 1)),
            -phi_shift[::-1],
        ),
        axis=3,
    )
    vekPN_shift_freq_domain = N_PN * S_f * np.exp(1j * phi_shift)
    vekPN_shift = np.real(np.fft.ifft(vekPN_shift_freq_domain, axis=3))

    plt.figure(2)
    plt.clf()
    plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 0, 0])
    plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 0, 1])
    # plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 0, 2])
    # plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 1, 0])
    # plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 1, 1])
    # plt.plot(1e6 * np.arange(N_PN) / fs_PN, vekPN_shift[0, 1, 2])
    # plt.legend(("$\\varphi_\\mathrm{{LO}}$", "$\\varphi_\\mathrm{{RX}}$"))
    plt.grid(True)
    plt.xlabel("t (us)")
    plt.ylabel("$\\varphi$ (rad)")
    plt.show()

    return vekPN_shift, vekPN_shift_freq_domain


def get_info(radar: skradar.FMCWRadar):
    max_range = radar.N_f * c0 / (4 * radar.B)
    range_resolution = 2 * max_range / radar.N_f
    max_velocity = c0 / (2 * radar.T_s * (2 * radar.fc))
    radar_info = {
        "max_range": max_range,
        "range_resolution": range_resolution,
        "max_velocity": max_velocity,
    }
    return radar_info


def RD_processing(
    timeDomain: np.array,
    range_fft_length: int,
    doppler_fft_length: int,
    rx_calib: np.array = np.array([1, 1, 1]),
):
    """Two-dimensional Fourier transform

    Args:
        timeDomain (numpy array): timeseries data in the shape n_channels x n_chirps x n_samples
        range_zeropad_factor (int): array length multiplier for the first Fourier transform
        doppler_zeropad_factor (int): array length multiplier for the second Fourier transform

    Returns:
        numpy array: range-doppler map in the shape n_channels x n_chirps*doppler_zeropad_factor x n_samples*range_zeropad_factor/2
    """
    num_rx = timeDomain.shape[0]
    num_chirps = timeDomain.shape[1]
    num_samples = timeDomain.shape[2]

    range_window_pre_normalization = scipy.signal.windows.hann(num_samples, sym=False)
    range_window = (
        range_window_pre_normalization
        / np.sum(range_window_pre_normalization)
        # range_window_pre_normalization / (np.sum(range_window_pre_normalization) / num_samples)
    )

    dop_window_pre_normalization = scipy.signal.windows.hann(num_chirps, sym=False)
    dop_window = (
        dop_window_pre_normalization
        / np.sum(dop_window_pre_normalization)
        # dop_window_pre_normalization / (np.sum(dop_window_pre_normalization) / num_chirps)
    )

    freqDomain = np.zeros(
        (num_rx, doppler_fft_length, range_fft_length // 2), dtype=np.cdouble
    )
    freqDomain_noMTI = np.zeros(
        (num_rx, doppler_fft_length, range_fft_length // 2), dtype=np.cdouble
    )
    # Select part of data from every receiving channel
    for rx in range(num_rx):
        time_data_rx = timeDomain[rx].transpose()
        # Subtract mean
        time_data_rx = time_data_rx - np.mean(time_data_rx.flatten())
        # First transform for range extraction
        range_data = np.fft.fft(
            time_data_rx * np.expand_dims(range_window, axis=1),
            n=range_fft_length,
            axis=0,
        )
        # Cut half of the spectrum, its a copy for real-valued signals
        range_data = range_data[: range_fft_length // 2 :, :]

        # do calibration of channel
        range_data = range_data / rx_calib[rx]

        range_data_diff = np.diff(range_data, 1, 1, append=0)

        # Second transform for Doppler extraction
        doppler_data = np.fft.fftshift(
            np.fft.fft(
                range_data_diff * np.expand_dims(dop_window, axis=0),
                n=doppler_fft_length,
                axis=1,
            ),
            axes=1,
        )

        doppler_data_noMTI = np.fft.fftshift(
            np.fft.fft(
                range_data * np.expand_dims(dop_window, axis=0),
                n=doppler_fft_length,
                axis=1,
            ),
            axes=1,
        )

        doppler_data_T = doppler_data.transpose()
        freqDomain[rx, :, :] = doppler_data_T
        doppler_data_T_noMTI = doppler_data_noMTI.transpose()
        freqDomain_noMTI[rx, :, :] = doppler_data_T_noMTI
    return freqDomain, freqDomain_noMTI


def plot_RD_maps(
    RD_map_left: np.array,
    RD_map_right: np.array,
    radar_info,
    figure_name: str = "RD_maps",
    left_title: str = "",
    left_cbar_label="",
    right_title: str = "",
    right_cbar_label="",
    size_scaling: float = 1,
):

    # setup range angle plot
    RD_figure, RD_figure_ax = plt.subplots(
        1,
        2,
        figsize=[size_scaling * FIGURE_WIDTH, size_scaling * 4],
        layout="compressed",
        num=figure_name,
    )
    RD_figure_ax[1].set_title(right_title)
    RD_figure_ax[0].set_title(left_title)

    viridis_cmap = plt.colormaps["viridis"]
    cmap_left = viridis_cmap
    cmap_right = viridis_cmap

    max_cell_value = np.nanmax(np.vstack((RD_map_left, RD_map_right)))
    RD_map_left = RD_map_left - max_cell_value
    RD_map_right = RD_map_right - max_cell_value

    if left_cbar_label == right_cbar_label:
        max_cell_value = np.nanmax(np.vstack((RD_map_left, RD_map_right)))
        min_cell_value = np.nanmin(np.vstack((RD_map_left, RD_map_right)))
        normalizer = Normalize(min_cell_value, max_cell_value)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap_left)
        c0 = RD_figure_ax[0].imshow(
            RD_map_left,
            origin="lower",
            aspect="equal",  # "auto"
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_left,
            norm=normalizer,
        )
        c1 = RD_figure_ax[1].imshow(
            RD_map_right,
            origin="lower",
            aspect="equal",  # "auto"
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_left,
            norm=normalizer,
        )
        cbar = RD_figure.colorbar(
            im,
            ax=RD_figure_ax.ravel().tolist(),
            location="bottom",
            label=left_cbar_label,
            shrink=0.7,
        )
        if "Truth" in right_cbar_label:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["False", "True"])

    else:
        c0 = RD_figure_ax[0].imshow(
            RD_map_left,
            origin="lower",
            aspect="equal",  # "auto"
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_left,
        )
        c1 = RD_figure_ax[1].imshow(
            RD_map_right,
            origin="lower",
            aspect="equal",  # "auto"
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_right,
        )
        cbar0 = RD_figure.colorbar(
            c0,
            ax=RD_figure_ax[0],
            location="bottom",
            label=left_cbar_label,
            shrink=0.7,
        )
        cbar1 = RD_figure.colorbar(
            c1,
            ax=RD_figure_ax[1],
            location="bottom",
            label=right_cbar_label,
            shrink=0.7,
        )

    for i in range(2):
        RD_figure_ax[i].set_ylabel("Velocity in m/s")
        RD_figure_ax[i].set_xlabel("Range in m")

    return RD_figure, RD_figure_ax


def plot_three_RD_maps(
    RD_map_left: np.array,
    RD_map_center: np.array,
    RD_map_right: np.array,
    radar_info,
    figure_name: str = "RD_maps",
    left_title: str = "",
    center_title: str = "",
    right_title: str = "",
    left_cbar_label="",
    center_cbar_label="",
    right_cbar_label="",
    size_scaling: float = 1,
):
    # setup range-angle plot with 3 subplots
    RD_figure, RD_figure_ax = plt.subplots(
        1,
        3,
        figsize=[size_scaling * FIGURE_WIDTH * 1.5, size_scaling * 4],
        layout="compressed",
        num=figure_name,
    )

    # Set titles for all three subplots
    RD_figure_ax[0].set_title(left_title)
    RD_figure_ax[1].set_title(center_title)
    RD_figure_ax[2].set_title(right_title)

    # Use the same colormap for all plots
    viridis_cmap = plt.colormaps["viridis"]
    cmap_left = viridis_cmap
    cmap_center = viridis_cmap
    cmap_right = viridis_cmap

    # Normalize the color range across all three maps if labels are the same
    max_cell_value = np.nanmax(np.vstack((RD_map_left, RD_map_center, RD_map_right)))
    RD_map_left = RD_map_left - max_cell_value
    RD_map_center = RD_map_center - max_cell_value
    RD_map_right = RD_map_right - max_cell_value

    if left_cbar_label == center_cbar_label == right_cbar_label:
        max_cell_value = np.nanmax(
            np.vstack((RD_map_left, RD_map_center, RD_map_right))
        )
        min_cell_value = np.nanmin(
            np.vstack((RD_map_left, RD_map_center, RD_map_right))
        )
        normalizer = Normalize(min_cell_value, max_cell_value)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap_left)

        # Plot each RD map with shared color normalization
        c0 = RD_figure_ax[0].imshow(
            RD_map_left,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_left,
            norm=normalizer,
        )
        c1 = RD_figure_ax[1].imshow(
            RD_map_center,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_center,
            norm=normalizer,
        )
        c2 = RD_figure_ax[2].imshow(
            RD_map_right,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_right,
            norm=normalizer,
        )

        # Shared colorbar for all three plots
        cbar = RD_figure.colorbar(
            im,
            ax=RD_figure_ax.ravel().tolist(),
            location="bottom",
            label=left_cbar_label,
            shrink=0.7,
        )
        if "Truth" in right_cbar_label:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["False", "True"])

    else:
        # Separate colorbars for each RD map
        c0 = RD_figure_ax[0].imshow(
            RD_map_left,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_left,
        )
        c1 = RD_figure_ax[1].imshow(
            RD_map_center,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_center,
        )
        c2 = RD_figure_ax[2].imshow(
            RD_map_right,
            origin="lower",
            aspect="equal",
            extent=[
                0,
                radar_info["max_range"],
                -radar_info["max_velocity"],
                radar_info["max_velocity"],
            ],
            cmap=cmap_right,
        )

        # Add colorbars for each plot individually
        cbar0 = RD_figure.colorbar(
            c0,
            ax=RD_figure_ax[0],
            location="bottom",
            label=left_cbar_label,
            shrink=0.7,
        )
        cbar1 = RD_figure.colorbar(
            c1,
            ax=RD_figure_ax[1],
            location="bottom",
            label=center_cbar_label,
            shrink=0.7,
        )
        cbar2 = RD_figure.colorbar(
            c2,
            ax=RD_figure_ax[2],
            location="bottom",
            label=right_cbar_label,
            shrink=0.7,
        )

    for i in range(3):
        RD_figure_ax[i].set_ylabel("Velocity in m/s")
        RD_figure_ax[i].set_xlabel("Range in m")

    return RD_figure, RD_figure_ax
