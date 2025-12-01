import skradar
import numpy as np
import scipy as sp
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from skradar.detection import cfar

# plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
# params = {'text.usetex' : True,
#           'font.size' : 11,
#           'font.family' : 'lmodern',
#           }
# plt.rcParams.update(params)

from scipy.constants import speed_of_light as c0

FIGURE_WIDTH = 5.3


def os_cfar_2d(rd_map, guard_cells, training_cells, rank, threshold_factor):
    """
    Perform 2D Ordered Statistics CFAR on a range-Doppler map.
    
    Args:
        rd_map (np.ndarray): Range-Doppler map (2D matrix).
        guard_cells (int): Number of guard cells around the CUT (Cell Under Test).
        training_cells (int): Number of training cells around the guard cells.
        rank (int): Rank to pick for ordered statistics (used for noise estimation).
        threshold_factor (float): Scaling factor for the detection threshold.

    Returns:
        np.ndarray: Binary mask of detected target locations.
    """
    rows, cols = rd_map.shape
    total_training_cells = (2 * training_cells + 1) ** 2 - (2 * guard_cells + 1) ** 2
    
    # Initialize CFAR result and padding for edge calculation
    cfar_map = np.zeros_like(rd_map)
    padded_map = np.pad(rd_map, pad_width=training_cells + guard_cells, mode='constant')
    
    # Sliding window approach
    for r in range(rows):
        for c in range(cols):
            # Extract window
            r_start = r
            r_end = r + 2 * (training_cells + guard_cells) + 1
            c_start = c
            c_end = c + 2 * (training_cells + guard_cells) + 1
            window = padded_map[r_start:r_end, c_start:c_end]
            
            # Exclude guard cells and CUT
            cut_start = training_cells
            cut_end = training_cells + 2 * guard_cells + 1
            cut_window = window[cut_start:cut_end, cut_start:cut_end]
            noise_training_cells = np.delete(window.flatten(), 
                                             np.arange(cut_start * window.shape[0] + cut_start, 
                                                       cut_end * window.shape[1] + cut_end))
            
            # Ordered Statistics CFAR: Select the k-th largest training value
            noise_estimate = np.sort(noise_training_cells)[-int(noise_training_cells.shape[0]*rank)]
            
            # Calculate threshold
            threshold = noise_estimate * threshold_factor
            
            # Compare the CUT with the threshold
            if rd_map[r, c] > threshold:
                cfar_map[r, c] = 1
    
    return cfar_map

def find_peaks_2d(cfar_map, rd_map, window_size):
    """
    Perform peak detection on a range-Doppler map after CFAR.
    
    Args:
        cfar_map (np.ndarray): Binary CFAR output map.
        rd_map (np.ndarray): Original range-Doppler map.
        window_size (int): Window size for local peak detection (should be odd).
    
    Returns:
        list: List of detected peak locations [(row, col), ...].
    """
    # Local maxima search
    local_max = maximum_filter(rd_map, size=window_size) == rd_map
    peaks_binary = local_max & (cfar_map > 0)
    peak_indices = np.argwhere(peaks_binary)
    
    return peak_indices

def detect_target_and_spurs(rd_map, debug_plot:bool=False):
    rd_map = np.concatenate((rd_map[:,-1,:][:,None,:],rd_map,rd_map[:,0,:][:,None,:]),axis=1)
    rd_map_vdim_mean = np.mean(np.abs(rd_map[:,:,:rd_map.shape[-1]//2]),(0,-2))

    cfarConfig_ft = cfar.CFARConfig(train_cells=20, guard_cells=10,
                                               pfa=1e-2, mode=cfar.CFARMode.CA)
    cfar_threshold_rp = cfar.cfar_threshold(rd_map_vdim_mean,cfarConfig_ft)
    mask = rd_map_vdim_mean>cfar_threshold_rp
    idx_array = np.arange(rd_map_vdim_mean.shape[0])[mask]
    max_idx_rp = idx_array[sp.signal.find_peaks(rd_map_vdim_mean[mask])[0][0]]
    
    # rd_v_slice = np.concatenate((rd_map[:,:,max_idx_rp],rd_map[:,:,max_idx_rp]),axis=1)
    rd_v_slice = rd_map[:,:,max_idx_rp]

    rd_v_slice_abs = np.mean(np.abs(rd_v_slice),axis=0)
    cfarConfig_st = cfar.CFARConfig(train_cells=25, guard_cells=10,
                                               pfa=1e-4, mode=cfar.CFARMode.CA)
    cfar_threshold_vp = cfar.cfar_threshold(rd_v_slice_abs,cfarConfig_st)
    mask = rd_v_slice_abs>cfar_threshold_vp
    idx_array = np.arange(rd_v_slice_abs.shape[0])[mask]
    peak_idx_vp = idx_array[sp.signal.find_peaks(rd_v_slice_abs[mask])[0]]
    
    # first idx should be peak, not spur
    target_peak = np.argmax(rd_v_slice_abs[peak_idx_vp])
    peak_spur_idx = np.hstack((peak_idx_vp[target_peak:], peak_idx_vp[:target_peak]))
    
    to_ifft = rd_v_slice[:,peak_spur_idx]/np.abs(rd_v_slice[:,peak_idx_vp])[:,target_peak,None]
    
    angle_result = np.angle(np.fft.ifft(to_ifft,axis=1))
    print((angle_result-angle_result[:,0,None])*180/np.pi)
    print(np.mean((angle_result-angle_result[:,0,None]),0)*180/np.pi)

    if debug_plot:
        rd_map_vdim_mean_dB = 20*np.log(rd_map_vdim_mean)
        maximum = np.max(rd_map_vdim_mean_dB)
        fig, (ax, ax1) = plt.subplots(1,2,figsize=[12,4],num="estimation_profiles")
        ax.plot(rd_map_vdim_mean_dB-maximum)
        ax.plot(20*np.log(cfar_threshold_rp)-maximum)
        ax.plot(max_idx_rp,rd_map_vdim_mean_dB[max_idx_rp]-maximum,'x')
        ax.set_xlim([0, rd_map_vdim_mean.shape[0]])
        ax.set_xlabel("Range bins (1)")
        ax.set_ylabel("Normalized power (dB)")
        ax.legend(["profile", "cfar threshold"])
        ax.grid()
        ax.set_title("Range profile")
        rd_v_slice_abs_dB = 20*np.log(rd_v_slice_abs)
        maximum = np.max(rd_v_slice_abs_dB)
        ax1.plot(rd_v_slice_abs_dB-maximum)
        ax1.plot(20*np.log(cfar_threshold_vp)-maximum)
        ax1.plot(peak_idx_vp,rd_v_slice_abs_dB[peak_idx_vp]-maximum,'x')
        ax1.set_title("Doppler cross section")
        ax1.set_xlabel("Doppler bins (1)")
        ax1.set_ylabel("Normalized power (dB)")
        ax1.set_xlim([0, rd_v_slice_abs.shape[0]])
        ax1.legend(["profile", "cfar threshold"])
        ax1.grid()
        plt.savefig("estimation_profiles.pdf",
            bbox_inches = 'tight'
            )
        plt.show()

    return np.mean((angle_result-angle_result[:,0,None]),0)*180/np.pi


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
