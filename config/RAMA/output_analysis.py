"""
Read and plot SPECULA output for the most recent run in ./output.

Products found in each timestamped output folder (params.yml -> data_store):
    res_ef.fits     residual electric field on NGS source (amplitude, phase)
    ccd.fits        pyramid detector frames
    slopes.fits     WFS slopes
    delta_comm.fits reconstructed modal commands (rec.out_modes)
    comm.fits       integrator output commands (control.out_comm)
    sr.fits         Strehl ratio time series (psf.out_sr)

Each *.fits file has two HDUs: HDU0 = data, HDU1 = time vector (s).
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def find_latest_output_dir(root=OUTPUT_ROOT):
    subdirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found in {root}")
    return max(subdirs, key=os.path.getmtime)


def load(path):
    """Load (data, time_seconds) from a SPECULA output FITS file.

    HDU1 stores time as integer nanoseconds since simulation start.
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
        time_ns = hdul[1].data
    time_s = np.asarray(time_ns, dtype=np.float64) * 1e-9
    return np.asarray(data), time_s


def main():
    out_dir = find_latest_output_dir()
    print(f"Using output folder: {out_dir}")

    sr, sr_t = load(os.path.join(out_dir, "sr.fits"))
    comm, comm_t = load(os.path.join(out_dir, "comm.fits"))
    delta_comm, dc_t = load(os.path.join(out_dir, "delta_comm.fits"))
    slopes, sl_t = load(os.path.join(out_dir, "slopes.fits"))
    ccd, ccd_t = load(os.path.join(out_dir, "ccd.fits"))
    res_ef, ef_t = load(os.path.join(out_dir, "res_ef.fits"))
    modes, m_t = load(os.path.join(out_dir, "modes.fits"))

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"SPECULA output — {os.path.basename(out_dir)}")

    # 1. Strehl ratio vs time
    ax = axs[0, 0]
    valid = sr > 0
    ax.plot(sr_t[valid], sr[valid])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strehl ratio")
    ax.set_title(f"SR (mean={sr[valid].mean():.3f})")

    # 2. Commanded modes: RMS per mode + time series of a few modes
    ax = axs[0, 1]
    n_modes_to_show = min(5, comm.shape[1])
    for i in range(n_modes_to_show):
        ax.plot(comm_t, comm[:, i], label=f"mode {i}", alpha=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Command [m]")
    ax.set_title("control.out_comm (first modes)")
    ax.legend(fontsize=7)

    # 3. RMS of delta_comm and comm per mode
    ax = axs[0, 2]
    ax.plot(np.std(delta_comm, axis=0), label="delta_comm (rec.out_modes)")
    ax.plot(np.std(comm, axis=0), label="comm (control.out_comm)")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("RMS")
    ax.set_title("Per-mode RMS")
    ax.legend(fontsize=8)

    # 4. Slopes RMS vs time
    ax = axs[1, 0]
    ax.plot(sl_t, np.std(slopes, axis=1))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Slopes RMS")
    ax.set_title("slopec.out_slopes RMS")

    # 5. Example CCD frame (last valid one)
    ax = axs[1, 1]
    frame = ccd[-1]
    im = ax.imshow(frame, origin="lower", cmap="inferno")
    ax.set_title(f"CCD frame @ t={ccd_t[-1]:.3f}s")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 6. Example residual electric field (amplitude, last frame)
    ax = axs[1, 2]
    amp = res_ef[-1, 0]
    im = ax.imshow(amp, origin="lower", cmap="viridis")
    ax.set_title(f"res_ef amplitude @ t={ef_t[-1]:.3f}s")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()

    out_png = os.path.join(out_dir, "output_analysis.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved figure to {out_png}")
    plt.show()


    plt.figure()
    plt.plot(m_t, modes)
    plt.title("Reconstructed modal commands (rec.out_modes)")
    plt.xlabel("Time [s]")
    plt.ylabel("Mode amplitude [m]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "modes_time_series.png"), dpi=150)
    plt.show()

    #plot modal RMS
    plt.figure()
    plt.plot(np.std(modes, axis=0))
    plt.title("RMS of reconstructed modal commands (rec.out_modes)")
    plt.xlabel("Mode index")
    plt.ylabel("RMS [m]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "modes_rms.png"), dpi=150)
    plt.show()

    #plot tip and tilt PSD using welch method IN LOGARITMIC FREQUENCY SCALE
    from scipy.signal import welch
    plt.figure()
    f, Pxx = welch(modes[:, 0], fs=1/(m_t[1]-m_t[0]), nperseg=512)
    plt.loglog(f, Pxx, label="Tip")
    f, Pxx = welch(modes[:, 1], fs=1/(m_t[1]-m_t[0]), nperseg=512)
    plt.loglog(f, Pxx, label="Tilt")
    plt.title("PSD of tip and tilt modes (rec.out_modes)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [m^2/Hz]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tip_tilt_psd.png"), dpi=150)
    plt.show()

    #save the tip and tilt temporal evolution as a fits file
    tilt_rama = modes[:, 1]
    tip_rama = modes[:, 0]
    hdu_data = fits.PrimaryHDU(np.vstack((tip_rama, tilt_rama)))
    hdu_time = fits.ImageHDU((m_t * 1e9).astype(np.int64))
    fits.HDUList([hdu_data, hdu_time]).writeto(
        os.path.join(out_dir, "tip_tilt_rama.fits"), overwrite=True
    )

    #as a double check, let's plot the tip and tilt temporal evolution from the fits file we just saved
    tip_tilt_rama, t_tilt = load(os.path.join(out_dir, "tip_tilt_rama.fits"))


    plt.figure()
    plt.plot(t_tilt, tip_tilt_rama[0], label="Tip")
    plt.plot(t_tilt, tip_tilt_rama[1], label="Tilt")
    plt.title("Temporal evolution of tip and tilt modes (rec.out_modes)")
    plt.xlabel("Time [s]")
    plt.ylabel("Mode amplitude [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tip_tilt_time_series.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
