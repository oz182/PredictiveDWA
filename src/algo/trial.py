import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def weights_to_beta_params(left_weight: float, right_weight: float, pmin: float = 0.5, pmax: float = 5.0, k: float = 1.0) -> tuple[float, float]:
    # a = k * float(left_weight)
    # b = k * float(right_weight)
    # m = max(a, b)
    # ea = np.exp(a - m)
    # eb = np.exp(b - m)
    # p_left = float(ea / (ea + eb))
    # p_right = 1.0 - p_left
    # alpha = pmin + p_left * (pmax - pmin)
    # beta_param = pmin + p_right * (pmax - pmin)
    return float(left_weight), float(right_weight)


def beta_headings(theta_ph: float, theta_range: float, n_samples: int,
                  left_weight: float, right_weight: float,
                  bin_multiplier: int = 3) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Threshold-based selection: keep x where Beta pdf >= 0.5, then pick n
    equally spaced samples across the remaining x-domain.
    """
    alpha_param, beta_param = weights_to_beta_params(left_weight, right_weight)

    n = max(1, int(n_samples))
    # Dense grid to approximate the continuous domain
    u = np.linspace(0.0, 1.0, 5001)
    pdf = beta.pdf(u, alpha_param, beta_param)

    # Keep only u where pdf >= 0.5 * max(pdf) computed over u in [0.1, 0.9]
    inner_mask = (u >= 0.01) & (u <= 0.99)
    max_pdf = float(np.max(pdf[inner_mask])) if np.any(inner_mask) else float(np.max(pdf))
    print(f"max_pdf (0.1-0.9): {max_pdf}")
    threshold = 0.5 * max_pdf
    mask = pdf >= threshold
    if not np.any(mask):
        # Fallback: if nothing passes threshold, distribute uniformly over [0,1]
        u_samples = np.linspace(0.0, 1.0, n)
    else:
        u_kept = u[mask]
        if len(u_kept) >= n:
            # Equally distribute by index across the kept domain
            idxs = np.linspace(0, len(u_kept) - 1, n)
            u_samples = u_kept[np.round(idxs).astype(int)]
        else:
            # If very narrow region, spread n points across its extent
            u_samples = np.linspace(float(u_kept[0]), float(u_kept[-1]), n)

    # Map to heading range
    headings = theta_ph - theta_range + u_samples * (2.0 * theta_range)
    return headings.astype(np.float64), np.asarray(u_samples, dtype=np.float64), (alpha_param, beta_param)


def main():
    parser = argparse.ArgumentParser(description="Visualize Beta-based heading sampling")
    parser.add_argument("--left", type=float, default=5, help="left_weight")
    parser.add_argument("--right", type=float, default=1.1, help="right_weight")
    parser.add_argument("--theta-ph-deg", type=float, default=0.0, help="theta_ph in degrees")
    parser.add_argument("--theta-range-deg", type=float, default=30.0, help="theta_range in degrees (half-range)")
    parser.add_argument("--samples", type=int, default=9, help="number of heading samples")
    parser.add_argument("--bin-mult", type=int, default=3, help="bin multiplier (M = n * bin_mult)")
    args = parser.parse_args()

    theta_ph = math.radians(args.theta_ph_deg)
    theta_range = math.radians(args.theta_range_deg)

    headings, u_samples, (alpha_param, beta_param) = beta_headings(
        theta_ph=theta_ph,
        theta_range=theta_range,
        n_samples=args.samples,
        left_weight=args.left,
        right_weight=args.right,
        bin_multiplier=args.bin_mult,
    )

    # Prepare plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Beta pdf and retained region with samples in [0,1]
    u = np.linspace(0.0, 1.0, 1000)
    pdf = beta.pdf(u, alpha_param, beta_param)
    axes[0].plot(u, pdf, color="tab:gray", linewidth=2, label=f"Beta(a={alpha_param:.2f}, b={beta_param:.2f})")
    axes[0].vlines(u_samples, 0.0, beta.pdf(u_samples, alpha_param, beta_param), color="tab:orange", alpha=0.8, linewidth=1.5, label="samples (u)")
    axes[0].set_title("Beta PDF and sample positions (u)")
    axes[0].set_xlabel("u in [0,1]")
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot mapped headings in radians and degrees
    axes[1].hlines(0.0, theta_ph - theta_range, theta_ph + theta_range, color="lightgray", linewidth=6, alpha=0.4)
    axes[1].vlines(headings, ymin=-0.1, ymax=0.1, color="tab:blue", linewidth=2, label="headings")
    axes[1].vlines([theta_ph], ymin=-0.15, ymax=0.15, color="tab:red", linewidth=2, label="theta_ph")
    axes[1].set_title("Mapped heading samples")
    axes[1].set_xlabel("heading (radians)")
    axes[1].set_yticks([])
    axes[1].set_xlim([theta_ph - theta_range, theta_ph + theta_range])
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()


