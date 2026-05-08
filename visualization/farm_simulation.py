import os
import sys
import numpy as np
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agriculture_env import ACTION_NAMES

# Aksiyon renkleri
ACTION_COLORS = [
    "#95a5a6",  # 0: Bekle
    "#3498db",  # 1: Hafif sulama
    "#2980b9",  # 2: Yogun sulama
    "#27ae60",  # 3: Hafif gubre
    "#1e8449",  # 4: Yogun gubre
    "#f39c12",  # 5: Combo
]

ACTION_SHORT = ["--", "S~", "S~~", "G+", "G++", "S+G"]


def _fig_to_array(fig, target_size=None) -> np.ndarray:
    """Figure'i numpy array'e cevir."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img)
    buf.close()
    return arr


def _health_color(val):
    """Saglik degerine gore renk dondur."""
    if val >= 3.5:
        return "#27ae60"
    elif val >= 2.0:
        return "#f39c12"
    elif val >= 1.0:
        return "#e67e22"
    else:
        return "#e74c3c"


def _draw_dashboard_frame(
    history: List[dict],
    current_step: int,
    episode_label: str,
    total_days: int = 30,
) -> plt.Figure:
    """
    Sade dashboard: 3 cizgi grafigi + aksiyon seridi + buyuk skor gostergesi.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7),
                              gridspec_kw={"height_ratios": [1.2, 1.2, 0.6]})

    data = history[:current_step + 1]
    days = [d["day"] for d in data]
    curr = data[-1]
    day = curr.get("day", 0)

    # Kumulatif odul
    cum_reward = sum(d.get("reward", 0.0) for d in data)

    # ── BASLIK ──
    health_val = curr.get("crop_health", 0)
    h_color = _health_color(health_val)

    # Hasat bilgisi (son frame)
    harvest_text = ""
    if "harvest_efficiency" in curr:
        harvest_text = f"   |   HASAT: {curr['harvest_efficiency']:.1f}%"

    fig.suptitle(
        f"{episode_label}   —   Gun {day}/{total_days}{harvest_text}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── PANEL 1: Nem + Besin ──
    ax1 = axes[0]
    moisture = [d.get("moisture", 0) for d in data]
    nutrient = [d.get("nutrient", 0) for d in data]

    ax1.axhspan(1.5, 3.0, alpha=0.08, color="#27ae60")
    ax1.plot(days, moisture, "-o", color="#3498db", markersize=3, linewidth=2, label="Nem")
    ax1.plot(days, nutrient, "-s", color="#e67e22", markersize=3, linewidth=2, label="Besin")
    ax1.axhline(1.5, color="#27ae60", linestyle=":", alpha=0.4, linewidth=1)
    ax1.axhline(3.0, color="#27ae60", linestyle=":", alpha=0.4, linewidth=1)
    ax1.set_xlim(-0.5, total_days + 0.5)
    ax1.set_ylim(-0.1, 5.1)
    ax1.set_ylabel("Seviye")
    ax1.set_title("Toprak Durumu", fontsize=10, fontweight="bold", loc="left")
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(alpha=0.2)
    ax1.text(total_days + 0.3, 2.25, "ideal", fontsize=7, color="#27ae60",
             va="center", ha="left", fontstyle="italic")

    # ── PANEL 2: Bitki Sagligi + Toksisite ──
    ax2 = axes[1]
    health = [d.get("crop_health", 0) for d in data]
    toxicity = [d.get("toxicity", 0) for d in data]

    ax2.fill_between(days, health, alpha=0.15, color="#2ecc71")
    ax2.plot(days, health, "-o", color="#27ae60", markersize=3, linewidth=2.5, label="Saglik")
    ax2.plot(days, toxicity, "-^", color="#e74c3c", markersize=3, linewidth=1.5, label="Toksisite")
    ax2.set_xlim(-0.5, total_days + 0.5)
    ax2.set_ylim(-0.1, 5.1)
    ax2.set_ylabel("Seviye")
    ax2.set_title("Bitki Sagligi & Toksisite", fontsize=10, fontweight="bold", loc="left")
    ax2.legend(fontsize=8, loc="upper right", ncol=2)
    ax2.grid(alpha=0.2)

    # Sag tarafta buyuk saglik skoru
    ax2.text(total_days + 0.3, health_val, f"{health_val:.1f}",
             fontsize=14, fontweight="bold", color=h_color,
             va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.2", facecolor=h_color, alpha=0.15))

    # ── PANEL 3: Aksiyon Seridi ──
    ax3 = axes[2]
    ax3.set_xlim(-0.5, total_days + 0.5)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    ax3.set_xlabel("Gun")
    ax3.set_title(f"Aksiyonlar   |   Kumulatif Odul: {cum_reward:+.1f}",
                  fontsize=10, fontweight="bold", loc="left")

    for d_info in data[1:]:  # ilk gun aksiyon yok
        act = d_info.get("action", 0)
        d = d_info.get("day", 0)
        color = ACTION_COLORS[act] if 0 <= act < 6 else "#ccc"
        rect = Rectangle((d - 0.4, 0.1), 0.8, 0.8,
                          facecolor=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax3.add_patch(rect)
        ax3.text(d, 0.5, ACTION_SHORT[act] if 0 <= act < 6 else "?",
                 ha="center", va="center", fontsize=6, color="white", fontweight="bold")

    # Legend for actions
    patches = [mpatches.Patch(color=ACTION_COLORS[i], label=ACTION_NAMES[i])
               for i in range(6)]
    ax3.legend(handles=patches, fontsize=6, loc="upper right", ncol=6,
               framealpha=0.7, handlelength=1, handletextpad=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_farm_simulation_gif(
    history: List[dict],
    label: str,
    output_path: str,
    fps: int = 2,
):
    """
    Tek bir episode icin dashboard GIF olustur.
    """
    figures = []
    for i in range(len(history)):
        fig = _draw_dashboard_frame(
            history=history,
            current_step=i,
            episode_label=label,
            total_days=30,
        )
        figures.append(fig)

    frames = []
    target_size = None
    for fig in figures:
        arr = _fig_to_array(fig, target_size=target_size)
        if target_size is None:
            target_size = (arr.shape[1], arr.shape[0])
        frames.append(arr)
        plt.close(fig)

    # Son frame'i 3 kez tekrarla
    for _ in range(3):
        frames.append(frames[-1])

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"[OK] Farm simulation GIF: {output_path}")
    return output_path


def create_best_worst_avg_gifs(
    agent,
    output_dir: str = "results",
    n_eval: int = 50,
    base_seed: int = 5555,
    fps: int = 2,
) -> Dict[str, str]:
    """
    En iyi, en kotu ve ortalamaya en yakin episode'lari bulup
    her biri icin dashboard GIF olustur.

    Ayrica Egitilmis vs Rastgele karsilastirma GIF'i de uretir.
    """
    from env.agriculture_env import SmartAgricultureEnv
    from evaluation.evaluate import run_policy_episodes

    env = SmartAgricultureEnv(weather_variance=0.35)
    seeds = [base_seed + i for i in range(n_eval)]

    # Tum episode'lari calistir
    all_harvests = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        state = env.obs_to_tuple(obs)
        for _ in range(env.N_DAYS):
            action = agent.get_action(state, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            state = env.obs_to_tuple(obs)
            if terminated or truncated:
                break
        all_harvests.append(info.get("harvest_efficiency", 0.0))

    harvests = np.array(all_harvests)
    best_idx = int(np.argmax(harvests))
    worst_idx = int(np.argmin(harvests))
    avg_val = np.mean(harvests)
    avg_idx = int(np.argmin(np.abs(harvests - avg_val)))

    print(f"  Ornek 1: seed={seeds[best_idx]}, hasat={harvests[best_idx]:.1f}")
    print(f"  Ornek 2: seed={seeds[worst_idx]}, hasat={harvests[worst_idx]:.1f}")
    print(f"  Ornek 3: seed={seeds[avg_idx]}, hasat={harvests[avg_idx]:.1f} (ort={avg_val:.1f})")

    selected_seeds = [seeds[best_idx], seeds[worst_idx], seeds[avg_idx]]
    selected_labels = [
        f"Ornek 1 (Hasat: {harvests[best_idx]:.1f}%)",
        f"Ornek 2 (Hasat: {harvests[worst_idx]:.1f}%)",
        f"Ornek 3 (Hasat: {harvests[avg_idx]:.1f}%)",
    ]
    selected_names = ["ornek1", "ornek2", "ornek3"]

    histories = run_policy_episodes(agent, selected_seeds, output_dir)

    result_paths = {}
    for history, label, name in zip(histories, selected_labels, selected_names):
        path = os.path.join(output_dir, f"farm_sim_{name}.gif")
        create_farm_simulation_gif(
            history=history,
            label=label,
            output_path=path,
            fps=fps,
        )
        result_paths[name] = path

    # ── BONUS: Egitilmis vs Rastgele GIF ──
    comparison_seed = seeds[avg_idx]  # ortalama seed ile kiyasla
    trained_hist = run_policy_episodes(agent, [comparison_seed], output_dir)[0]

    # Rastgele ajan episode
    random_hist = _run_random_episode(comparison_seed)

    # Karsilastirma GIF
    comp_path = os.path.join(output_dir, "farm_sim_trained_vs_random.gif")
    _create_comparison_gif(trained_hist, random_hist, comp_path, fps=fps)
    result_paths["trained_vs_random"] = comp_path

    return result_paths


def _run_random_episode(seed: int) -> List[dict]:
    """Rastgele ajan ile tek episode calistir, gecmisi dondur."""
    from env.agriculture_env import SmartAgricultureEnv

    env = SmartAgricultureEnv(weather_variance=0.35)
    obs, info = env.reset(seed=seed)
    np.random.seed(seed)

    history = [{
        "day": 0,
        "action": -1,
        "action_name": "Baslangic",
        "moisture": info["moisture_raw"],
        "nutrient": info["nutrient_raw"],
        "crop_health": info["crop_health_raw"],
        "toxicity": info["toxicity_raw"],
        "reward": 0.0,
    }]

    for step in range(env.N_DAYS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        history.append({
            "day": step + 1,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "moisture": info["moisture_raw"],
            "nutrient": info["nutrient_raw"],
            "crop_health": info["crop_health_raw"],
            "toxicity": info["toxicity_raw"],
            "reward": round(reward, 3),
        })
        if terminated or truncated:
            break

    history[-1]["harvest_efficiency"] = info.get("harvest_efficiency", 0.0)
    return history


def _create_comparison_gif(
    trained_hist: List[dict],
    random_hist: List[dict],
    output_path: str,
    fps: int = 2,
):
    """Egitilmis ve Rastgele ajanin ayni seed uzerindeki performansini yan yana gosterir."""

    n_frames = max(len(trained_hist), len(random_hist))
    figures = []

    for i in range(n_frames):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        t_idx = min(i, len(trained_hist) - 1)
        r_idx = min(i, len(random_hist) - 1)
        t_data = trained_hist[:t_idx + 1]
        r_data = random_hist[:r_idx + 1]
        t_curr = t_data[-1]
        r_curr = r_data[-1]

        t_harvest = t_curr.get("harvest_efficiency", "")
        r_harvest = r_curr.get("harvest_efficiency", "")
        t_h_text = f"  —  HASAT: {t_harvest:.1f}%" if isinstance(t_harvest, float) else ""
        r_h_text = f"  —  HASAT: {r_harvest:.1f}%" if isinstance(r_harvest, float) else ""

        fig.suptitle(
            f"Egitilmis Ajan vs Rastgele Ajan   —   Gun {min(i, 30)}/30",
            fontsize=14, fontweight="bold", y=0.99,
        )

        # Sol: Egitilmis - Saglik
        ax_tl = axes[0, 0]
        t_days = [d["day"] for d in t_data]
        t_health = [d["crop_health"] for d in t_data]
        t_tox = [d["toxicity"] for d in t_data]
        ax_tl.fill_between(t_days, t_health, alpha=0.15, color="#2ecc71")
        ax_tl.plot(t_days, t_health, "-o", color="#27ae60", markersize=3, linewidth=2.5, label="Saglik")
        ax_tl.plot(t_days, t_tox, "-^", color="#e74c3c", markersize=2, linewidth=1.2, label="Toksisite")
        ax_tl.set_xlim(-0.5, 30.5)
        ax_tl.set_ylim(-0.1, 5.1)
        ax_tl.set_ylabel("Seviye")
        ax_tl.set_title(f"EGITILMIS AJAN{t_h_text}", fontsize=10, fontweight="bold",
                         color="#27ae60")
        ax_tl.legend(fontsize=7, loc="upper right")
        ax_tl.grid(alpha=0.2)

        # Sag: Rastgele - Saglik
        ax_tr = axes[0, 1]
        r_days = [d["day"] for d in r_data]
        r_health = [d["crop_health"] for d in r_data]
        r_tox = [d["toxicity"] for d in r_data]
        ax_tr.fill_between(r_days, r_health, alpha=0.15, color="#e74c3c")
        ax_tr.plot(r_days, r_health, "-o", color="#c0392b", markersize=3, linewidth=2.5, label="Saglik")
        ax_tr.plot(r_days, r_tox, "-^", color="#e74c3c", markersize=2, linewidth=1.2, label="Toksisite")
        ax_tr.set_xlim(-0.5, 30.5)
        ax_tr.set_ylim(-0.1, 5.1)
        ax_tr.set_title(f"RASTGELE AJAN{r_h_text}", fontsize=10, fontweight="bold",
                         color="#e74c3c")
        ax_tr.legend(fontsize=7, loc="upper right")
        ax_tr.grid(alpha=0.2)

        # Sol Alt: Egitilmis - Nem/Besin
        ax_bl = axes[1, 0]
        t_moisture = [d["moisture"] for d in t_data]
        t_nutrient = [d["nutrient"] for d in t_data]
        ax_bl.axhspan(1.5, 3.0, alpha=0.06, color="#27ae60")
        ax_bl.plot(t_days, t_moisture, "-", color="#3498db", linewidth=2, label="Nem")
        ax_bl.plot(t_days, t_nutrient, "-", color="#e67e22", linewidth=2, label="Besin")
        ax_bl.set_xlim(-0.5, 30.5)
        ax_bl.set_ylim(-0.1, 5.1)
        ax_bl.set_xlabel("Gun")
        ax_bl.set_ylabel("Seviye")
        ax_bl.legend(fontsize=7, loc="upper right")
        ax_bl.grid(alpha=0.2)

        # Aksiyon cubugu (alt cizgi)
        for d_info in t_data[1:]:
            act = d_info.get("action", 0)
            d = d_info["day"]
            ax_bl.axvline(d, color=ACTION_COLORS[act], alpha=0.25, linewidth=4)

        # Sag Alt: Rastgele - Nem/Besin
        ax_br = axes[1, 1]
        r_moisture = [d["moisture"] for d in r_data]
        r_nutrient = [d["nutrient"] for d in r_data]
        ax_br.axhspan(1.5, 3.0, alpha=0.06, color="#27ae60")
        ax_br.plot(r_days, r_moisture, "-", color="#3498db", linewidth=2, label="Nem")
        ax_br.plot(r_days, r_nutrient, "-", color="#e67e22", linewidth=2, label="Besin")
        ax_br.set_xlim(-0.5, 30.5)
        ax_br.set_ylim(-0.1, 5.1)
        ax_br.set_xlabel("Gun")
        ax_br.legend(fontsize=7, loc="upper right")
        ax_br.grid(alpha=0.2)

        for d_info in r_data[1:]:
            act = d_info.get("action", 0)
            d = d_info["day"]
            ax_br.axvline(d, color=ACTION_COLORS[act], alpha=0.25, linewidth=4)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)

    # Frames
    frames = []
    target_size = None
    for fig in figures:
        arr = _fig_to_array(fig, target_size=target_size)
        if target_size is None:
            target_size = (arr.shape[1], arr.shape[0])
        frames.append(arr)
        plt.close(fig)

    for _ in range(4):
        frames.append(frames[-1])

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"[OK] Karsilastirma GIF: {output_path}")
    return output_path
