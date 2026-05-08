import os
import sys
import json
import numpy as np
from typing import List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
from io import BytesIO
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agriculture_env import ACTION_NAMES

# ─── Renk paleti ───
ACTION_COLORS = [
    "#95a5a6",  # 0: Bekle (gri)
    "#3498db",  # 1: Hafif sulama (açık mavi)
    "#2980b9",  # 2: Yoğun sulama (koyu mavi)
    "#27ae60",  # 3: Hafif gübreleme (açık yeşil)
    "#1e8449",  # 4: Yoğun gübreleme (koyu yeşil)
    "#f39c12",  # 5: Combo (turuncu)
]

# Türkçe font desteği
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================================
# 1. TRAINING CURVE
# =========================================================================
def plot_training_curve(
    episode_rewards: list,
    episode_harvests: list,
    epsilon_history: list,
    td_errors: list,
    output_dir: str = "results",
    window: int = 100,
):
    """
    4-panel eğitim eğrisi:
      (1) Toplam ödül + moving average
      (2) Hasat verimliliği + moving average
      (3) Epsilon & Alpha decay
      (4) Ortalama TD-error
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Eğitim Metrikleri — Akıllı Tarım RL", fontsize=16, fontweight="bold")

    episodes = np.arange(1, len(episode_rewards) + 1)

    def moving_avg(data, w):
        if len(data) < w:
            w = len(data)
        return np.convolve(data, np.ones(w) / w, mode="valid")

    # --- Panel 1: Ödül ---
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.15, color="#3498db", linewidth=0.5)
    ma = moving_avg(episode_rewards, window)
    ax1.plot(np.arange(window, len(episode_rewards) + 1), ma, color="#e74c3c", linewidth=2, label=f"Hareketli Ort. ({window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Toplam Ödül")
    ax1.set_title("Episode Ödülü")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Panel 2: Hasat verimliliği ---
    ax2 = axes[0, 1]
    ax2.plot(episodes, episode_harvests, alpha=0.15, color="#27ae60", linewidth=0.5)
    ma_h = moving_avg(episode_harvests, window)
    ax2.plot(np.arange(window, len(episode_harvests) + 1), ma_h, color="#e74c3c", linewidth=2, label=f"Hareketli Ort. ({window})")
    ax2.axhline(y=80, color="#f39c12", linestyle="--", alpha=0.7, label="Hedef: 80")
    ax2.axhline(y=90, color="#2ecc71", linestyle="--", alpha=0.7, label="Hedef: 90")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Hasat Verimliliği (%)")
    ax2.set_title("Hasat Verimliliği")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --- Panel 3: Epsilon ---
    ax3 = axes[1, 0]
    ax3.plot(episodes, epsilon_history, color="#9b59b6", linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_title("Keşif Oranı (ε-Decay)")
    ax3.grid(alpha=0.3)

    # --- Panel 4: TD Error ---
    ax4 = axes[1, 1]
    ax4.plot(episodes, td_errors, alpha=0.15, color="#e67e22", linewidth=0.5)
    ma_td = moving_avg(td_errors, window)
    ax4.plot(np.arange(window, len(td_errors) + 1), ma_td, color="#e74c3c", linewidth=2, label=f"Hareketli Ort. ({window})")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Ortalama |TD Error|")
    ax4.set_title("TD Hatası")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Training curve kaydedildi: {path}")
    return path


# =========================================================================
# 2. LEARNING PROGRESSION GIF
# =========================================================================
def create_learning_progression_gif(
    snapshots: List[dict],
    output_dir: str = "results",
    fps: int = 1,
):
    """
    Policy snapshot'larından öğrenme sürecini gösteren GIF oluştur.
    Her frame: belirli state alt-kümesi için en iyi aksiyonların heatmap'i.
    """
    figures = []
    for snap in snapshots:
        fig = _draw_policy_frame(
            snap["policy"],
            title=f"Episode {snap['episode']} | e={snap['epsilon']:.3f} | "
                  f"Ort. Hasat: {snap.get('mean_harvest_last100', 0):.1f}",
        )
        figures.append(fig)

    frames = _make_consistent_frames(figures)
    for fig in figures:
        plt.close(fig)

    path = os.path.join(output_dir, "learning_progression.gif")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"[OK] Learning progression GIF kaydedildi: {path}")
    return path


def _draw_policy_frame(policy: dict, title: str = ""):
    """
    Policy'yi moisture × nutrient grid'inde görselleştir.
    day=15 (orta gün), crop_health=2 (orta), toxicity=0 sabitlenir.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    grid = np.full((5, 5), -1, dtype=int)
    for m in range(5):
        for n in range(5):
            state_key = str((15, m, n, 2, 0))
            if state_key in policy:
                grid[m, n] = policy[state_key]

    # Renk haritası
    cmap_colors = ACTION_COLORS + ["#ffffff"]  # -1 için beyaz
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)

    im = ax.imshow(grid + 1, cmap=cmap, vmin=0, vmax=7, aspect="equal")

    # Hücre etiketleri
    for m in range(5):
        for n in range(5):
            val = grid[m, n]
            if val >= 0:
                ax.text(n, m, ACTION_NAMES[val][:6], ha="center", va="center",
                        fontsize=7, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4))
            else:
                ax.text(n, m, "?", ha="center", va="center", fontsize=10, color="#bdc3c7")

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"N={i}" for i in range(5)])
    ax.set_yticklabels([f"M={i}" for i in range(5)])
    ax.set_xlabel("Toprak Besini (Nutrient)")
    ax.set_ylabel("Toprak Nemi (Moisture)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=ACTION_COLORS[i], label=ACTION_NAMES[i]) for i in range(6)]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.tight_layout()
    return fig


# =========================================================================
# 3. Q-VALUE ACTION MAP
# =========================================================================
def plot_q_action_map(
    agent,
    output_dir: str = "results",
):
    """
    Farklı gün × crop_health kombinasyonları için
    en iyi aksiyonları gösteren heatmap'ler.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Q-Değer Aksiyon Haritası (moisture=2, nutrient=2 sabit)", fontsize=14, fontweight="bold")

    toxicity_vals = [0, 1, 2]
    tox_labels = ["Toksisite=0 (Yok)", "Toksisite=1 (Orta)", "Toksisite=2 (Yüksek)"]

    for idx, (tox, tox_label) in enumerate(zip(toxicity_vals, tox_labels)):
        ax = axes[idx]
        grid = np.full((5, 30), -1, dtype=int)
        q_max_grid = np.full((5, 30), np.nan)

        for day in range(30):
            for ch in range(5):
                state = (day, 2, 2, ch, tox)
                if state in agent.q_table:
                    q_vals = agent.q_table[state]
                    grid[ch, day] = int(np.argmax(q_vals))
                    q_max_grid[ch, day] = np.max(q_vals)

        from matplotlib.colors import ListedColormap
        cmap_colors = ACTION_COLORS + ["#ecf0f1"]
        cmap = ListedColormap(cmap_colors)

        im = ax.imshow(grid + 1, cmap=cmap, vmin=0, vmax=7, aspect="auto",
                       origin="lower", interpolation="nearest")

        ax.set_xlabel("Gün")
        ax.set_ylabel("Bitki Sağlığı")
        ax.set_title(tox_label, fontsize=11)
        ax.set_yticks(range(5))
        ax.set_xticks(range(0, 30, 5))

    # Shared legend
    patches = [mpatches.Patch(color=ACTION_COLORS[i], label=ACTION_NAMES[i]) for i in range(6)]
    patches.append(mpatches.Patch(color="#ecf0f1", label="Ziyaret edilmedi"))
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = os.path.join(output_dir, "q_action_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Q-action map kaydedildi: {path}")
    return path


# =========================================================================
# 4. POLICY SIMULATION GIF
# =========================================================================
def create_policy_simulation_gif(
    history: List[dict],
    seed: int,
    output_dir: str = "results",
    fps: int = 2,
):
    """
    Tek bir episode'un adım-adım görselleştirmesini GIF olarak oluştur.
    """
    figures = []
    for i, step in enumerate(history):
        fig = _draw_simulation_frame(history, current_step=i, seed=seed)
        figures.append(fig)

    frames = _make_consistent_frames(figures)
    for fig in figures:
        plt.close(fig)

    path = os.path.join(output_dir, f"policy_sim_seed_{seed}.gif")
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"[OK] Policy simulation GIF kaydedildi: {path}")
    return path


def _draw_simulation_frame(history: List[dict], current_step: int, seed: int):
    """Episode simülasyonunun tek bir frame'ini çiz."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Politika Simülasyonu — Seed {seed} | Gün {current_step}/{len(history)-1}",
                 fontsize=14, fontweight="bold")

    steps = list(range(current_step + 1))
    data = history[:current_step + 1]

    # Geçerli durum bilgisi
    curr = history[current_step]
    action_text = curr.get("action_name", "?")

    # --- Panel 1: Toprak Metrikleri ---
    ax1 = axes[0, 0]
    days = [d["day"] for d in data]
    moisture = [d["moisture"] for d in data]
    nutrient = [d["nutrient"] for d in data]
    ax1.plot(days, moisture, "-o", color="#3498db", markersize=3, label="Nem", linewidth=2)
    ax1.plot(days, nutrient, "-s", color="#27ae60", markersize=3, label="Besin", linewidth=2)
    ax1.axhspan(1.5, 3.0, alpha=0.1, color="green", label="İdeal Bölge")
    ax1.set_xlim(-0.5, 30.5)
    ax1.set_ylim(-0.2, 5.2)
    ax1.set_xlabel("Gün")
    ax1.set_ylabel("Seviye")
    ax1.set_title("Toprak Durumu")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Panel 2: Bitki Sağlığı ---
    ax2 = axes[0, 1]
    health = [d["crop_health"] for d in data]
    ax2.fill_between(days, health, alpha=0.3, color="#2ecc71")
    ax2.plot(days, health, "-o", color="#27ae60", markersize=3, linewidth=2)
    ax2.set_xlim(-0.5, 30.5)
    ax2.set_ylim(-0.2, 5.2)
    ax2.set_xlabel("Gün")
    ax2.set_ylabel("Sağlık")
    ax2.set_title("Bitki Sağlığı")
    ax2.grid(alpha=0.3)

    # --- Panel 3: Toksisite + Aksiyonlar ---
    ax3 = axes[1, 0]
    toxicity = [d["toxicity"] for d in data]
    ax3.fill_between(days, toxicity, alpha=0.3, color="#e74c3c")
    ax3.plot(days, toxicity, "-o", color="#c0392b", markersize=3, linewidth=2, label="Toksisite")
    ax3.set_xlim(-0.5, 30.5)
    ax3.set_ylim(-0.1, 3.2)
    ax3.set_xlabel("Gün")
    ax3.set_ylabel("Toksisite")
    ax3.set_title("Toksisite Seviyesi")
    ax3.grid(alpha=0.3)

    # Aksiyon bar'ları (üst kısmında)
    for d_info in data[1:]:  # ilk gün aksiyon yok
        act = d_info.get("action", 0)
        if act >= 0:
            ax3.axvline(d_info["day"], color=ACTION_COLORS[act], alpha=0.4, linewidth=3)

    ax3.legend(fontsize=8)

    # --- Panel 4: Kümülatif Ödül + Mevcut Aksiyon ---
    ax4 = axes[1, 1]
    cum_rewards = np.cumsum([d["reward"] for d in data])
    ax4.plot(days, cum_rewards, "-o", color="#9b59b6", markersize=3, linewidth=2)
    ax4.set_xlim(-0.5, 30.5)
    ax4.set_xlabel("Gün")
    ax4.set_ylabel("Kümülatif Ödül")
    ax4.set_title("Kümülatif Ödül")
    ax4.grid(alpha=0.3)

    # Mevcut aksiyon etiketi
    if current_step > 0:
        ax4.text(0.98, 0.95, f"Aksiyon: {action_text}",
                 transform=ax4.transAxes, ha="right", va="top",
                 fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=ACTION_COLORS[curr.get("action", 0)],
                           alpha=0.7),
                 color="white")

    # Hasat bilgisi (son frame)
    if "harvest_efficiency" in curr:
        fig.text(0.5, 0.01,
                 f"HASAT Verimliligi: {curr['harvest_efficiency']:.1f}%",
                 ha="center", fontsize=14, fontweight="bold",
                 color="#27ae60")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


# =========================================================================
# 5. RANDOM vs TRAINED COMPARISON
# =========================================================================
def plot_random_vs_trained(
    trained_harvests: list,
    random_harvests: list,
    output_dir: str = "results",
):
    """Eğitilmiş vs rastgele ajan karşılaştırma kutu grafiği."""
    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        [random_harvests, trained_harvests],
        labels=["Rastgele Ajan", "Eğitilmiş Ajan (Q-Learning)"],
        patch_artist=True,
        widths=0.5,
    )

    colors = ["#e74c3c", "#27ae60"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=80, color="#f39c12", linestyle="--", alpha=0.7, label="Hedef: 80")
    ax.axhline(y=90, color="#2ecc71", linestyle="--", alpha=0.7, label="Hedef: 90")

    ax.set_ylabel("Hasat Verimliliği (%)", fontsize=12)
    ax.set_title("Rastgele vs Eğitilmiş Ajan Karşılaştırması", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 100)

    # Ortalama değerleri ekle
    for i, (data, label) in enumerate([(random_harvests, "Rastgele"), (trained_harvests, "Eğitilmiş")]):
        mean_val = np.mean(data)
        ax.text(i + 1, mean_val + 2, f"Ort: {mean_val:.1f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "random_vs_trained.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Random vs Trained karşılaştırma kaydedildi: {path}")
    return path


# =========================================================================
# UTILS
# =========================================================================
def _fig_to_image(fig, target_size=None) -> np.ndarray:
    """
    Matplotlib figure'ı numpy array'e çevir.
    target_size: (width, height) tuple — tüm frame'leri aynı boyuta zorlamak için.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img.convert("RGBA"))
    buf.close()
    return arr


def _make_consistent_frames(figures) -> list:
    """
    Figure listesini tutarlı boyutlarda frame array listesine çevir.
    """
    frames = []
    target_size = None
    for fig in figures:
        arr = _fig_to_image(fig, target_size=target_size)
        if target_size is None:
            target_size = (arr.shape[1], arr.shape[0])
        frames.append(arr)
    return frames

