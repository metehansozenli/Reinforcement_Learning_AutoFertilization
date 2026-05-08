import os
import sys
import time
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.agriculture_env import SmartAgricultureEnv
from agent.q_learning import QLearningAgent
from evaluation.evaluate import evaluate

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

N_EPISODES = 2500
SEED = 42


# DENEY GRUPLARI
# --- Grup 1: Ogrenme Orani (alpha) ---
ALPHA_EXPERIMENTS = {
    "alpha_005": {
        "label": "α = 0.05",
        "alpha": 0.05, "alpha_end": 0.01, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#3498db",
    },
    "alpha_015": {
        "label": "α = 0.15 (Varsayilan)",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#27ae60",
    },
    "alpha_030": {
        "label": "α = 0.30",
        "alpha": 0.30, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#e74c3c",
    },
}

# --- Grup 2: Indirim Faktoru (gamma) ---
GAMMA_EXPERIMENTS = {
    "gamma_080": {
        "label": "γ = 0.80",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.80,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#e74c3c",
    },
    "gamma_095": {
        "label": "γ = 0.95 (Varsayilan)",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#27ae60",
    },
    "gamma_099": {
        "label": "γ = 0.99",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.99,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#3498db",
    },
}

# --- Grup 3: Epsilon Decay Stratejisi ---
EPSILON_EXPERIMENTS = {
    "eps_fast": {
        "label": "Hizli Decay (ε=0.9950)",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9950,
        "color": "#e74c3c",
    },
    "eps_medium": {
        "label": "Orta Decay (ε=0.9985)",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9985,
        "color": "#27ae60",
    },
    "eps_slow": {
        "label": "Yavas Decay (ε=0.9997)",
        "alpha": 0.15, "alpha_end": 0.05, "alpha_decay": 0.9990,
        "gamma": 0.95,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.9997,
        "color": "#3498db",
    },
}

ALL_GROUPS = {
    "alpha": ("Ogrenme Orani (α) Karsilastirmasi", ALPHA_EXPERIMENTS),
    "gamma": ("Indirim Faktoru (γ) Karsilastirmasi", GAMMA_EXPERIMENTS),
    "epsilon": ("Kesif Stratejisi (ε-Decay) Karsilastirmasi", EPSILON_EXPERIMENTS),
}


# ================================================================
# EGITIM
# ================================================================
def run_single_experiment(config, seed=SEED):
    """Tek bir konfigurasyonla egitim yap, sonuclari dondur."""
    np.random.seed(seed)

    env = SmartAgricultureEnv(seed=seed, weather_variance=0.35)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        alpha_end=config["alpha_end"],
        alpha_decay=config["alpha_decay"],
    )

    episode_rewards = []
    episode_harvests = []
    epsilon_history = []
    td_errors_per_episode = []

    start_time = time.time()

    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=seed + ep)
        state = env.obs_to_tuple(obs)
        total_reward = 0.0
        ep_td_errors = []

        for step in range(env.N_DAYS):
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.obs_to_tuple(next_obs)
            done = terminated or truncated
            td_error = agent.update(state, action, reward, next_state, done)
            ep_td_errors.append(abs(td_error))
            total_reward += reward
            state = next_state
            if done:
                break

        agent.end_episode()
        episode_rewards.append(total_reward)
        episode_harvests.append(info.get("harvest_efficiency", 0.0))
        epsilon_history.append(agent.epsilon)
        td_errors_per_episode.append(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0)

    elapsed = time.time() - start_time

    # Greedy evaluation
    eval_results = evaluate(agent, n_episodes=100, output_dir=".", base_seed=9999, verbose=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_harvests": episode_harvests,
        "epsilon_history": epsilon_history,
        "td_errors": td_errors_per_episode,
        "eval_summary": eval_results["summary"],
        "eval_harvests": eval_results["harvests"],
        "elapsed_seconds": round(elapsed, 2),
        "unique_states": len(agent.q_table),
    }


# ================================================================
# GORSELLESTIME
# ================================================================
def plot_group_comparison(group_name, group_title, experiments, results, output_dir):
    """Bir parametre grubu icin 4-panel karsilastirma grafigi."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(group_title, fontsize=16, fontweight="bold")
    window = 50

    def moving_avg(data, w):
        if len(data) < w:
            w = len(data)
        return np.convolve(data, np.ones(w) / w, mode="valid")

    # Panel 1: Hasat Verimliligi
    ax1 = axes[0, 0]
    for name, cfg in experiments.items():
        ma = moving_avg(results[name]["episode_harvests"], window)
        x = np.arange(window, len(results[name]["episode_harvests"]) + 1)
        ax1.plot(x, ma, color=cfg["color"], linewidth=2, label=cfg["label"])
    ax1.axhline(y=80, color="#999", linestyle="--", alpha=0.4, linewidth=1)
    ax1.axhline(y=90, color="#666", linestyle="--", alpha=0.4, linewidth=1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Hasat Verimliligi (%)")
    ax1.set_title("Hasat Verimliligi (Hareketli Ort.)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    # Panel 2: Odul
    ax2 = axes[0, 1]
    for name, cfg in experiments.items():
        ma = moving_avg(results[name]["episode_rewards"], window)
        x = np.arange(window, len(results[name]["episode_rewards"]) + 1)
        ax2.plot(x, ma, color=cfg["color"], linewidth=2, label=cfg["label"])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Toplam Odul")
    ax2.set_title("Episode Odulu (Hareketli Ort.)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    # Panel 3: TD Error
    ax3 = axes[1, 0]
    for name, cfg in experiments.items():
        ma = moving_avg(results[name]["td_errors"], window)
        x = np.arange(window, len(results[name]["td_errors"]) + 1)
        ax3.plot(x, ma, color=cfg["color"], linewidth=2, label=cfg["label"])
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Ortalama |TD Error|")
    ax3.set_title("TD Hatasi (Hareketli Ort.)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.2)

    # Panel 4: Boxplot
    ax4 = axes[1, 1]
    data = []
    labels = []
    colors = []
    for name, cfg in experiments.items():
        data.append(results[name]["eval_harvests"])
        labels.append(cfg["label"].replace(" (Varsayilan)", "\n(Varsayilan)"))
        colors.append(cfg["color"])

    bp = ax4.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax4.text(i + 1, mean_val + 1.5, f"Ort: {mean_val:.1f}",
                 ha="center", fontsize=9, fontweight="bold")

    ax4.set_ylabel("Hasat Verimliligi (%)")
    ax4.set_title("Degerlendirme Sonuclari (100 Episode)")
    ax4.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, f"comparison_{group_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")
    return path


def plot_summary_table_image(all_group_results, output_dir):
    """Tum sonuclari tek bir ozet tablo gorseli olarak uret."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_title("Hiperparametre Karsilastirma Ozet Tablosu", fontsize=16,
                 fontweight="bold", pad=20)

    # Tablo verisi
    headers = ["Grup", "Konfigürasyon", "Ort. Hasat", "Std", "Min", "Max",
               ">=80%", "Ort. Odul", "Durum S.", "Sure (s)"]
    rows = []

    for gname, (gtitle, experiments) in ALL_GROUPS.items():
        for ename, cfg in experiments.items():
            res = all_group_results[gname][ename]
            s = res["eval_summary"]
            h = s["harvest_efficiency"]
            r = s["total_reward"]
            rows.append([
                gtitle.split("(")[0].strip(),
                cfg["label"],
                f"{h['mean']:.1f}%",
                f"{h['std']:.1f}",
                f"{h['min']:.1f}%",
                f"{h['max']:.1f}%",
                f"{h['pct_above_80']:.0f}%",
                f"{r['mean']:.1f}",
                str(res["unique_states"]),
                f"{res['elapsed_seconds']:.1f}",
            ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Baslik hucreleri stillendir
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Satir renkleri — gruplara gore renklendir
    group_colors = {"alpha": "#ebf5fb", "gamma": "#eafaf1", "epsilon": "#fef9e7"}
    row_idx = 0
    for gname, (_, experiments) in ALL_GROUPS.items():
        for _ in experiments:
            row_idx += 1
            for j in range(len(headers)):
                table[row_idx, j].set_facecolor(group_colors[gname])

    # En iyi sonuclari kalinlastir
    # Her grup icinde en iyi harvest mean'i bul
    row_idx = 0
    for gname, (_, experiments) in ALL_GROUPS.items():
        group_rows = []
        for ename in experiments:
            res = all_group_results[gname][ename]
            group_rows.append((row_idx + 1, res["eval_summary"]["harvest_efficiency"]["mean"]))
            row_idx += 1
        best_row = max(group_rows, key=lambda x: x[1])[0]
        for j in range(len(headers)):
            table[best_row, j].set_text_props(fontweight="bold")
            table[best_row, j].set_edgecolor("#27ae60")
            table[best_row, j].set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Ozet tablo: {path}")
    return path


def save_all_results_json(all_group_results, output_dir):
    """Tum sonuclari JSON olarak kaydet."""
    summary = {}
    for gname, (gtitle, experiments) in ALL_GROUPS.items():
        summary[gname] = {"title": gtitle, "experiments": {}}
        for ename, cfg in experiments.items():
            res = all_group_results[gname][ename]
            h = res["eval_summary"]["harvest_efficiency"]
            r = res["eval_summary"]["total_reward"]
            summary[gname]["experiments"][ename] = {
                "label": cfg["label"],
                "params": {
                    "alpha": cfg["alpha"],
                    "gamma": cfg["gamma"],
                    "epsilon_decay": cfg["epsilon_decay"],
                },
                "eval_harvest_mean": h["mean"],
                "eval_harvest_std": h["std"],
                "eval_harvest_min": h["min"],
                "eval_harvest_max": h["max"],
                "eval_harvest_pct_above_80": h["pct_above_80"],
                "eval_reward_mean": r["mean"],
                "unique_states": res["unique_states"],
                "elapsed_seconds": res["elapsed_seconds"],
            }

    path = os.path.join(output_dir, "comparative_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [OK] JSON ozet: {path}")
    return summary


# ================================================================
# ANA PIPELINE
# ================================================================
def main():
    output_dir = os.path.join(PROJECT_ROOT, "results_comparative")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("   HIPERPARAMETRE KARSILASTIRMA DENEYLERI")
    print(f"   Sabit: {N_EPISODES} episode, seed={SEED}")
    print("=" * 65)
    total_start = time.time()

    all_group_results = {}

    for gname, (gtitle, experiments) in ALL_GROUPS.items():
        print(f"\n{'='*65}")
        print(f"  GRUP: {gtitle}")
        print(f"{'='*65}")

        group_results = {}
        for ename, cfg in experiments.items():
            print(f"\n  >>> {cfg['label']} "
                  f"(alpha={cfg['alpha']}, gamma={cfg['gamma']}, "
                  f"eps_decay={cfg['epsilon_decay']})")

            result = run_single_experiment(cfg)
            group_results[ename] = result

            h = result["eval_summary"]["harvest_efficiency"]
            print(f"      Sonuc: Hasat Ort={h['mean']:.1f}%, "
                  f"Std={h['std']:.1f}, >=80%: {h['pct_above_80']:.0f}%, "
                  f"Sure: {result['elapsed_seconds']:.1f}s")

        all_group_results[gname] = group_results

        # Grup grafigi
        plot_group_comparison(gname, gtitle, experiments, group_results, output_dir)

    # Genel ozet
    print(f"\n{'='*65}")
    print("  OZET GORSELLER OLUSTURULUYOR")
    print(f"{'='*65}")

    plot_summary_table_image(all_group_results, output_dir)
    summary = save_all_results_json(all_group_results, output_dir)

    total_elapsed = time.time() - total_start

    # Sonuc tablosu yazdir
    print(f"\n{'='*65}")
    print(f"  TAMAMLANDI! Toplam sure: {total_elapsed:.1f}s")
    print(f"{'='*65}")

    for gname, (gtitle, experiments) in ALL_GROUPS.items():
        print(f"\n  --- {gtitle} ---")
        print(f"  {'Konfig':<28} {'Hasat Ort':>10} {'Std':>6} "
              f"{'>=80%':>6} {'Sure':>7}")
        print(f"  {'-'*60}")
        for ename, cfg in experiments.items():
            r = all_group_results[gname][ename]
            h = r["eval_summary"]["harvest_efficiency"]
            print(f"  {cfg['label']:<28} {h['mean']:>9.1f}% {h['std']:>5.1f} "
                  f"{h['pct_above_80']:>5.0f}% {r['elapsed_seconds']:>6.1f}s")

    print(f"\n  Cikti dizini: {output_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
