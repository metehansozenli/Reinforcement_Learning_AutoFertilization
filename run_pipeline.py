import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from training.train import train
from evaluation.evaluate import evaluate, run_policy_episodes
from visualization.visualize import (
    plot_training_curve,
    create_learning_progression_gif,
    plot_q_action_map,
    plot_random_vs_trained,
)
from visualization.farm_simulation import create_best_worst_avg_gifs
from env.agriculture_env import SmartAgricultureEnv
from agent.q_learning import QLearningAgent


def run_random_baseline(n_episodes=100, base_seed=7777):
    """Rastgele ajan baseline - karsilastirma icin."""
    env = SmartAgricultureEnv(weather_variance=0.35)
    harvests = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        for _ in range(env.N_DAYS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        harvests.append(info.get("harvest_efficiency", 0.0))

    print(f"\n  Rastgele Baseline: Ort={np.mean(harvests):.1f}, "
          f"Std={np.std(harvests):.1f}, Min={np.min(harvests):.1f}, Max={np.max(harvests):.1f}")
    return harvests


def main():
    """Ana pipeline fonksiyonu."""
    output_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("   AKILLI TARIM RL - DENEY PIPELINE'I")
    print("=" * 60)
    total_start = time.time()

    # --- ASAMA 1: Egitim ---
    print("\n" + "=" * 60)
    print("  ASAMA 1: Q-Learning Egitimi")
    print("=" * 60)
    train_results = train(
        n_episodes=2500,
        snapshot_interval=250,
        output_dir=output_dir,
        seed=42,
        verbose=True,
    )

    # --- ASAMA 2: Training Curve ---
    print("\n" + "=" * 60)
    print("  ASAMA 2: Training Curve Olusturuluyor")
    print("=" * 60)
    plot_training_curve(
        episode_rewards=train_results["episode_rewards"],
        episode_harvests=train_results["episode_harvests"],
        epsilon_history=train_results["epsilon_history"],
        td_errors=train_results["td_errors"],
        output_dir=output_dir,
    )

    # --- ASAMA 3: Learning Progression GIF ---
    print("\n" + "=" * 60)
    print("  ASAMA 3: Learning Progression GIF")
    print("=" * 60)
    create_learning_progression_gif(
        snapshots=train_results["snapshots"],
        output_dir=output_dir,
        fps=1,
    )

    # --- ASAMA 4: Q-Action Map ---
    print("\n" + "=" * 60)
    print("  ASAMA 4: Q-Value Action Map")
    print("=" * 60)
    plot_q_action_map(
        agent=train_results["agent"],
        output_dir=output_dir,
    )

    # --- ASAMA 5: Random Baseline ---
    print("\n" + "=" * 60)
    print("  ASAMA 5: Rastgele Baseline Evaluation")
    print("=" * 60)
    random_harvests = run_random_baseline(n_episodes=100)

    # --- ASAMA 6: Trained Agent Evaluation ---
    print("\n" + "=" * 60)
    print("  ASAMA 6: Egitilmis Ajan Evaluation")
    print("=" * 60)
    eval_results = evaluate(
        agent=train_results["agent"],
        n_episodes=100,
        output_dir=output_dir,
        base_seed=9999,
        verbose=True,
    )

    # --- ASAMA 7: Tarla Simulasyon GIF'leri (Ornek 1 / 2 / 3) ---
    print("\n" + "=" * 60)
    print("  ASAMA 7: Tarla Simulasyon GIF'leri (Ornek 1/2/3)")
    print("=" * 60)
    farm_gif_paths = create_best_worst_avg_gifs(
        agent=train_results["agent"],
        output_dir=output_dir,
        n_eval=50,
        base_seed=5555,
        fps=2,
    )

    # --- ASAMA 8: Random vs Trained Karsilastirma ---
    print("\n" + "=" * 60)
    print("  ASAMA 8: Random vs Trained Karsilastirmasi")
    print("=" * 60)
    plot_random_vs_trained(
        trained_harvests=eval_results["harvests"],
        random_harvests=random_harvests,
        output_dir=output_dir,
    )

    # --- OZET ---
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("   PIPELINE TAMAMLANDI!")
    print("=" * 60)
    print(f"\n  Toplam sure: {total_elapsed:.1f}s")
    print(f"\n  Cikti dizini: {output_dir}")
    print(f"\n  Uretilen dosyalar:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath) / 1024
        ext = os.path.splitext(fname)[1]
        icon = "JSON" if ext == ".json" else "IMG"
        print(f"   [{icon}] {fname} ({size:.1f} KB)")

    # Son degerlendirme ozeti
    summary = eval_results["summary"]
    h = summary["harvest_efficiency"]
    print(f"\n{'=' * 60}")
    print(f"  SONUC OZETI")
    print(f"{'=' * 60}")
    print(f"  Hasat Verimliligi (Egitilmis):")
    print(f"    Ortalama: {h['mean']:.1f}% +/- {h['std']:.1f}%")
    print(f"    CI95: [{h['ci95_lower']:.1f}%, {h['ci95_upper']:.1f}%]")
    print(f"    >=80%%: {h['pct_above_80']:.0f}% of episodes")
    print(f"  Rastgele Baseline:")
    print(f"    Ortalama: {np.mean(random_harvests):.1f}%")
    print(f"  Iyilesme: +{h['mean'] - np.mean(random_harvests):.1f}%")
    print(f"\n  Tarla Simulasyon GIF'leri:")
    for name, path in farm_gif_paths.items():
        print(f"    {name}: {path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
