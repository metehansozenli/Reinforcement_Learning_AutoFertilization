import os
import sys
import json
import numpy as np
from typing import List
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agriculture_env import SmartAgricultureEnv, ACTION_NAMES
from agent.q_learning import QLearningAgent


def evaluate(
    agent: QLearningAgent,
    n_episodes: int = 100,
    output_dir: str = "results",
    base_seed: int = 9999,
    verbose: bool = True,
) -> dict:
    """
    Eğitilmiş ajanı greedy politika ile değerlendir.

    Returns:
        dict: Detaylı evaluation sonuçları.
    """
    os.makedirs(output_dir, exist_ok=True)

    env = SmartAgricultureEnv(weather_variance=0.35)

    harvests = []
    rewards = []
    action_counts = {i: 0 for i in range(agent.n_actions)}
    episode_details = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        state = env.obs_to_tuple(obs)

        total_reward = 0.0
        ep_actions = []

        for step in range(env.N_DAYS):
            action = agent.get_action(state, greedy=True)
            ep_actions.append(action)
            action_counts[action] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.obs_to_tuple(next_obs)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        harvest_eff = info.get("harvest_efficiency", 0.0)
        harvests.append(harvest_eff)
        rewards.append(total_reward)

        episode_details.append({
            "episode": ep,
            "seed": base_seed + ep,
            "harvest_efficiency": round(harvest_eff, 2),
            "total_reward": round(total_reward, 3),
            "actions": ep_actions,
            "final_info": {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in info.items()
            },
        })

    # İstatistikler
    harvests_arr = np.array(harvests)
    rewards_arr = np.array(rewards)

    # %95 güven aralığı
    ci95 = sp_stats.t.interval(
        0.95,
        len(harvests_arr) - 1,
        loc=np.mean(harvests_arr),
        scale=sp_stats.sem(harvests_arr),
    )

    # Action distribution
    total_actions = sum(action_counts.values())
    action_distribution = {
        ACTION_NAMES[i]: {
            "count": action_counts[i],
            "percentage": round(action_counts[i] / total_actions * 100, 1),
        }
        for i in range(agent.n_actions)
    }

    # Özet istatistikler
    summary_stats = {
        "harvest_efficiency": {
            "mean": round(float(np.mean(harvests_arr)), 2),
            "std": round(float(np.std(harvests_arr)), 2),
            "median": round(float(np.median(harvests_arr)), 2),
            "min": round(float(np.min(harvests_arr)), 2),
            "max": round(float(np.max(harvests_arr)), 2),
            "ci95_lower": round(float(ci95[0]), 2),
            "ci95_upper": round(float(ci95[1]), 2),
            "pct_above_80": round(float(np.mean(harvests_arr >= 80) * 100), 1),
            "pct_above_70": round(float(np.mean(harvests_arr >= 70) * 100), 1),
        },
        "total_reward": {
            "mean": round(float(np.mean(rewards_arr)), 2),
            "std": round(float(np.std(rewards_arr)), 2),
            "median": round(float(np.median(rewards_arr)), 2),
            "min": round(float(np.min(rewards_arr)), 2),
            "max": round(float(np.max(rewards_arr)), 2),
        },
        "action_distribution": action_distribution,
        "n_episodes": n_episodes,
    }

    # Detaylı rapor kaydet
    eval_report = {
        "summary": summary_stats,
        "episodes": episode_details,
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)

    # Özet rapor kaydet
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RAPORU")
        print("=" * 60)
        print(f"Episode sayısı: {n_episodes}")
        print(f"\nHasat Verimliliği:")
        print(f"  Ortalama : {summary_stats['harvest_efficiency']['mean']:.1f}")
        print(f"  Std      : {summary_stats['harvest_efficiency']['std']:.1f}")
        print(f"  Medyan   : {summary_stats['harvest_efficiency']['median']:.1f}")
        print(f"  Min      : {summary_stats['harvest_efficiency']['min']:.1f}")
        print(f"  Max      : {summary_stats['harvest_efficiency']['max']:.1f}")
        print(f"  CI95     : [{summary_stats['harvest_efficiency']['ci95_lower']:.1f}, "
              f"{summary_stats['harvest_efficiency']['ci95_upper']:.1f}]")
        print(f"  Ortalama : {summary_stats['total_reward']['mean']:.2f}")
        print(f"  Std      : {summary_stats['total_reward']['std']:.2f}")
        print(f"\nAksiyon Dağılımı:")
        for name, d in action_distribution.items():
            bar = "█" * int(d["percentage"] / 3)
            print(f"  {name:<20s}: {d['percentage']:>5.1f}% ({d['count']:>5d}) {bar}")
        print("=" * 60)

    return {
        "summary": summary_stats,
        "harvests": harvests,
        "rewards": rewards,
        "episode_details": episode_details,
    }


def run_policy_episodes(
    agent: QLearningAgent,
    seeds: List[int],
    output_dir: str = "results",
) -> List[List[dict]]:
    """
    Belirli seed'lerle politika episode'ları çalıştır.
    Her episode'un adım-adım geçmişini döndürür (GIF üretimi için).
    """
    env = SmartAgricultureEnv(weather_variance=0.35)
    all_histories = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        state = env.obs_to_tuple(obs)

        history = [{
            "day": 0,
            "action": -1,
            "action_name": "Başlangıç",
            "moisture": info["moisture_raw"],
            "nutrient": info["nutrient_raw"],
            "crop_health": info["crop_health_raw"],
            "toxicity": info["toxicity_raw"],
            "reward": 0.0,
            "obs": obs.tolist(),
        }]

        for step in range(env.N_DAYS):
            action = agent.get_action(state, greedy=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.obs_to_tuple(next_obs)

            history.append({
                "day": step + 1,
                "action": action,
                "action_name": ACTION_NAMES[action],
                "moisture": info["moisture_raw"],
                "nutrient": info["nutrient_raw"],
                "crop_health": info["crop_health_raw"],
                "toxicity": info["toxicity_raw"],
                "reward": round(reward, 3),
                "obs": next_obs.tolist(),
            })

            state = next_state
            if terminated or truncated:
                break

        history[-1]["harvest_efficiency"] = info.get("harvest_efficiency", 0.0)
        all_histories.append(history)

    return all_histories
