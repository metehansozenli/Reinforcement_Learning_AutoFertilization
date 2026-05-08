import os
import sys
import json
import time
import copy
import numpy as np

# Proje kök dizini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agriculture_env import SmartAgricultureEnv
from agent.q_learning import QLearningAgent


def train(
    n_episodes: int = 5000,
    snapshot_interval: int = 500,
    output_dir: str = "results",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Ana eğitim fonksiyonu.

    Returns:
        dict: Eğitim sonuçları (rewards, snapshots, meta).
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    # Ortam ve ajan oluştur
    env = SmartAgricultureEnv(seed=seed, weather_variance=0.35)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.15,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9993,
        alpha_end=0.05,
        alpha_decay=0.9997,
    )

    # Eğitim logları
    episode_rewards = []
    episode_harvests = []
    episode_lengths = []
    epsilon_history = []
    alpha_history = []
    td_errors_per_episode = []
    snapshots = []  # (episode_num, policy_dict)

    start_time = time.time()

    for ep in range(n_episodes):
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

        harvest_eff = info.get("harvest_efficiency", 0.0)
        episode_rewards.append(total_reward)
        episode_harvests.append(harvest_eff)
        episode_lengths.append(step + 1)
        epsilon_history.append(agent.epsilon)
        alpha_history.append(agent.alpha)
        td_errors_per_episode.append(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0)

        # Snapshot kaydet
        if (ep + 1) % snapshot_interval == 0 or ep == 0:
            snapshot = {
                "episode": ep + 1,
                "policy": agent.get_policy_snapshot(),
                "epsilon": agent.epsilon,
                "alpha": agent.alpha,
                "mean_reward_last100": float(np.mean(episode_rewards[-100:])),
                "mean_harvest_last100": float(np.mean(episode_harvests[-100:])),
            }
            snapshots.append(snapshot)

            if verbose:
                print(
                    f"Episode {ep+1:>5d}/{n_episodes} | "
                    f"Reward: {total_reward:>7.2f} | "
                    f"Harvest: {harvest_eff:>5.1f} | "
                    f"ε: {agent.epsilon:.4f} | "
                    f"α: {agent.alpha:.4f} | "
                    f"Avg100 Reward: {np.mean(episode_rewards[-100:]):>7.2f} | "
                    f"Avg100 Harvest: {np.mean(episode_harvests[-100:]):>5.1f} | "
                    f"States: {len(agent.q_table)}"
                )

    elapsed = time.time() - start_time

    # Son snapshot
    if snapshots[-1]["episode"] != n_episodes:
        snapshots.append({
            "episode": n_episodes,
            "policy": agent.get_policy_snapshot(),
            "epsilon": agent.epsilon,
            "alpha": agent.alpha,
            "mean_reward_last100": float(np.mean(episode_rewards[-100:])),
            "mean_harvest_last100": float(np.mean(episode_harvests[-100:])),
        })

    # Q-table kaydet
    q_table_path = os.path.join(output_dir, "q_table.json")
    agent.save(q_table_path)

    # Training meta — ensure all values are native Python types
    def _to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_native(v) for v in obj]
        return obj

    training_meta = _to_native({
        "n_episodes": n_episodes,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "hyperparams": agent.get_hyperparams(),
        "final_epsilon": agent.epsilon,
        "final_alpha": agent.alpha,
        "unique_states": len(agent.q_table),
        "total_updates": agent.total_updates,
        "final_mean_reward_100": float(np.mean(episode_rewards[-100:])),
        "final_mean_harvest_100": float(np.mean(episode_harvests[-100:])),
        "final_std_harvest_100": float(np.std(episode_harvests[-100:])),
        "best_harvest": float(np.max(episode_harvests)),
        "reward_weights": dict(SmartAgricultureEnv.DEFAULT_REWARD_WEIGHTS),
    })

    meta_path = os.path.join(output_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Eğitim tamamlandı! ({elapsed:.1f}s)")
        print(f"Son 100 ep ortalama ödül: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"Son 100 ep ortalama hasat: {np.mean(episode_harvests[-100:]):.1f}")
        print(f"En iyi hasat: {np.max(episode_harvests):.1f}")
        print(f"Ziyaret edilen benzersiz durum: {len(agent.q_table)}")
        print(f"Q-table kaydedildi: {q_table_path}")
        print(f"{'='*60}\n")

    return {
        "episode_rewards": episode_rewards,
        "episode_harvests": episode_harvests,
        "episode_lengths": episode_lengths,
        "epsilon_history": epsilon_history,
        "alpha_history": alpha_history,
        "td_errors": td_errors_per_episode,
        "snapshots": snapshots,
        "agent": agent,
        "meta": training_meta,
    }


if __name__ == "__main__":
    results = train()
