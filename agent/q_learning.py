from typing import Dict, Tuple

import numpy as np
import json
import os
from collections import defaultdict


class QLearningAgent:
    """Q-Learning ajanı."""

    def __init__(
        self,
        n_actions: int = 6,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.9995,
        alpha_end: float = 0.05,
        alpha_decay: float = 0.9998,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.alpha_start = alpha
        self.alpha_end = alpha_end
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: dict[(state_tuple)] → np.array(n_actions)
        self.q_table: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # İstatistikler
        self.episode_count = 0
        self.total_updates = 0

    def get_action(self, state: tuple, greedy: bool = False) -> int:
        """
        Epsilon-greedy veya tamamen greedy aksiyon seçimi.
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.q_table[state]
        # Eşit değerler varsa rastgele bölüştür
        max_q = np.max(q_values)
        best_actions = np.where(np.abs(q_values - max_q) < 1e-8)[0]
        return int(np.random.choice(best_actions))

    def update(
        self, state: tuple, action: int, reward: float,
        next_state: tuple, done: bool
    ):
        """
        Q-Learning güncelleme kuralı:
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error
        self.total_updates += 1

        return td_error

    def end_episode(self):
        """Episode sonunda epsilon ve alpha decay uygula."""
        self.episode_count += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        self.alpha = max(
            self.alpha_end,
            self.alpha * self.alpha_decay
        )

    def get_policy_snapshot(self) -> dict:
        """
        Q-table'dan mevcut politikayı çıkar.
        Her ziyaret edilmiş state için en iyi aksiyonu döndürür.
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[str(state)] = int(np.argmax(q_values))
        return policy

    def get_q_values_snapshot(self) -> dict:
        """Q-table'ın serileştirilebilir kopyası."""
        snapshot = {}
        for state, q_values in self.q_table.items():
            snapshot[str(state)] = q_values.tolist()
        return snapshot

    def save(self, filepath: str):
        """Q-table'ı ve meta bilgileri JSON olarak kaydet."""

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        data = {
            "meta": {
                "n_actions": int(self.n_actions),
                "alpha": float(self.alpha),
                "alpha_start": float(self.alpha_start),
                "gamma": float(self.gamma),
                "epsilon": float(self.epsilon),
                "epsilon_start": float(self.epsilon_start),
                "epsilon_end": float(self.epsilon_end),
                "epsilon_decay": float(self.epsilon_decay),
                "episode_count": int(self.episode_count),
                "total_updates": int(self.total_updates),
                "unique_states_visited": len(self.q_table),
            },
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    def load(self, filepath: str):
        """Kaydedilmiş Q-table'ı yükle."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["meta"]
        self.n_actions = meta["n_actions"]
        self.alpha = meta["alpha"]
        self.gamma = meta["gamma"]
        self.epsilon = meta["epsilon"]
        self.episode_count = meta["episode_count"]
        self.total_updates = meta["total_updates"]

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for key_str, values in data["q_table"].items():
            # Key string'i tuple'a çevir
            key = tuple(int(x) for x in key_str.strip("()").split(", "))
            self.q_table[key] = np.array(values)

    def get_hyperparams(self) -> dict:
        """Hiperparametreleri sözlük olarak döndür."""
        return {
            "n_actions": self.n_actions,
            "alpha_start": self.alpha_start,
            "alpha_current": round(self.alpha, 5),
            "alpha_end": self.alpha_end,
            "alpha_decay": self.alpha_decay,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_current": round(self.epsilon, 5),
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
        }
