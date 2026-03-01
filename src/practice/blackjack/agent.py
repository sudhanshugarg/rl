from collections import defaultdict
import pickle
from pathlib import Path
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    @staticmethod
    def _extract_env_metadata(env: gym.Env) -> tuple[str, dict]:
        """Extract enough metadata to reconstruct the environment."""
        spec = getattr(env, "spec", None)
        if spec is None and hasattr(env, "unwrapped"):
            spec = getattr(env.unwrapped, "spec", None)

        env_id = "Blackjack-v1"
        env_kwargs = {"sab": False}
        if spec is not None:
            env_id = getattr(spec, "id", env_id)
            env_kwargs = dict(getattr(spec, "kwargs", {}) or env_kwargs)

        return env_id, env_kwargs

    @classmethod
    def create(cls, path: str | Path, render_mode: str = None) -> "BlackjackAgent":
        """Create and return an agent directly from a saved file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        env_id = data.get("env_id", "Blackjack-v1")
        env_kwargs = data.get("env_kwargs", {"sab": False, "render_mode": render_mode})
        env = gym.make(env_id, **env_kwargs)

        agent = cls(
            env=env,
            learning_rate=data["lr"],
            initial_epsilon=data["epsilon"],
            epsilon_decay=data["epsilon_decay"],
            final_epsilon=data["final_epsilon"],
            discount_factor=data["discount_factor"],
        )
        agent.load(path)
        return agent

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Persist the learned agent state to disk."""
        env_id, env_kwargs = self._extract_env_metadata(self.env)
        data = {
            "q_values": dict(self.q_values),
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
            "training_error": self.training_error,
            "env_id": env_id,
            "env_kwargs": env_kwargs,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load a previously saved agent state from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.q_values = defaultdict(
            lambda: np.zeros(self.env.action_space.n),
            data["q_values"],
        )
        self.lr = data["lr"]
        self.discount_factor = data["discount_factor"]
        self.epsilon = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.final_epsilon = data["final_epsilon"]
        self.training_error = data["training_error"]
