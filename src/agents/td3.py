import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

################################## set device ##################################
print("============================================================================================")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class ReplayBuffer:
    """Simple FIFO experience replay buffer (optionally with short history sampling)."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.max_size = int(max_size)

        self.obs_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(self.max_size, dtype=np.float32)
        self.done_buf = np.zeros(self.max_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.act_buf[self.ptr] = np.asarray(act, dtype=np.float32)
        self.rew_buf[self.ptr] = float(rew)
        self.obs2_buf[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 256) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=int(batch_size))
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size: int = 256, max_hist_len: int = 10) -> Dict[str, torch.Tensor]:
        """
        Sample transitions with a short preceding history segment for recurrent encoders.

        History is extracted from contiguous replay indices and truncated at episode boundaries
        (done == 1 marks episode end).

        Returns tensors:
          - obs, act, rew, obs2, done: [B, ...]
          - hist_obs:  [B, H, obs_dim]
          - hist_act:  [B, H, act_dim]
          - hist_len:  [B] number of valid steps in hist (0..H)
          - hist_obs2: [B, H, obs_dim] (next obs history)
          - hist_act2: [B, H, act_dim] (next act history, aligned with obs2 history)
          - hist_len2: [B]
        """
        H = int(max_hist_len)
        B = int(batch_size)

        if self.size <= 1:
            raise RuntimeError("ReplayBuffer is empty; cannot sample.")

        if H < 0:
            H = 0

        # ensure we have room for H steps of history behind idx
        low = max(H, 1)
        idxs = np.random.randint(low, self.size, size=B)

        if H == 0:
            hist_obs = np.zeros((B, 1, self.obs_dim), dtype=np.float32)
            hist_act = np.zeros((B, 1, self.act_dim), dtype=np.float32)
            hist_obs2 = np.zeros((B, 1, self.obs_dim), dtype=np.float32)
            hist_act2 = np.zeros((B, 1, self.act_dim), dtype=np.float32)
            hist_len = np.zeros((B,), dtype=np.float32)
            hist_len2 = np.zeros((B,), dtype=np.float32)
        else:
            hist_obs = np.zeros((B, H, self.obs_dim), dtype=np.float32)
            hist_act = np.zeros((B, H, self.act_dim), dtype=np.float32)
            hist_obs2 = np.zeros((B, H, self.obs_dim), dtype=np.float32)
            hist_act2 = np.zeros((B, H, self.act_dim), dtype=np.float32)
            hist_len = np.full((B,), float(H), dtype=np.float32)
            hist_len2 = np.full((B,), float(H), dtype=np.float32)

            for i, idx in enumerate(idxs):
                start = idx - H
                if start < 0:
                    start = 0

                # cut at last done within [start, idx)
                done_idxs = np.where(self.done_buf[start:idx] == 1.0)[0]
                if len(done_idxs) > 0:
                    start = start + int(done_idxs[-1]) + 1

                seg_len = idx - start
                hist_len[i] = float(seg_len)
                if seg_len > 0:
                    hist_obs[i, :seg_len] = self.obs_buf[start:idx]
                    hist_act[i, :seg_len] = self.act_buf[start:idx]
                    hist_obs2[i, :seg_len] = self.obs2_buf[start:idx]
                    # align acts to next step for obs2 history
                    # (shift by +1, clamp to available indices)
                    a2_start = min(start + 1, self.size - 1)
                    a2_end = min(idx + 1, self.size)
                    # a2 segment length may differ by 1 at episode start
                    seg2 = max(0, a2_end - a2_start)
                    if seg2 > 0:
                        hist_act2[i, :min(seg_len, seg2)] = self.act_buf[a2_start:a2_start + min(seg_len, seg2)]
                        hist_len2[i] = float(min(seg_len, seg2))
                    else:
                        hist_len2[i] = 0.0
                else:
                    # first step of episode
                    hist_len2[i] = 0.0

        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
            hist_obs=hist_obs,
            hist_act=hist_act,
            hist_len=hist_len,
            hist_obs2=hist_obs2,
            hist_act2=hist_act2,
            hist_len2=hist_len2,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def _last_valid_from_sequence(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    lengths: [B] float or int, counts of valid steps in x (0..T)
    returns: [B, D] x[b, lengths[b]-1, :]
    If length==0 -> use index 0.
    """
    lengths_i = lengths.to(dtype=torch.long)
    lengths_i = torch.clamp(lengths_i, min=0)
    idx = torch.clamp(lengths_i - 1, min=0)  # [B]
    B, _, D = x.shape
    gather_idx = idx.view(B, 1, 1).repeat(1, 1, D)
    return x.gather(1, gather_idx).squeeze(1)


class HistoryEncoder(nn.Module):
    """
    Small MLP -> LSTM -> MLP encoder for history.
    Mimics the structure in the provided LSTM-TD3 reference but simplified.
    """

    def __init__(
        self,
        in_dim: int,
        pre_mlp: Tuple[int, ...] = (128,),
        lstm_hid: int = 128,
        post_mlp: Tuple[int, ...] = (128,),
    ):
        super().__init__()

        layers = []
        prev = int(in_dim)
        for h in pre_mlp:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        self.pre = nn.Sequential(*layers) if layers else nn.Identity()

        self.lstm = nn.LSTM(input_size=prev, hidden_size=int(lstm_hid), batch_first=True)

        layers = []
        prev2 = int(lstm_hid)
        for h in post_mlp:
            layers += [nn.Linear(prev2, int(h)), nn.ReLU()]
            prev2 = int(h)
        self.post = nn.Sequential(*layers) if layers else nn.Identity()
        self.out_dim = prev2

    def forward(self, hist_x: torch.Tensor, hist_len: torch.Tensor) -> torch.Tensor:
        # hist_x: [B, T, in_dim]
        B, T, _ = hist_x.shape
        x = self.pre(hist_x.view(B * T, -1)).view(B, T, -1)
        x, _ = self.lstm(x)
        x = self.post(x)
        return _last_valid_from_sequence(x, hist_len)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, use_history: bool, hist_with_past_act: bool):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_limit = float(act_limit)
        self.use_history = bool(use_history)
        self.hist_with_past_act = bool(hist_with_past_act)

        if self.use_history:
            in_dim = self.obs_dim + (self.act_dim if self.hist_with_past_act else 0)
            self.hist_enc = HistoryEncoder(in_dim=in_dim, pre_mlp=(128,), lstm_hid=128, post_mlp=(128,))
            hist_out = self.hist_enc.out_dim
        else:
            self.hist_enc = None
            hist_out = 0

        self.cur = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + hist_out, 256),
            nn.ReLU(),
            nn.Linear(256, self.act_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        hist_obs: Optional[torch.Tensor] = None,
        hist_act: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # obs: [B, obs_dim]
        cur = self.cur(obs)
        if self.use_history:
            if hist_obs is None or hist_len is None:
                B = obs.shape[0]
                hist_obs = torch.zeros((B, 1, self.obs_dim), device=obs.device, dtype=obs.dtype)
                hist_len = torch.zeros((B,), device=obs.device, dtype=obs.dtype)
            if self.hist_with_past_act:
                if hist_act is None:
                    B = obs.shape[0]
                    hist_act = torch.zeros((B, 1, self.act_dim), device=obs.device, dtype=obs.dtype)
                hist_x = torch.cat([hist_obs, hist_act], dim=-1)
            else:
                hist_x = hist_obs
            mem = self.hist_enc(hist_x, hist_len)
            x = torch.cat([mem, cur], dim=-1)
        else:
            x = cur
        a = self.head(x)
        return self.act_limit * a


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, use_history: bool, hist_with_past_act: bool):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.use_history = bool(use_history)
        self.hist_with_past_act = bool(hist_with_past_act)

        if self.use_history:
            in_dim = self.obs_dim + (self.act_dim if self.hist_with_past_act else 0)
            self.hist_enc = HistoryEncoder(in_dim=in_dim, pre_mlp=(128,), lstm_hid=128, post_mlp=(128,))
            hist_out = self.hist_enc.out_dim
        else:
            self.hist_enc = None
            hist_out = 0

        self.cur = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + hist_out, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        hist_obs: Optional[torch.Tensor] = None,
        hist_act: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cur = self.cur(torch.cat([obs, act], dim=-1))
        if self.use_history:
            if hist_obs is None or hist_len is None:
                B = obs.shape[0]
                hist_obs = torch.zeros((B, 1, self.obs_dim), device=obs.device, dtype=obs.dtype)
                hist_len = torch.zeros((B,), device=obs.device, dtype=obs.dtype)
            if self.hist_with_past_act:
                if hist_act is None:
                    B = obs.shape[0]
                    hist_act = torch.zeros((B, 1, self.act_dim), device=obs.device, dtype=obs.dtype)
                hist_x = torch.cat([hist_obs, hist_act], dim=-1)
            else:
                hist_x = hist_obs
            mem = self.hist_enc(hist_x, hist_len)
            x = torch.cat([mem, cur], dim=-1)
        else:
            x = cur
        q = self.head(x)
        return q.view(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, use_history: bool, hist_with_past_act: bool):
        super().__init__()
        self.pi = Actor(obs_dim, act_dim, act_limit, use_history=use_history, hist_with_past_act=hist_with_past_act)
        self.q1 = Critic(obs_dim, act_dim, use_history=use_history, hist_with_past_act=hist_with_past_act)
        self.q2 = Critic(obs_dim, act_dim, use_history=use_history, hist_with_past_act=hist_with_past_act)

    def act(
        self,
        obs: torch.Tensor,
        hist_obs: Optional[torch.Tensor] = None,
        hist_act: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.pi(obs, hist_obs, hist_act, hist_len)


@dataclass
class TD3UpdateStats:
    loss_q: float
    loss_pi: float
    q1_mean: float
    q2_mean: float


class TD3:
    """
    TD3 agent with optional short-history LSTM encoders (for POMDP / partial observability).

    This is a standalone agent module; integrate it into your training loop by:
      - creating a ReplayBuffer
      - calling `select_action(obs)` to act
      - storing transitions into the buffer
      - calling `update(buffer)` periodically
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float = 1.0,
        gamma: float = 0.99,
        polyak: float = 0.995,
        pi_lr: float = 1e-3,
        q_lr: float = 1e-3,
        policy_delay: int = 2,
        act_noise: float = 0.1,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        use_history: bool = False,
        hist_with_past_act: bool = False,
        max_hist_len: int = 10,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.act_limit = float(act_limit)

        self.gamma = float(gamma)
        self.polyak = float(polyak)
        self.policy_delay = int(policy_delay)

        self.act_noise = float(act_noise)
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)

        self.use_history = bool(use_history)
        self.hist_with_past_act = bool(hist_with_past_act)
        self.max_hist_len = int(max_hist_len)

        self.ac = ActorCritic(self.obs_dim, self.act_dim, self.act_limit, use_history=self.use_history,
                              hist_with_past_act=self.hist_with_past_act).to(device)
        self.ac_targ = copy.deepcopy(self.ac).to(device)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=float(pi_lr))
        q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters())
        self.q_optimizer = Adam(q_params, lr=float(q_lr))

        self._update_step = 0

    def select_action(
        self,
        obs: np.ndarray,
        noise_scale: Optional[float] = None,
        hist_obs: Optional[np.ndarray] = None,
        hist_act: Optional[np.ndarray] = None,
        hist_len: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Deterministic policy with optional Gaussian exploration noise."""
        if noise_scale is None:
            noise_scale = self.act_noise

        o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)

        h_o = None
        h_a = None
        h_l = None
        if self.use_history and hist_obs is not None and hist_len is not None:
            h_o = torch.as_tensor(hist_obs, dtype=torch.float32, device=device).view(1, -1, self.obs_dim)
            h_l = torch.as_tensor(hist_len, dtype=torch.float32, device=device).view(1)
            if self.hist_with_past_act:
                if hist_act is None:
                    h_a = torch.zeros((1, 1, self.act_dim), device=device, dtype=torch.float32)
                else:
                    h_a = torch.as_tensor(hist_act, dtype=torch.float32, device=device).view(1, -1, self.act_dim)

        a = self.ac.act(o, h_o, h_a, h_l).view(-1).cpu().numpy()
        a = a + float(noise_scale) * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def update(self, replay: ReplayBuffer, batch_size: int = 256) -> TD3UpdateStats:
        self._update_step += 1

        if self.use_history:
            batch = replay.sample_batch_with_history(batch_size=batch_size, max_hist_len=self.max_hist_len)
            o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
            h_o, h_a, h_l = batch["hist_obs"], batch["hist_act"], batch["hist_len"]
            h_o2, h_a2, h_l2 = batch["hist_obs2"], batch["hist_act2"], batch["hist_len2"]
        else:
            batch = replay.sample_batch(batch_size=batch_size)
            o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]
            h_o = h_a = h_l = None
            h_o2 = h_a2 = h_l2 = None

        # --- Q update ---
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(o2, h_o2, h_a2, h_l2)
            eps = torch.randn_like(pi_targ) * self.target_noise
            eps = torch.clamp(eps, -self.noise_clip, self.noise_clip)
            a2 = torch.clamp(pi_targ + eps, -self.act_limit, self.act_limit)

            q1_pi_targ = self.ac_targ.q1(o2, a2, h_o2, h_a2, h_l2)
            q2_pi_targ = self.ac_targ.q2(o2, a2, h_o2, h_a2, h_l2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1.0 - d) * q_pi_targ

        q1 = self.ac.q1(o, a, h_o, h_a, h_l)
        q2 = self.ac.q2(o, a, h_o, h_a, h_l)
        loss_q = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # --- Policy update (delayed) ---
        loss_pi = torch.tensor(0.0, device=device)
        if self._update_step % self.policy_delay == 0:
            # Freeze Q params for the policy update
            for p in self.ac.q1.parameters():
                p.requires_grad = False
            for p in self.ac.q2.parameters():
                p.requires_grad = False

            pi = self.ac.pi(o, h_o, h_a, h_l)
            q_pi = self.ac.q1(o, pi, h_o, h_a, h_l)
            loss_pi = -(q_pi.mean())

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            # Unfreeze Q params
            for p in self.ac.q1.parameters():
                p.requires_grad = True
            for p in self.ac.q2.parameters():
                p.requires_grad = True

            # Update target networks (polyak averaging)
            with torch.no_grad():
                for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1.0 - self.polyak) * p.data)

        return TD3UpdateStats(
            loss_q=float(loss_q.item()),
            loss_pi=float(loss_pi.item()),
            q1_mean=float(q1.mean().item()),
            q2_mean=float(q2.mean().item()),
        )

    def save(self, checkpoint_path: str):
        payload = {
            "ac_state_dict": self.ac.state_dict(),
            "ac_targ_state_dict": self.ac_targ.state_dict(),
            "pi_optimizer": self.pi_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "update_step": self._update_step,
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "act_limit": self.act_limit,
                "gamma": self.gamma,
                "polyak": self.polyak,
                "policy_delay": self.policy_delay,
                "act_noise": self.act_noise,
                "target_noise": self.target_noise,
                "noise_clip": self.noise_clip,
                "use_history": self.use_history,
                "hist_with_past_act": self.hist_with_past_act,
                "max_hist_len": self.max_hist_len,
            },
        }
        torch.save(payload, checkpoint_path)

    def load(self, checkpoint_path: str):
        payload = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if isinstance(payload, dict) and "ac_state_dict" in payload:
            self.ac.load_state_dict(payload["ac_state_dict"])
            self.ac_targ.load_state_dict(payload.get("ac_targ_state_dict", payload["ac_state_dict"]))
            if "pi_optimizer" in payload:
                try:
                    self.pi_optimizer.load_state_dict(payload["pi_optimizer"])
                except Exception:
                    pass
            if "q_optimizer" in payload:
                try:
                    self.q_optimizer.load_state_dict(payload["q_optimizer"])
                except Exception:
                    pass
            self._update_step = int(payload.get("update_step", 0))
        else:
            # Backwards compat: allow loading a raw state_dict (actor-critic only)
            self.ac.load_state_dict(payload)
            self.ac_targ.load_state_dict(payload)


