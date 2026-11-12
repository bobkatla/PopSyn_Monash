"""Weighted conditional WGAN utilities for household population synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

HOUSEHOLD_ATTRIBUTES = ("dwelltype", "owndwell", "hhinc", "totalvehs", "hhsize")
CONDITION_ATTRIBUTES = ("hhsize", "hhinc")
GENERATED_ATTRIBUTES = ("dwelltype", "owndwell", "totalvehs")


@dataclass(frozen=True)
class WGANConfig:
    latent_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 512
    epochs: int = 100
    n_critic: int = 5
    lambda_gp: float = 10.0
    learning_rate: float = 1e-4
    temperature_start: float = 1.0
    temperature_end: float = 0.2
    max_grad_norm: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("latent_dim", "hidden_dim", "batch_size", "epochs", "n_critic"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.lambda_gp) or self.lambda_gp < 0:
            raise ValueError("lambda_gp must be finite and nonnegative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not np.isfinite(self.temperature_start)
            or not np.isfinite(self.temperature_end)
            or self.temperature_start <= 0
            or self.temperature_end <= 0
        ):
            raise ValueError("temperatures must be finite and positive")
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")


@dataclass(frozen=True)
class PreparedHouseholdData:
    frame: pd.DataFrame
    state_names: dict[str, list[str]]
    income_given_hhsize: dict[str, dict[str, float]]
    negative_income_probability: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class WGANTrainingResult:
    model: "ConditionalHouseholdWGAN"
    state_names: dict[str, list[str]]
    income_given_hhsize: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]


def select_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weights(values: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(values, errors="raise").astype(float)
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} must be finite")
    if (result <= 0).any():
        raise ValueError(f"{name} must be positive")
    return result


def _census_probability(census: pd.DataFrame, attribute: str, state: str) -> float:
    if not isinstance(census.columns, pd.MultiIndex) or census.columns.nlevels != 2:
        raise ValueError("Census columns must be a two-level MultiIndex")
    columns = [column for column in census.columns if column[0] == attribute]
    if not columns:
        raise ValueError(f"Census is missing attribute {attribute!r}")
    values = census[columns].apply(pd.to_numeric, errors="raise")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or array.sum() <= 0:
        raise ValueError(f"Census values for {attribute!r} must be finite and nonnegative")
    state_columns = [column for column in columns if str(column[1]) == state]
    if not state_columns:
        raise ValueError(f"Census is missing {attribute!r} state {state!r}")
    return float(values[state_columns].to_numpy(dtype=float).sum() / array.sum())


def _conditional_table(
    frame: pd.DataFrame,
    weight_column: str,
    state_names: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    grouped = frame.groupby(["hhsize", "hhinc"], observed=False)[weight_column].sum()
    result = {}
    for hhsize in state_names["hhsize"]:
        probabilities = np.array(
            [float(grouped.get((hhsize, income), 0.0)) for income in state_names["hhinc"]]
        )
        if probabilities.sum() <= 0:
            raise ValueError(f"No weighted records for hhsize={hhsize!r}")
        probabilities /= probabilities.sum()
        result[hhsize] = dict(zip(state_names["hhinc"], probabilities.tolist()))
    return result


def prepare_household_training_data(
    seed: pd.DataFrame,
    census: pd.DataFrame,
    attributes: Sequence[str] = HOUSEHOLD_ATTRIBUTES,
    weight_column: str = "_weight",
    negative_income_probability: float | None = None,
    nil_income_state: str = "Nil income",
    negative_income_state: str = "Negative income",
) -> PreparedHouseholdData:
    """Add weighted Negative-income copies of Nil-income donors per hhsize.

    Each household-size stratum keeps its original total weight. Existing
    records get ``1-p`` of the mass and relabelled donors get ``p``.
    """

    attributes = list(dict.fromkeys(attributes))
    missing = (set(attributes) | {weight_column}) - set(seed.columns)
    if missing:
        raise ValueError(f"Seed is missing columns: {sorted(missing)}")
    required_attributes = set(HOUSEHOLD_ATTRIBUTES) - set(attributes)
    if required_attributes:
        raise ValueError(f"attributes are missing: {sorted(required_attributes)}")
    probability = (
        _census_probability(census, "hhinc", negative_income_state)
        if negative_income_probability is None
        else float(negative_income_probability)
    )
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("negative_income_probability must be between zero and one")

    original = seed[attributes + [weight_column]].copy()
    original[attributes] = original[attributes].astype(str)
    original[weight_column] = _weights(original[weight_column], weight_column)
    if original["hhinc"].eq(negative_income_state).any():
        raise ValueError("Seed already contains Negative income")

    total_weight = float(original[weight_column].sum())
    pieces, strata = [], []
    for hhsize, group in original.groupby("hhsize", sort=True, observed=False):
        group = group.copy()
        group_total = float(group[weight_column].sum())
        donors = group.loc[group["hhinc"].eq(nil_income_state)].copy()
        donor_total = float(donors[weight_column].sum())
        if donor_total <= 0:
            raise ValueError(
                f"Cannot create Negative-income donors for hhsize={hhsize!r}: "
                f"no {nil_income_state!r} records"
            )
        group[weight_column] *= 1.0 - probability
        donors["hhinc"] = negative_income_state
        donors[weight_column] *= probability * group_total / donor_total
        pieces.extend([group, donors])
        strata.append(
            {
                "hhsize": str(hhsize),
                "original_weight": group_total,
                "negative_weight": float(donors[weight_column].sum()),
                "donor_rows": len(donors),
            }
        )
    augmented = pd.concat(pieces, ignore_index=True)
    if not np.isclose(augmented[weight_column].sum(), total_weight):
        raise AssertionError("Donor augmentation changed total seed weight")
    state_names = {
        attribute: sorted(augmented[attribute].unique().tolist()) for attribute in attributes
    }
    conditional = _conditional_table(augmented, weight_column, state_names)
    for hhsize, probabilities in conditional.items():
        if not np.isclose(probabilities[negative_income_state], probability):
            raise AssertionError(f"Incorrect Negative-income mass for hhsize={hhsize!r}")
    diagnostics = {
        "assumption": "Negative income uses Nil-income donors within each hhsize",
        "negative_income_probability": probability,
        "original_rows": len(original),
        "augmented_rows": len(augmented),
        "total_weight": total_weight,
        "strata": strata,
    }
    return PreparedHouseholdData(
        augmented, state_names, conditional, probability, diagnostics
    )


class _Generator(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int, hidden_dim: int, dimensions: Sequence[int]):
        super().__init__()
        self.dimensions = list(dimensions)
        self.network = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, sum(dimensions)),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor, temperature: float) -> torch.Tensor:
        logits, parts, start = self.network(torch.cat([noise, condition], dim=1)), [], 0
        for dimension in self.dimensions:
            parts.append(_stable_gumbel_softmax(
                logits[:, start:start + dimension], temperature=temperature
            ))
            start += dimension
        return torch.cat(parts, dim=1)


def _stable_gumbel_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Straight-through Gumbel-Softmax without log(0) samples.

    ``torch.nn.functional.gumbel_softmax`` samples an exponential variate and
    takes its logarithm. A rare zero-valued draw can therefore introduce an
    infinity and poison every subsequent optimizer step. Clamped uniform noise
    avoids that failure while preserving the Gumbel distribution numerically.
    """

    epsilon = torch.finfo(logits.dtype).eps
    uniform = torch.rand_like(logits).clamp(min=epsilon, max=1.0 - epsilon)
    gumbel = -torch.log(-torch.log(uniform))
    soft = torch.softmax((logits + gumbel) / temperature, dim=1)
    indices = soft.argmax(dim=1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(1, indices, 1.0)
    return hard - soft.detach() + soft


def _assert_finite_module(module: nn.Module, name: str) -> None:
    for parameter_name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"{name} parameter {parameter_name!r} contains NaN or infinity"
            )


def _step_with_finite_gradients(
    loss: torch.Tensor,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    context: str,
) -> float:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite {context} loss")
    loss.backward()
    try:
        nn.utils.clip_grad_norm_(
            module.parameters(), max_norm=max_grad_norm, error_if_nonfinite=True
        )
    except RuntimeError as error:
        raise FloatingPointError(f"Non-finite {context} gradient") from error
    optimizer.step()
    _assert_finite_module(module, context)
    return float(loss.detach().cpu())


class _Critic(nn.Module):
    def __init__(self, data_dim: int, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(data_dim + condition_dim, hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([data, condition], dim=1))


class ConditionalHouseholdWGAN:
    """Categorical WGAN-GP conditioned on household size and income."""

    def __init__(
        self,
        state_names: Mapping[str, Sequence[object]],
        income_given_hhsize: Mapping[str, Mapping[str, float]],
        config: WGANConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.state_names = {key: [str(value) for value in values] for key, values in state_names.items()}
        missing = set(HOUSEHOLD_ATTRIBUTES) - set(self.state_names)
        if missing:
            raise ValueError(f"State schema is missing attributes: {sorted(missing)}")
        self.income_given_hhsize = {
            str(size): {str(income): float(value) for income, value in probabilities.items()}
            for size, probabilities in income_given_hhsize.items()
        }
        self.config, self.device = config or WGANConfig(), select_device(device)
        self._indices = {
            attribute: {state: index for index, state in enumerate(self.state_names[attribute])}
            for attribute in HOUSEHOLD_ATTRIBUTES
        }
        dimensions = [len(self.state_names[a]) for a in GENERATED_ATTRIBUTES]
        condition_dim = sum(len(self.state_names[a]) for a in CONDITION_ATTRIBUTES)
        self.generator = _Generator(
            self.config.latent_dim, condition_dim, self.config.hidden_dim, dimensions
        ).to(self.device)
        self.critic = _Critic(sum(dimensions), condition_dim, self.config.hidden_dim).to(self.device)
        self.g_losses: list[float] = []
        self.d_losses: list[float] = []

    def _encode(self, frame: pd.DataFrame, attributes: Sequence[str]) -> torch.Tensor:
        pieces = []
        for attribute in attributes:
            values = frame[attribute].astype(str)
            unknown = set(values.unique()) - set(self._indices[attribute])
            if unknown:
                raise ValueError(f"Unknown {attribute!r} states: {sorted(unknown)}")
            indices = torch.tensor([self._indices[attribute][v] for v in values])
            pieces.append(nn.functional.one_hot(indices, len(self.state_names[attribute])).float())
        return torch.cat(pieces, dim=1)

    def _decode(self, encoded: torch.Tensor) -> pd.DataFrame:
        result, start = {}, 0
        encoded = encoded.detach().cpu()
        for attribute in GENERATED_ATTRIBUTES:
            states = self.state_names[attribute]
            indices = encoded[:, start:start + len(states)].argmax(dim=1).tolist()
            result[attribute] = [states[index] for index in indices]
            start += len(states)
        return pd.DataFrame(result)

    def fit(self, prepared: PreparedHouseholdData, weight_column: str = "_weight") -> dict[str, Any]:
        frame = prepared.frame.reset_index(drop=True)
        real = self._encode(frame, GENERATED_ATTRIBUTES)
        conditions = self._encode(frame, CONDITION_ATTRIBUTES)
        weights = _weights(frame[weight_column], weight_column)
        keys = frame[list(CONDITION_ATTRIBUTES)].astype(str).agg("\x1f".join, axis=1)
        sampling_weights = torch.tensor((weights / weights.groupby(keys).transform("sum")).to_numpy())
        sampler = WeightedRandomSampler(
            sampling_weights.double(), len(frame), replacement=True,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        loader = DataLoader(
            TensorDataset(real, conditions), batch_size=self.config.batch_size, sampler=sampler
        )
        _seed_everything(self.config.seed)
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=self.config.learning_rate, betas=(0.0, 0.9)
        )
        d_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate, betas=(0.0, 0.9)
        )
        self.g_losses.clear(); self.d_losses.clear()
        for epoch in range(self.config.epochs):
            fraction = epoch / max(self.config.epochs - 1, 1)
            temperature = self.config.temperature_start * (
                self.config.temperature_end / self.config.temperature_start
            ) ** fraction
            d_total = g_total = 0.0
            batches = 0
            self.generator.train(); self.critic.train()
            for batch_index, (real_batch, condition_batch) in enumerate(loader):
                real_batch, condition_batch = real_batch.to(self.device), condition_batch.to(self.device)
                for _ in range(self.config.n_critic):
                    d_optimizer.zero_grad(set_to_none=True)
                    noise = torch.randn(len(real_batch), self.config.latent_dim, device=self.device)
                    fake = self.generator(noise, condition_batch, temperature)
                    loss = (
                        -self.critic(real_batch, condition_batch).mean()
                        + self.critic(fake.detach(), condition_batch).mean()
                        + self.config.lambda_gp * self._gradient_penalty(real_batch, fake.detach(), condition_batch)
                    )
                    try:
                        d_total += _step_with_finite_gradients(
                            loss,
                            self.critic,
                            d_optimizer,
                            self.config.max_grad_norm,
                            "critic",
                        )
                    except FloatingPointError as error:
                        raise FloatingPointError(
                            f"WGAN training failed at epoch {epoch + 1}, "
                            f"batch {batch_index + 1}: {error}"
                        ) from error
                g_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(len(real_batch), self.config.latent_dim, device=self.device)
                fake = self.generator(noise, condition_batch, temperature)
                g_loss = -self.critic(fake, condition_batch).mean()
                try:
                    g_total += _step_with_finite_gradients(
                        g_loss,
                        self.generator,
                        g_optimizer,
                        self.config.max_grad_norm,
                        "generator",
                    )
                except FloatingPointError as error:
                    raise FloatingPointError(
                        f"WGAN training failed at epoch {epoch + 1}, "
                        f"batch {batch_index + 1}: {error}"
                    ) from error
                batches += 1
            self.d_losses.append(d_total / max(batches * self.config.n_critic, 1))
            self.g_losses.append(g_total / max(batches, 1))
        return {
            "device": str(self.device), "epochs": self.config.epochs,
            "batches_per_epoch": len(loader), "generator_losses": self.g_losses,
            "critic_losses": self.d_losses,
        }

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(len(real), 1, device=self.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        score = self.critic(interpolated, condition)
        gradient = torch.autograd.grad(
            score, interpolated, torch.ones_like(score), create_graph=True,
            retain_graph=True, only_inputs=True,
        )[0]
        return ((gradient.reshape(len(real), -1).norm(2, dim=1) - 1) ** 2).mean()

    def sample_conditions(self, conditions: pd.DataFrame, chunk_size: int = 100_000, seed: int = 42) -> pd.DataFrame:
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        missing = set(CONDITION_ATTRIBUTES) - set(conditions.columns)
        if missing:
            raise ValueError(f"Conditions are missing columns: {sorted(missing)}")
        _assert_finite_module(self.generator, "generator")
        conditions = conditions[list(CONDITION_ATTRIBUTES)].astype(str).reset_index(drop=True)
        encoded = self._encode(conditions, CONDITION_ATTRIBUTES)
        was_training = self.generator.training
        self.generator.eval(); _seed_everything(seed)
        pieces = []
        try:
            with torch.no_grad():
                for start in range(0, len(conditions), chunk_size):
                    end = min(start + chunk_size, len(conditions))
                    condition = encoded[start:end].to(self.device)
                    noise = torch.randn(end - start, self.config.latent_dim, device=self.device)
                    generated = self.generator(noise, condition, self.config.temperature_end)
                    if not torch.isfinite(generated).all():
                        raise FloatingPointError("Generator produced NaN or infinity")
                    decoded = self._decode(generated)
                    decoded["hhinc"] = conditions.iloc[start:end]["hhinc"].to_numpy()
                    decoded["hhsize"] = conditions.iloc[start:end]["hhsize"].to_numpy()
                    pieces.append(decoded[list(HOUSEHOLD_ATTRIBUTES)])
        finally:
            self.generator.train(was_training)
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=HOUSEHOLD_ATTRIBUTES)


def train_household_wgan(
    prepared: PreparedHouseholdData,
    config: WGANConfig | None = None,
    device: str | torch.device | None = None,
    weight_column: str = "_weight",
) -> WGANTrainingResult:
    model = ConditionalHouseholdWGAN(
        prepared.state_names, prepared.income_given_hhsize, config, device
    )
    training = model.fit(prepared, weight_column)
    diagnostics = {"preparation": prepared.diagnostics, "training": training}
    return WGANTrainingResult(
        model, prepared.state_names, prepared.income_given_hhsize, diagnostics
    )


def _targets(model: ConditionalHouseholdWGAN, targets: Mapping[object, int]) -> dict[str, int]:
    result = {}
    for state, count in targets.items():
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError("Target counts must be nonnegative integers")
        result[str(state)] = int(count)
    unknown = set(result) - set(model.state_names["hhsize"])
    if unknown:
        raise ValueError(f"Model does not support hhsize states: {sorted(unknown)}")
    if sum(result.values()) <= 0:
        raise ValueError("At least one target count must be positive")
    return result


def sample_exact_marginal(
    model: ConditionalHouseholdWGAN,
    attribute: str,
    targets: Mapping[object, int],
    seed: int = 42,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    if attribute != "hhsize":
        raise ValueError("Conditional WGAN exact sampling supports only 'hhsize'")
    targets, rng, pieces = _targets(model, targets), np.random.default_rng(seed), []
    income_states = model.state_names["hhinc"]
    for hhsize, count in targets.items():
        if count == 0:
            continue
        probabilities = model.income_given_hhsize.get(hhsize)
        if probabilities is None:
            raise ValueError(f"No income distribution for hhsize={hhsize!r}")
        values = np.array([probabilities.get(state, 0) for state in income_states])
        if not np.isclose(values.sum(), 1):
            raise ValueError(f"Income probabilities for hhsize={hhsize!r} do not sum to one")
        pieces.append(pd.DataFrame({
            "hhsize": np.repeat(hhsize, count),
            "hhinc": rng.choice(income_states, count, p=values),
        }))
    conditions = pd.concat(pieces, ignore_index=True)
    conditions = conditions.iloc[rng.permutation(len(conditions))].reset_index(drop=True)
    sampled = model.sample_conditions(conditions, chunk_size, seed)
    actual = sampled[attribute].value_counts().to_dict()
    expected = {state: count for state, count in targets.items() if count}
    if actual != expected:
        raise AssertionError(f"Sampled marginal {actual} does not match targets {expected}")
    return sampled


def sample_zonal_exact_marginal(
    model: ConditionalHouseholdWGAN,
    attribute: str,
    zone_targets: pd.DataFrame,
    zone_field: str = "zone_id",
    seed: int = 42,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    if attribute != "hhsize":
        raise ValueError("Conditional WGAN zonal sampling supports only 'hhsize'")
    if zone_targets.empty or not zone_targets.index.is_unique:
        raise ValueError("zone_targets must be nonempty with a unique index")
    targets = zone_targets.copy(); targets.columns = targets.columns.map(str)
    for column in targets:
        values = pd.to_numeric(targets[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Zone target counts must be finite")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Zone target counts must be nonnegative integers")
        targets[column] = values.astype("int64")
    unknown = set(targets.columns) - set(model.state_names["hhsize"])
    if unknown:
        raise ValueError(f"Model does not support hhsize states: {sorted(unknown)}")
    aggregate = {state: int(value) for state, value in targets.sum().items()}
    sampled = sample_exact_marginal(model, attribute, aggregate, seed, chunk_size)
    state_samples = {state: sampled[sampled[attribute].eq(state)].reset_index(drop=True) for state in targets.columns}
    offsets, pieces = {state: 0 for state in targets.columns}, []
    for zone, row in targets.iterrows():
        for state, value in row.items():
            count = int(value)
            if not count:
                continue
            start, end = offsets[state], offsets[state] + count
            piece = state_samples[state].iloc[start:end].copy()
            if len(piece) != count:
                raise AssertionError(f"Insufficient records for hhsize={state!r}")
            piece.insert(0, zone_field, zone); pieces.append(piece); offsets[state] = end
    result = pd.concat(pieces, ignore_index=True)
    actual = result.groupby([zone_field, attribute]).size().unstack(fill_value=0).reindex(
        index=targets.index, columns=targets.columns, fill_value=0
    ).astype("int64")
    actual.index.name, actual.columns.name = targets.index.name, targets.columns.name
    if not actual.equals(targets):
        raise AssertionError("Zonal household-size marginals do not match targets")
    return result


def save_wgan_artifact(
    model: ConditionalHouseholdWGAN,
    metadata: Mapping[str, Any],
    model_path: Path | str,
) -> tuple[Path, Path]:
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError("model_path must use the .pt suffix")
    _assert_finite_module(model.generator, "generator")
    _assert_finite_module(model.critic, "critic")
    if not np.isfinite(model.g_losses).all() or not np.isfinite(model.d_losses).all():
        raise FloatingPointError("Refusing to save an artifact with non-finite losses")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path.with_suffix(".metadata.json")
    torch.save({"generator": model.generator.state_dict(), "critic": model.critic.state_dict()}, model_path)
    artifact = dict(metadata)
    artifact.update({
        "format_version": 1, "state_names": model.state_names,
        "income_given_hhsize": model.income_given_hhsize,
        "config": asdict(model.config), "generator_losses": model.g_losses,
        "critic_losses": model.d_losses,
    })
    metadata_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def load_wgan_artifact(
    model_path: Path | str,
    device: str | torch.device | None = None,
) -> tuple[ConditionalHouseholdWGAN, dict[str, Any]]:
    model_path = Path(model_path); metadata_path = model_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported WGAN artifact format")
    model = ConditionalHouseholdWGAN(
        metadata["state_names"], metadata["income_given_hhsize"],
        WGANConfig(**metadata["config"]), device,
    )
    state = torch.load(model_path, map_location=model.device, weights_only=True)
    model.generator.load_state_dict(state["generator"]); model.critic.load_state_dict(state["critic"])
    model.g_losses = [float(value) for value in metadata.get("generator_losses", [])]
    model.d_losses = [float(value) for value in metadata.get("critic_losses", [])]
    _assert_finite_module(model.generator, "generator")
    _assert_finite_module(model.critic, "critic")
    if not np.isfinite(model.g_losses).all() or not np.isfinite(model.d_losses).all():
        raise FloatingPointError("WGAN artifact contains non-finite losses")
    return model, metadata


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
