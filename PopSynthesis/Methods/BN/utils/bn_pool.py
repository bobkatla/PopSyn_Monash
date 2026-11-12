"""Census-informed Bayesian-network learning and pool sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import State, TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter
from pgmpy.sampling import BayesianModelSampling

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score


@dataclass(frozen=True)
class BNLearningResult:
    """Result and diagnostics from either supported BN learning mode."""

    model: Any
    state_names: dict[str, list[str]]
    pseudo_counts: dict[str, np.ndarray]
    census_probabilities: dict[str, dict[str, float]]
    prior_strength: float
    diagnostics: dict[str, Any]
    mode: str


# Backwards-compatible name for callers introduced with the census-only API.
CensusPriorBNResult = BNLearningResult


class _PgmpyRoundoffFilter(logging.Filter):
    """Hide only pgmpy's harmless machine-epsilon sampling correction."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            "Probability values don't exactly sum to 1. Differ by:"
        )


def _sample_without_roundoff_warnings(sampling_method, **kwargs):
    """Call pgmpy sampling without logging harmless floating-point residuals."""

    pgmpy_logger = logging.getLogger("pgmpy")
    roundoff_filter = _PgmpyRoundoffFilter()
    pgmpy_logger.addFilter(roundoff_filter)
    try:
        return sampling_method(**kwargs)
    finally:
        pgmpy_logger.removeFilter(roundoff_filter)


def file_sha256(path: Path | str) -> str:
    """Return a stable SHA-256 digest for artifact metadata."""

    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_state_mapping(
    values: Mapping[str, Sequence[object]] | None,
) -> dict[str, list[str]]:
    return {
        str(attribute): list(dict.fromkeys(str(state) for state in states))
        for attribute, states in (values or {}).items()
    }


def _validate_census(census: pd.DataFrame, attributes: Sequence[str]) -> None:
    if not isinstance(census.columns, pd.MultiIndex) or census.columns.nlevels != 2:
        raise ValueError("Census columns must be a two-level MultiIndex")
    missing = [
        attribute
        for attribute in attributes
        if attribute not in census.columns.get_level_values(0)
    ]
    if missing:
        raise ValueError(f"Census is missing attributes: {missing}")


def _census_state_counts(
    census: pd.DataFrame,
    attribute: str,
) -> dict[str, float]:
    columns = [column for column in census.columns if column[0] == attribute]
    counts: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(census[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Census values for {attribute!r} must be finite")
        if (values < 0).any():
            raise ValueError(f"Census values for {attribute!r} must be nonnegative")
        state = str(column[1])
        counts[state] = counts.get(state, 0.0) + float(values.sum())
    return counts


def _model_marginals(model: Any, attributes: Sequence[str]) -> dict[str, dict[str, float]]:
    inference = VariableElimination(model)
    result: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        marginal = inference.query([attribute], show_progress=False)
        states = marginal.state_names[attribute]
        result[attribute] = {
            str(state): float(probability)
            for state, probability in zip(states, marginal.values)
        }
    return result


def _make_cpd_columns_sum_exactly(cpds: Sequence[Any]) -> None:
    """Avoid pgmpy's per-draw warnings for machine-epsilon residuals."""

    for cpd in cpds:
        values = cpd.get_values().copy()
        for column, error in enumerate(1.0 - values.sum(axis=0)):
            if error != 0:
                row = int(np.argmax(values[:, column]))
                values[row, column] += error
        cpd.values = values.reshape(tuple(cpd.cardinality))


def learn_census_prior_bn(
    seed: pd.DataFrame,
    census: pd.DataFrame,
    attributes: Sequence[str],
    extra_states: Mapping[str, Sequence[object]] | None = None,
    weight_column: str = "_weight",
    prior_strength: float | None = None,
    ignored_census_states: Mapping[str, Sequence[object]] | None = None,
    scoring_method: str = "bic-d",
    n_jobs: int = 1,
) -> BNLearningResult:
    """Learn a weighted BN with local Dirichlet priors from census marginals.

    Structure learning uses only states observed in ``seed``. Parameter learning
    expands the state space with ``extra_states``. For a node ``X`` and parent
    configuration ``u``, its pseudo-count is::

        strength * P_census(X=x) * product(P_census(parent=p) for p in u)

    The parent-product assumption only distributes prior strength across CPD
    columns; dependencies are learned from the weighted seed observations.
    """

    attributes = list(dict.fromkeys(attributes))
    if not attributes:
        raise ValueError("At least one attribute is required")
    missing_seed = set(attributes) - set(seed.columns)
    if missing_seed:
        raise ValueError(f"Seed is missing attributes: {sorted(missing_seed)}")
    if weight_column not in seed.columns:
        raise ValueError(f"Seed is missing weight column {weight_column!r}")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")
    _validate_census(census, attributes)

    prepared = seed[attributes + [weight_column]].copy()
    prepared[attributes] = prepared[attributes].astype(str)
    weights = pd.to_numeric(prepared[weight_column], errors="raise")
    if weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Seed weights must be finite")
    if (weights < 0).any():
        raise ValueError("Seed weights must be nonnegative")
    prepared = prepared.rename(columns={weight_column: "_weight"})

    observed_state_names = {
        attribute: sorted(prepared[attribute].dropna().unique().tolist())
        for attribute in attributes
    }
    extras = _normalise_state_mapping(extra_states)
    ignored = {
        attribute: set(states)
        for attribute, states in _normalise_state_mapping(
            ignored_census_states
        ).items()
    }
    unknown_extra_attributes = set(extras) - set(attributes)
    if unknown_extra_attributes:
        raise ValueError(
            f"Extra states reference unknown attributes: {sorted(unknown_extra_attributes)}"
        )

    state_names = {attribute: list(states) for attribute, states in observed_state_names.items()}
    for attribute, states in extras.items():
        for state in states:
            if state not in state_names[attribute]:
                state_names[attribute].append(state)

    raw_counts = {
        attribute: _census_state_counts(census, attribute)
        for attribute in attributes
    }
    raw_totals = {
        attribute: float(sum(counts.values()))
        for attribute, counts in raw_counts.items()
    }
    reference_total = raw_totals[attributes[0]]
    if any(not np.isclose(total, reference_total) for total in raw_totals.values()):
        raise ValueError(f"Census attribute totals do not agree: {raw_totals}")
    probabilities: dict[str, np.ndarray] = {}
    probability_metadata: dict[str, dict[str, float]] = {}
    for attribute in attributes:
        supported = set(state_names[attribute])
        unsupported = {
            state
            for state, count in raw_counts[attribute].items()
            if count > 0 and state not in supported and state not in ignored.get(attribute, set())
        }
        if unsupported:
            raise ValueError(
                f"Positive census states for {attribute!r} are unsupported: "
                f"{sorted(unsupported)}. Add them to extra_states or explicitly ignore them."
            )
        counts = np.array(
            [raw_counts[attribute].get(state, 0.0) for state in state_names[attribute]],
            dtype=float,
        )
        total = float(counts.sum())
        if total <= 0:
            raise ValueError(f"Census total for {attribute!r} must be positive")
        probabilities[attribute] = counts / total
        probability_metadata[attribute] = {
            state: float(probability)
            for state, probability in zip(state_names[attribute], probabilities[attribute])
        }

    if prior_strength is None:
        prior_strength = reference_total
    prior_strength = float(prior_strength)
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior_strength must be finite and positive")

    model = learn_struct_BN_score(
        prepared[attributes],
        state_names=observed_state_names,
        show_struct=False,
        scoring_method=scoring_method,
    )

    pseudo_counts: dict[str, np.ndarray] = {}
    for node in model.nodes():
        parents = sorted(model.get_parents(node))
        if parents:
            parent_probabilities = np.array(
                [
                    float(np.prod(combination))
                    for combination in product(*(probabilities[parent] for parent in parents))
                ],
                dtype=float,
            )
        else:
            parent_probabilities = np.ones(1, dtype=float)
        pseudo_counts[node] = prior_strength * np.outer(
            probabilities[node], parent_probabilities
        )

    estimator = BayesianEstimator(
        model=model,
        data=prepared,
        state_names=state_names,
    )
    # pgmpy 1.0 rebuilds DAG-like models from edges and consequently drops
    # isolated nodes. Those nodes still need unconditional CPDs.
    estimator.model.add_nodes_from(model.nodes())
    cpds = estimator.get_parameters(
        prior_type="dirichlet",
        pseudo_counts=pseudo_counts,
        n_jobs=n_jobs,
        weighted=True,
    )
    _make_cpd_columns_sum_exactly(cpds)
    model.add_cpds(*cpds)
    model.check_model()

    fitted_marginals = _model_marginals(model, attributes)
    diagnostics = {
        "edges": [list(edge) for edge in sorted(model.edges())],
        "census_marginals": probability_metadata,
        "fitted_marginals": fitted_marginals,
        "marginal_difference": {
            attribute: {
                state: fitted_marginals[attribute][state]
                - probability_metadata[attribute][state]
                for state in state_names[attribute]
            }
            for attribute in attributes
        },
    }
    return BNLearningResult(
        model=model,
        state_names=state_names,
        pseudo_counts=pseudo_counts,
        census_probabilities=probability_metadata,
        prior_strength=prior_strength,
        diagnostics=diagnostics,
        mode="census",
    )


def learn_seed_only_bn(
    seed: pd.DataFrame,
    attributes: Sequence[str],
    extra_states: Mapping[str, Sequence[object]] | None = None,
    unseen_state_probabilities: Mapping[str, Mapping[object, float]] | None = None,
    weight_column: str = "_weight",
    scoring_method: str = "bic-d",
    n_jobs: int = 1,
) -> BNLearningResult:
    """Learn structure and observed-state CPDs solely from the weighted seed.

    An unseen state cannot be estimated from seed data. If ``extra_states`` is
    supplied, every extra state therefore requires an explicit probability in
    ``unseen_state_probabilities``. The fitted probabilities of the observed
    states are rescaled proportionally into the remaining mass. No census data
    is read or used in this mode.

    Extra states currently must belong to leaf nodes: without observations, the
    seed cannot identify distributions for children conditioned on that state.
    """

    attributes = list(dict.fromkeys(attributes))
    if not attributes:
        raise ValueError("At least one attribute is required")
    missing_seed = set(attributes) - set(seed.columns)
    if missing_seed:
        raise ValueError(f"Seed is missing attributes: {sorted(missing_seed)}")
    if weight_column not in seed.columns:
        raise ValueError(f"Seed is missing weight column {weight_column!r}")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")

    prepared = seed[attributes + [weight_column]].copy()
    prepared[attributes] = prepared[attributes].astype(str)
    weights = pd.to_numeric(prepared[weight_column], errors="raise")
    if weights.isna().any() or not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Seed weights must be finite")
    if (weights < 0).any():
        raise ValueError("Seed weights must be nonnegative")
    prepared = prepared.rename(columns={weight_column: "_weight"})

    observed_state_names = {
        attribute: sorted(prepared[attribute].dropna().unique().tolist())
        for attribute in attributes
    }
    extras = _normalise_state_mapping(extra_states)
    unknown_extra_attributes = set(extras) - set(attributes)
    if unknown_extra_attributes:
        raise ValueError(
            f"Extra states reference unknown attributes: {sorted(unknown_extra_attributes)}"
        )
    probabilities = {
        str(attribute): {str(state): float(probability) for state, probability in states.items()}
        for attribute, states in (unseen_state_probabilities or {}).items()
    }
    unknown_probability_attributes = set(probabilities) - set(attributes)
    if unknown_probability_attributes:
        raise ValueError(
            "Unseen-state probabilities reference unknown attributes: "
            f"{sorted(unknown_probability_attributes)}"
        )
    for attribute, states in extras.items():
        configured = probabilities.get(attribute, {})
        missing = set(states) - set(configured)
        unexpected = set(configured) - set(states)
        if missing or unexpected:
            raise ValueError(
                f"Unseen-state probabilities for {attribute!r} must exactly match "
                f"extra states; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        values = np.array([configured[state] for state in states], dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or values.sum() >= 1:
            raise ValueError(
                f"Unseen-state probabilities for {attribute!r} must be positive, "
                "finite, and sum to less than one"
            )

    model = learn_struct_BN_score(
        prepared[attributes],
        state_names=observed_state_names,
        show_struct=False,
        scoring_method=scoring_method,
    )
    extra_parent_nodes = {
        attribute
        for attribute, states in extras.items()
        if states and model.get_children(attribute)
    }
    if extra_parent_nodes:
        raise ValueError(
            "Seed-only unseen states require leaf attributes because child "
            "conditionals are not identifiable from the seed. Non-leaf attributes: "
            f"{sorted(extra_parent_nodes)}"
        )

    estimator = MaximumLikelihoodEstimator(
        model=model,
        data=prepared,
        state_names=observed_state_names,
    )
    estimator.model.add_nodes_from(model.nodes())
    learned_cpds = estimator.get_parameters(n_jobs=n_jobs, weighted=True)
    expanded_cpds = []
    state_names = {
        attribute: list(states)
        for attribute, states in observed_state_names.items()
    }
    for cpd in learned_cpds:
        node = cpd.variable
        node_extras = extras.get(node, [])
        if not node_extras:
            expanded_cpds.append(cpd)
            continue
        configured = probabilities[node]
        remaining_probability = 1.0 - sum(configured.values())
        values = cpd.get_values() * remaining_probability
        extra_rows = np.array(
            [
                np.full(values.shape[1], configured[state], dtype=float)
                for state in node_extras
            ]
        )
        expanded_values = np.vstack([values, extra_rows])
        state_names[node].extend(node_extras)
        evidence = cpd.variables[1:]
        expanded_cpds.append(
            TabularCPD(
                variable=node,
                variable_card=len(state_names[node]),
                values=expanded_values,
                evidence=evidence or None,
                evidence_card=[int(cardinality) for cardinality in cpd.cardinality[1:]] or None,
                state_names={
                    variable: (
                        state_names[node]
                        if variable == node
                        else list(cpd.state_names[variable])
                    )
                    for variable in cpd.variables
                },
            )
        )

    _make_cpd_columns_sum_exactly(expanded_cpds)
    model.add_cpds(*expanded_cpds)
    model.check_model()
    fitted_marginals = _model_marginals(model, attributes)
    diagnostics = {
        "edges": [list(edge) for edge in sorted(model.edges())],
        "fitted_marginals": fitted_marginals,
        "unseen_state_probabilities": probabilities,
    }
    return BNLearningResult(
        model=model,
        state_names=state_names,
        pseudo_counts={},
        census_probabilities={},
        prior_strength=0.0,
        diagnostics=diagnostics,
        mode="seed",
    )


def learn_bn(
    seed: pd.DataFrame,
    attributes: Sequence[str],
    mode: str = "seed",
    census: pd.DataFrame | None = None,
    extra_states: Mapping[str, Sequence[object]] | None = None,
    unseen_state_probabilities: Mapping[str, Mapping[object, float]] | None = None,
    weight_column: str = "_weight",
    prior_strength: float | None = None,
    ignored_census_states: Mapping[str, Sequence[object]] | None = None,
    scoring_method: str = "bic-d",
    n_jobs: int = 1,
) -> BNLearningResult:
    """Learn a BN using either seed-only or all-attribute census-prior mode."""

    if mode == "seed":
        return learn_seed_only_bn(
            seed=seed,
            attributes=attributes,
            extra_states=extra_states,
            unseen_state_probabilities=unseen_state_probabilities,
            weight_column=weight_column,
            scoring_method=scoring_method,
            n_jobs=n_jobs,
        )
    if mode == "census":
        if census is None:
            raise ValueError("census is required when mode='census'")
        return learn_census_prior_bn(
            seed=seed,
            census=census,
            attributes=attributes,
            extra_states=extra_states,
            weight_column=weight_column,
            prior_strength=prior_strength,
            ignored_census_states=ignored_census_states,
            scoring_method=scoring_method,
            n_jobs=n_jobs,
        )
    raise ValueError("mode must be 'seed' or 'census'")


def sample_condensed_pool(
    model: Any,
    size: int,
    chunk_size: int = 100_000,
    seed: int = 42,
    required_states: Mapping[str, Sequence[object]] | None = None,
    attributes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Forward-sample in bounded chunks and return unique records with counts."""

    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("size must be a positive integer")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    attributes = list(model.nodes()) if attributes is None else list(attributes)
    missing = set(attributes) - set(model.nodes())
    if missing:
        raise ValueError(f"Model is missing attributes: {sorted(missing)}")

    sampler = BayesianModelSampling(model)
    chunks: list[pd.DataFrame] = []
    remaining = size
    chunk_number = 0
    pgmpy_logger = logging.getLogger("pgmpy")
    roundoff_filter = _PgmpyRoundoffFilter()
    pgmpy_logger.addFilter(roundoff_filter)
    try:
        while remaining:
            draw_size = min(chunk_size, remaining)
            sampled = sampler.forward_sample(
                size=draw_size,
                seed=seed + chunk_number,
                show_progress=False,
            )[attributes]
            chunks.append(
                sampled.groupby(attributes, sort=False, dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            remaining -= draw_size
            chunk_number += 1
    finally:
        pgmpy_logger.removeFilter(roundoff_filter)

    condensed = (
        pd.concat(chunks, ignore_index=True)
        .groupby(attributes, sort=True, dropna=False, as_index=False)["count"]
        .sum()
        .sort_values(attributes, kind="stable")
        .reset_index(drop=True)
    )
    condensed["count"] = condensed["count"].astype("int64")
    if int(condensed["count"].sum()) != size:
        raise AssertionError("Condensed pool count does not equal requested size")

    for attribute, states in _normalise_state_mapping(required_states).items():
        if attribute not in condensed.columns:
            raise ValueError(f"Required-state attribute {attribute!r} is not in the pool")
        observed = set(condensed[attribute].astype(str))
        absent = set(states) - observed
        if absent:
            raise RuntimeError(
                f"Sampled pool is missing required {attribute!r} states: {sorted(absent)}. "
                "Increase the pool size or census-prior strength."
            )
    return condensed


def save_bn_artifact(
    model: Any,
    metadata: Mapping[str, Any],
    model_path: Path | str,
) -> tuple[Path, Path]:
    """Save an XMLBIF model and adjacent JSON metadata."""

    model_path = Path(model_path)
    if model_path.suffix.lower() != ".xml":
        raise ValueError("model_path must use the .xml suffix")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path.with_suffix(".metadata.json")
    artifact_metadata = dict(metadata)
    # XMLBIFWriter sanitizes labels such as ``8+`` and ``1-149``. Always keep
    # the authoritative labels beside the model so loading can restore them.
    artifact_metadata["state_names"] = {
        str(node): list(model.get_cpds(node).state_names[node])
        for node in model.nodes()
    }
    XMLBIFWriter(model).write_xmlbif(str(model_path))
    metadata_path.write_text(
        json.dumps(
            artifact_metadata,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_bn_artifact(model_path: Path | str) -> Any:
    """Load an XMLBIF model with categorical state labels preserved as strings."""

    model_path = Path(model_path)
    model = XMLBIFReader(path=str(model_path)).get_model(state_name_type=str)
    metadata_path = model_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stored_state_names = metadata.get("state_names")
        if stored_state_names:
            missing = set(model.nodes()) - set(stored_state_names)
            if missing:
                raise ValueError(
                    f"BN metadata is missing state names for nodes: {sorted(missing)}"
                )
            for cpd in model.get_cpds():
                restored = {
                    variable: list(stored_state_names[variable])
                    for variable in cpd.variables
                }
                for index, variable in enumerate(cpd.variables):
                    if len(restored[variable]) != int(cpd.cardinality[index]):
                        raise ValueError(
                            f"BN metadata cardinality for {variable!r} does not "
                            "match the XMLBIF model"
                        )
                cpd.store_state_names(cpd.variables, cpd.cardinality, restored)
    model.check_model()
    return model


def sample_exact_marginal(
    model: Any,
    attribute: str,
    targets: Mapping[object, int],
    method: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """Sample records whose selected attribute exactly matches target counts.

    ``conditional`` is valid when the controlled attribute is a root node.
    ``rejection`` is valid for any supported positive-probability state. ``auto``
    chooses conditional sampling for a root and rejection sampling otherwise.
    """

    if attribute not in model.nodes():
        raise ValueError(f"Attribute {attribute!r} is not in the model")
    if method not in {"auto", "conditional", "root", "rejection"}:
        raise ValueError(
            "method must be 'auto', 'conditional', 'root', or 'rejection'"
        )
    normalised_targets: dict[str, int] = {}
    for state, count in targets.items():
        if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 0:
            raise ValueError("Target counts must be nonnegative integers")
        normalised_targets[str(state)] = int(count)
    if sum(normalised_targets.values()) <= 0:
        raise ValueError("At least one target count must be positive")

    valid_states = set(model.get_cpds(attribute).state_names[attribute])
    invalid = set(normalised_targets) - valid_states
    if invalid:
        raise ValueError(f"Model does not support {attribute!r} states: {sorted(invalid)}")

    parents = model.get_parents(attribute)
    chosen_method = "root" if method == "conditional" else method
    chosen_method = (
        "root" if chosen_method == "auto" and not parents else chosen_method
    )
    if chosen_method == "auto":
        chosen_method = "rejection"
    if chosen_method == "root" and parents:
        raise ValueError(f"{attribute!r} has parents; root-clamped sampling is invalid")

    sampler = BayesianModelSampling(model)
    if chosen_method == "root":
        state_order = model.get_cpds(attribute).state_names[attribute]
        state_numbers = {state: index for index, state in enumerate(state_order)}
        values = np.concatenate(
            [
                np.repeat(state_numbers[state], count)
                for state, count in normalised_targets.items()
                if count > 0
            ]
        ).astype(np.int64)
        rng = np.random.default_rng(seed)
        partial = pd.DataFrame({attribute: values[rng.permutation(len(values))]})
        result = _sample_without_roundoff_warnings(
            sampler.forward_sample,
            size=len(partial),
            partial_samples=partial,
            seed=seed,
            show_progress=False,
        )
        # pgmpy intentionally leaves partial-sample columns encoded. Restore
        # their categorical labels before returning the complete records.
        result[attribute] = result[attribute].map(dict(enumerate(state_order)))
    else:
        marginal = VariableElimination(model).query([attribute], show_progress=False)
        probabilities = dict(zip(marginal.state_names[attribute], marginal.values))
        pieces = []
        for offset, (state, count) in enumerate(normalised_targets.items()):
            if count == 0:
                continue
            if probabilities[state] <= 0:
                raise ValueError(f"Cannot reject-sample zero-probability state {state!r}")
            pieces.append(
                _sample_without_roundoff_warnings(
                    sampler.rejection_sample,
                    evidence=[State(attribute, state)],
                    size=count,
                    seed=seed + offset,
                    show_progress=False,
                )
            )
        result = pd.concat(pieces, ignore_index=True)

    actual = result[attribute].astype(str).value_counts().to_dict()
    expected = {state: count for state, count in normalised_targets.items() if count > 0}
    if actual != expected:
        raise AssertionError(f"Sampled marginal {actual} does not match targets {expected}")
    return result.reset_index(drop=True)


def sample_zonal_exact_marginal(
    model: Any,
    attribute: str,
    zone_targets: pd.DataFrame,
    zone_field: str = "zone_id",
    method: str = "auto",
    seed: int = 42,
) -> pd.DataFrame:
    """Build a population matching an attribute marginal in every zone exactly.

    The global target is sampled once and then partitioned by state back to the
    zones. This is equivalent to sampling each zone from the same global BN but
    avoids repeating rejection sampling for every zone.
    """

    if zone_targets.empty:
        raise ValueError("zone_targets must contain at least one zone")
    if not zone_targets.index.is_unique:
        raise ValueError("zone_targets index must contain unique zones")
    targets = zone_targets.copy()
    targets.columns = targets.columns.map(str)
    for column in targets.columns:
        values = pd.to_numeric(targets[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Zone target counts must be finite")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Zone target counts must be nonnegative integers")
        targets[column] = values.astype("int64")

    aggregate_targets = {
        state: int(count)
        for state, count in targets.sum(axis=0).items()
    }
    sampled = sample_exact_marginal(
        model=model,
        attribute=attribute,
        targets=aggregate_targets,
        method=method,
        seed=seed,
    )
    state_samples = {
        state: sampled.loc[sampled[attribute].astype(str).eq(state)].reset_index(drop=True)
        for state in targets.columns
    }
    offsets = {state: 0 for state in targets.columns}
    pieces: list[pd.DataFrame] = []
    for zone, row in targets.iterrows():
        for state, count in row.items():
            count = int(count)
            if count == 0:
                continue
            start = offsets[state]
            end = start + count
            piece = state_samples[state].iloc[start:end].copy()
            if len(piece) != count:
                raise AssertionError(f"Insufficient sampled records for state {state!r}")
            piece.insert(0, zone_field, zone)
            pieces.append(piece)
            offsets[state] = end

    result = pd.concat(pieces, ignore_index=True)
    actual = (
        result.groupby([zone_field, attribute], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=targets.index, columns=targets.columns, fill_value=0)
        .astype("int64")
    )
    actual.index.name = targets.index.name
    actual.columns.name = targets.columns.name
    if not actual.equals(targets):
        raise AssertionError("Zonal sampled marginals do not match their targets")
    return result
