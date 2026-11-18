"""Champion/challenger registry policy (FR-7).

Two named aliases, never "latest": `champion` is what the scoring service serves,
`challenger` is what it shadow-scores. A new candidate always lands as challenger.
Swapping champion and challenger is a separate, human, audited action - the
scoring service picks the change up on its next registry poll.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ..evaluation.gates import challenger_wins

MODEL = "aegisflow-fraud"
CHAMPION = "champion"
CHALLENGER = "challenger"
PREVIOUS_CHAMPION = "previous_champion"


@dataclass
class Decision:
    changed: bool
    reason: str
    version: str | None = None


def _alias_version(client: MlflowClient, alias: str, model_name: str | None = None):
    """Retrieve the registered model version associated with a named alias.

    Handles genuinely missing aliases and uncreated models safely. Network,
    authentication, permission, or other server errors are re-raised and
    never masked as an alias that simply does not exist.
    """
    target_model = model_name or MODEL
    try:
        return client.get_model_version_by_alias(target_model, alias)
    except MlflowException as exc:
        error_code = getattr(exc, "error_code", None)
        msg = str(exc).lower()
        if error_code in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
            return None
        if "not found" in msg or "does not exist" in msg:
            return None
        raise
    except (KeyError, AttributeError):
        return None


def register_candidate(
    run_id: str,
    model: Any,
    calibrator: Any,
    feature_names: Sequence[str] | list[str],
    metrics: dict[str, Any],
    client: MlflowClient | None = None,
    model_name: str | None = None,
    frequency_encoding: dict[str, Any] | None = None,
) -> Decision:
    """Log the model plus its calibrator and column order as one artifact, register
    it, and set the `challenger` alias. The calibrator travels WITH the model:
    shipping them separately is how a service ends up serving raw margins.
    """
    target_model = model_name or MODEL
    client = client or MlflowClient()

    iso_model = getattr(calibrator, "iso", calibrator)

    def _log_artifacts() -> None:
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.sklearn.log_model(iso_model, artifact_path="calibrator")
        mlflow.log_dict({"feature_names": list(feature_names)}, "model/columns.json")
        if frequency_encoding is not None:
            mlflow.log_dict(frequency_encoding, "model/frequency_encoding.json")

    active = mlflow.active_run()
    if active is not None and active.info.run_id == run_id:
        _log_artifacts()
    else:
        with mlflow.start_run(run_id=run_id):
            _log_artifacts()

    reg_mv = mlflow.register_model(f"runs:/{run_id}/model", target_model)
    version = str(reg_mv.version)

    client.set_registered_model_alias(target_model, CHALLENGER, version)

    for k, v in metrics.items():
        if isinstance(v, float):
            client.set_model_version_tag(target_model, version, k, f"{v:.6f}")
        elif isinstance(v, int):
            client.set_model_version_tag(target_model, version, k, str(v))

    champ = _alias_version(client, CHAMPION, model_name=target_model)
    if champ is None:
        client.set_registered_model_alias(target_model, CHAMPION, version)
        return Decision(
            True,
            f"first model: v{version} is champion and challenger",
            version,
        )

    champ_ver = champ.version if hasattr(champ, "version") else str(champ)
    return Decision(
        True,
        f"v{version} registered as challenger (champion stays v{champ_ver}); run `make compare` to decide",
        version,
    )


def compare(client: MlflowClient | None = None, model_name: str | None = None) -> Decision:
    """Read both aliases' metrics and apply the promotion rule. Prints the reason
    either way - "no change" with a reason is a result, not a failure.
    """
    target_model = model_name or MODEL
    client = client or MlflowClient()
    champ = _alias_version(client, CHAMPION, model_name=target_model)
    chal = _alias_version(client, CHALLENGER, model_name=target_model)

    if champ is None or chal is None:
        return Decision(False, "both champion and challenger aliases must be set")

    champ_ver = str(champ.version)
    chal_ver = str(chal.version)
    if champ_ver == chal_ver:
        return Decision(False, "champion and challenger are the same version", chal_ver)

    def _extract_metrics(v: Any) -> dict[str, float]:
        m: dict[str, float] = {}
        run_id = getattr(v, "run_id", None)
        if run_id:
            try:
                run = client.get_run(run_id)
                data = getattr(run, "data", None)
                run_metrics = getattr(data, "metrics", None) if data else None
                if run_metrics:
                    m.update({k: float(val) for k, val in run_metrics.items()})
            except MlflowException as exc:
                error_code = getattr(exc, "error_code", None)
                msg = str(exc).lower()
                if (
                    error_code in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND")
                    or "not found" in msg
                    or "does not exist" in msg
                    or "deleted" in msg
                ):
                    pass
                else:
                    raise
            except (ConnectionError, TimeoutError, OSError):
                raise
        tags = getattr(v, "tags", None) or {}
        for k, val in tags.items():
            if k not in m:
                try:
                    m[k] = float(val)
                except (ValueError, TypeError):
                    pass
        return m

    champ_metrics = _extract_metrics(champ)
    chal_metrics = _extract_metrics(chal)

    ok, reason = challenger_wins(chal_metrics, champ_metrics)
    return Decision(ok, reason, chal_ver)


def promote_challenger(
    actor: str,
    client: MlflowClient | None = None,
    model_name: str | None = None,
) -> Decision:
    """Human action. Re-checks the rule first, so a stale terminal cannot promote a
    model that stopped winning. Failure-safe ordering preserves rollback capability
    before mutating the active champion alias.
    """
    if not actor or not isinstance(actor, str) or actor.strip() in ("", "unknown"):
        raise ValueError("actor must be an explicitly specified non-empty string and cannot be 'unknown'")

    target_model = model_name or MODEL
    actor = actor.strip()
    client = client or MlflowClient()

    # 1. Re-run compare() immediately before mutations
    decision = compare(client, model_name=target_model)
    if not decision.changed:
        return Decision(False, f"refusing to promote: {decision.reason}", decision.version)

    # 2. Retrieve and retain the current champion version
    old = _alias_version(client, CHAMPION, model_name=target_model)
    if old is None:
        return Decision(False, "cannot promote: current champion alias is not set", decision.version)
    old_ver = str(old.version)

    # 3. Preserve the former champion under previous_champion FIRST
    # If this fails, champion has not been mutated yet and remains safe.
    client.set_registered_model_alias(target_model, PREVIOUS_CHAMPION, old_ver)

    # 4. Only after successful preservation, change champion to the challenger
    try:
        client.set_registered_model_alias(target_model, CHAMPION, decision.version)
    except Exception as exc:
        try:
            curr_champ = _alias_version(client, CHAMPION, model_name=target_model)
            if curr_champ is None or str(curr_champ.version) != old_ver:
                client.set_registered_model_alias(target_model, CHAMPION, old_ver)
        except Exception:
            pass
        raise RuntimeError(
            f"Failed to update champion alias to v{decision.version} ({exc}). "
            f"Active champion remains v{old_ver}."
        ) from exc

    # 5. Apply the promoted_by audit tag
    try:
        client.set_model_version_tag(target_model, decision.version, "promoted_by", actor)
    except Exception as tag_exc:
        # Tagging failed after champion alias change: restore champion to old_ver to maintain audit integrity
        try:
            client.set_registered_model_alias(target_model, CHAMPION, old_ver)
            raise RuntimeError(
                f"Promotion aborted: failed to set 'promoted_by' tag on v{decision.version} ({tag_exc}). "
                f"Restored champion to v{old_ver}."
            ) from tag_exc
        except RuntimeError:
            raise
        except Exception as restore_exc:
            raise RuntimeError(
                f"CRITICAL PARTIAL STATE: failed to set 'promoted_by' tag on v{decision.version} ({tag_exc}), "
                f"AND failed to restore champion to v{old_ver} ({restore_exc}). "
                f"Active champion is v{decision.version} without an audit tag. Immediate operator intervention required."
            ) from tag_exc

    # 6. Return success only when all required operations have completed
    return Decision(
        True,
        f"champion v{old_ver} -> v{decision.version} by {actor}; {decision.reason}",
        decision.version,
    )


def rollback(
    actor: str,
    client: MlflowClient | None = None,
    model_name: str | None = None,
) -> Decision:
    """One command, because the runbook promises one command."""
    if not actor or not isinstance(actor, str) or actor.strip() in ("", "unknown"):
        raise ValueError("actor must be an explicitly specified non-empty string and cannot be 'unknown'")

    target_model = model_name or MODEL
    actor = actor.strip()
    client = client or MlflowClient()
    prev = _alias_version(client, PREVIOUS_CHAMPION, model_name=target_model)
    if prev is None:
        return Decision(False, "no previous_champion alias to roll back to")

    prev_ver = str(prev.version)
    client.set_registered_model_alias(target_model, CHAMPION, prev_ver)
    try:
        client.set_model_version_tag(target_model, prev_ver, "rolled_back_by", actor)
    except Exception as tag_exc:
        raise RuntimeError(
            f"PARTIAL ROLLBACK: champion alias was restored to v{prev_ver}, but failed to set "
            f"'rolled_back_by' tag ({tag_exc})."
        ) from tag_exc

    return Decision(
        True,
        f"champion rolled back to v{prev_ver} by {actor}",
        prev_ver,
    )
