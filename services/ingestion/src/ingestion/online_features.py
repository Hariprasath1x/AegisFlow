"""Online feature computation using Redis and Lua for AegisFlow M2.

Computes 14 dynamic transaction and velocity features atomically via a single
Redis Lua script, extracts 2 calendar features from transaction event time,
and reads 3 graph features from pre-computed hourly graph snapshots in Redis.
Returns all 19 canonical features defined in aegis_ml.features.definitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import math
from typing import Any, Mapping, Sequence

from aegis_ml.features import definitions as D

# ============================================================================
# Single Atomic Redis Lua Script for Online Feature Updates
# ============================================================================
SCRIPT: str = """
local seen_key = KEYS[1]
local feat_key = KEYS[2]

local txn_id = ARGV[1]
local ts = tonumber(ARGV[2])
local amt = tonumber(ARGV[3])
local has_device = tonumber(ARGV[4])
local has_ip = tonumber(ARGV[5])
local account_id = ARGV[6]
local device_id = ARGV[7]
local ip_id = ARGV[8]
local ttl = tonumber(ARGV[9])

local w_1m = tonumber(ARGV[10])
local w_1h = tonumber(ARGV[11])
local w_24h = tonumber(ARGV[12])
local w_7d = tonumber(ARGV[13])
local dist_w = tonumber(ARGV[14])
local share_w = tonumber(ARGV[15])

-- 1. Idempotency guard: Return 'dup' immediately if transaction was already processed
local is_new = redis.call('SET', seen_key, '1', 'NX', 'EX', ttl)
if not is_new then
    return {'dup'}
end

-- 2. Account timeline & chronological arrival check
local acct_meta = redis.call('HMGET', feat_key, 'last_ts', 'last_id', 'w_count', 'w_mean', 'w_m2', 'sum_1m', 'sum_1h', 'sum_24h', 'sum_7d')
local last_ts_raw = acct_meta[1]
local last_id_raw = acct_meta[2]

if last_ts_raw and last_ts_raw ~= false then
    local last_ts = tonumber(last_ts_raw)
    local last_id = last_id_raw or ''
    if ts < last_ts or (ts == last_ts and txn_id <= last_id) then
        redis.call('DEL', seen_key)
        return {'out_of_order', string.format('Arrival out of order: last (%.6f, %s) vs current (%.6f, %s)', last_ts, last_id, ts, txn_id)}
    end
end

-- 3. Welford's algorithm and recency
local w_count = tonumber(acct_meta[3]) or 0
local w_mean = tonumber(acct_meta[4]) or 0.0
local w_m2 = tonumber(acct_meta[5]) or 0.0

local zscore = 0.0
if w_count >= 2 then
    local variance = w_m2 / (w_count - 1)
    if variance > 0 then
        local std = math.sqrt(variance)
        zscore = (amt - w_mean) / std
    else
        zscore = 0.0
    end
else
    zscore = 0.0
end

-- Update Welford profile with current transaction
w_count = w_count + 1
local delta = amt - w_mean
w_mean = w_mean + (delta / w_count)
local delta2 = amt - w_mean
w_m2 = w_m2 + (delta * delta2)

-- Seconds since previous transaction on account
local seconds_since_prev = -1.0
if last_ts_raw and last_ts_raw ~= false then
    local lts = tonumber(last_ts_raw)
    seconds_since_prev = math.max(0.0, ts - lts)
end

-- 4. Velocity windows helper (removes expired, subtracts expired amount, adds current)
local function update_velocity(zkey, window_sec, cur_sum)
    local threshold = ts - window_sec
    local expired = redis.call('ZRANGEBYSCORE', zkey, '-inf', threshold)
    for _, item in ipairs(expired) do
        local colon = string.find(item, ':', 1, true)
        if colon then
            local exp_amt = tonumber(string.sub(item, colon + 1)) or 0
            cur_sum = cur_sum - exp_amt
        end
    end
    redis.call('ZREMRANGEBYSCORE', zkey, '-inf', threshold)
    local member = txn_id .. ':' .. tostring(amt)
    redis.call('ZADD', zkey, ts, member)
    cur_sum = cur_sum + amt
    if cur_sum < 0 then cur_sum = 0 end
    local count = tonumber(redis.call('ZCARD', zkey))
    if cur_sum == 0 and count > 1 then
        local all_items = redis.call('ZRANGE', zkey, 0, -1)
        local s = 0
        for _, it in ipairs(all_items) do
            local c = string.find(it, ':', 1, true)
            if c then
                s = s + (tonumber(string.sub(it, c + 1)) or 0)
            end
        end
        cur_sum = s
    end
    redis.call('EXPIRE', zkey, ttl)
    return count, cur_sum
end

local c_1m, cur_sum_1m = update_velocity(KEYS[3], w_1m, tonumber(acct_meta[6]) or 0)
local c_1h, cur_sum_1h = update_velocity(KEYS[4], w_1h, tonumber(acct_meta[7]) or 0)
local c_24h, cur_sum_24h = update_velocity(KEYS[5], w_24h, tonumber(acct_meta[8]) or 0)
local c_7d, cur_sum_7d = update_velocity(KEYS[6], w_7d, tonumber(acct_meta[9]) or 0)

-- 5. Distinct devices and IPs (24h)
local dist_thresh = ts - dist_w
redis.call('ZREMRANGEBYSCORE', KEYS[7], '-inf', dist_thresh)
if has_device == 1 and device_id ~= '' then
    redis.call('ZADD', KEYS[7], ts, device_id)
end
redis.call('EXPIRE', KEYS[7], ttl)
local dist_dev = tonumber(redis.call('ZCARD', KEYS[7]))

redis.call('ZREMRANGEBYSCORE', KEYS[8], '-inf', dist_thresh)
if has_ip == 1 and ip_id ~= '' then
    redis.call('ZADD', KEYS[8], ts, ip_id)
end
redis.call('EXPIRE', KEYS[8], ttl)
local dist_ip = tonumber(redis.call('ZCARD', KEYS[8]))

-- 6. Sharing features (7d)
local dev_accounts = 0.0
if has_device == 1 and device_id ~= '' then
    local dev_key = KEYS[9]
    local dev_thresh = ts - share_w
    redis.call('ZREMRANGEBYSCORE', dev_key, '-inf', dev_thresh)
    redis.call('ZADD', dev_key, ts, account_id)
    redis.call('EXPIRE', dev_key, ttl)
    dev_accounts = tonumber(redis.call('ZCARD', dev_key))
end

local ip_accounts = 0.0
if has_ip == 1 and ip_id ~= '' then
    local ip_key = KEYS[10]
    local ip_thresh = ts - share_w
    redis.call('ZREMRANGEBYSCORE', ip_key, '-inf', ip_thresh)
    redis.call('ZADD', ip_key, ts, account_id)
    redis.call('EXPIRE', ip_key, ttl)
    ip_accounts = tonumber(redis.call('ZCARD', ip_key))
end

-- 7. Persist updated account metadata
redis.call('HSET', feat_key,
    'last_ts', string.format('%.6f', ts),
    'last_id', txn_id,
    'w_count', tostring(w_count),
    'w_mean', string.format('%.17g', w_mean),
    'w_m2', string.format('%.17g', w_m2),
    'sum_1m', tostring(cur_sum_1m),
    'sum_1h', tostring(cur_sum_1h),
    'sum_24h', tostring(cur_sum_24h),
    'sum_7d', tostring(cur_sum_7d)
)
redis.call('EXPIRE', feat_key, ttl)

-- 8. Return 14 computed features formatted with full precision
return {
    'ok',
    string.format('%.17g', c_1m),
    string.format('%.17g', cur_sum_1m),
    string.format('%.17g', c_1h),
    string.format('%.17g', cur_sum_1h),
    string.format('%.17g', c_24h),
    string.format('%.17g', cur_sum_24h),
    string.format('%.17g', c_7d),
    string.format('%.17g', cur_sum_7d),
    string.format('%.17g', dist_dev),
    string.format('%.17g', dist_ip),
    string.format('%.17g', dev_accounts),
    string.format('%.17g', ip_accounts),
    string.format('%.17g', zscore),
    string.format('%.17g', seconds_since_prev)
}
"""

_SCRIPT_SHA: str | None = None


def _parse_timestamp(t: Mapping[str, Any]) -> tuple[datetime, float]:
    """Extract and validate UTC datetime and epoch seconds from transaction dict."""
    for key in ("timestamp", "ts", "occurred_at"):
        if key in t and t[key] is not None:
            raw = t[key]
            if isinstance(raw, datetime):
                sec = D.epoch_seconds(raw)
                dt_utc = raw.astimezone(timezone.utc)
                return dt_utc, sec
            if isinstance(raw, (int, float)):
                if not math.isfinite(raw):
                    raise ValueError(f"Timestamp must be a finite number, got {raw}")
                sec = float(raw)
                dt_utc = datetime.fromtimestamp(sec, tz=timezone.utc)
                return dt_utc, sec
            if isinstance(raw, str):
                try:
                    dt = datetime.fromisoformat(raw)
                    sec = D.epoch_seconds(dt)
                    dt_utc = dt.astimezone(timezone.utc)
                    return dt_utc, sec
                except ValueError:
                    sec = float(raw)
                    if not math.isfinite(sec):
                        raise ValueError(f"Timestamp string must represent finite time, got {raw}")
                    dt_utc = datetime.fromtimestamp(sec, tz=timezone.utc)
                    return dt_utc, sec
            raise ValueError(f"Unsupported timestamp type {type(raw)!r}: {raw!r}")
    raise ValueError("Transaction missing timestamp field ('timestamp', 'ts', or 'occurred_at')")


def _parse_amount_minor(t: Mapping[str, Any]) -> int:
    """Extract and validate amount in minor units (integer cents/paise)."""
    if "amount_minor" in t and t["amount_minor"] is not None:
        val = t["amount_minor"]
        if isinstance(val, float):
            if not math.isfinite(val) or not val.is_integer():
                raise ValueError(f"amount_minor float must be a finite integer, got {val}")
            return int(val)
        return int(val)
    if "amount" in t and t["amount"] is not None:
        return D.amount_minor(t["amount"])
    raise ValueError("Transaction missing amount field ('amount_minor' or 'amount')")


def _parse_id(t: Mapping[str, Any], keys: Sequence[str], required: bool = True) -> str:
    """Extract string identifier from transaction dict, rejecting empty / sentinel strings."""
    for k in keys:
        if k in t and t[k] is not None:
            s = str(t[k]).strip()
            if s and s not in ("None", "nan", "null"):
                return s
    if required:
        raise ValueError(f"Transaction missing required field: one of {keys}")
    return ""


def keys_for(t: dict) -> list[str]:
    """Generate the exact 10 Redis keys required by the online feature script."""
    txn_id = _parse_id(t, ("txn_id", "id"), required=True)
    account_id = _parse_id(t, ("account_id",), required=True)
    device_id = _parse_id(t, ("device_id",), required=False)
    ip_id = _parse_id(t, ("ip_id", "ip_address", "ip"), required=False)

    return [
        f"seen:txn:{txn_id}",
        f"feat:acct:{account_id}",
        f"vel:acct:{account_id}:1m",
        f"vel:acct:{account_id}:1h",
        f"vel:acct:{account_id}:24h",
        f"vel:acct:{account_id}:7d",
        f"dist:acct:{account_id}:dev",
        f"dist:acct:{account_id}:ip",
        f"share:dev:{device_id}" if device_id else "share:dev:-",
        f"share:ip:{ip_id}" if ip_id else "share:ip:-",
    ]


def args_for(t: dict) -> list:
    """Generate the exact 15 arguments required by the online feature script."""
    txn_id = _parse_id(t, ("txn_id", "id"), required=True)
    account_id = _parse_id(t, ("account_id",), required=True)
    _, ts_sec = _parse_timestamp(t)
    amt_minor = _parse_amount_minor(t)

    device_id = _parse_id(t, ("device_id",), required=False)
    ip_id = _parse_id(t, ("ip_id", "ip_address", "ip"), required=False)
    has_device = 1 if device_id else 0
    has_ip = 1 if ip_id else 0

    return [
        txn_id,
        f"{ts_sec:.6f}",
        amt_minor,
        has_device,
        has_ip,
        account_id,
        device_id,
        ip_id,
        D.KEY_TTL_S,
        D.WINDOWS["1m"],
        D.WINDOWS["1h"],
        D.WINDOWS["24h"],
        D.WINDOWS["7d"],
        D.DISTINCT_WINDOW_S,
        D.SHARE_WINDOW_S,
    ]


def read_graph(r, account_id: str) -> dict[str, float]:
    """Read pre-computed graph features from Redis hash 'gfeat:acct:{account_id}'.

    Returns dictionary mapping canonical feature names to float values,
    falling back to canonical default values if key or specific fields are missing.
    """
    key = f"gfeat:acct:{account_id}"
    raw = r.hgetall(key)
    out: dict[str, float] = {}
    for feat_name, field_name in D.GRAPH_FIELDS.items():
        val = None
        if raw:
            raw_val = raw.get(field_name)
            if raw_val is None:
                raw_val = raw.get(field_name.encode("utf-8"))
            if raw_val is not None:
                try:
                    if isinstance(raw_val, bytes):
                        raw_val = raw_val.decode("utf-8")
                    parsed = float(raw_val)
                    if math.isfinite(parsed):
                        val = parsed
                except (ValueError, TypeError):
                    val = None
        if val is None:
            val = D.BY_NAME[feat_name].default
        out[feat_name] = val
    return out


def _eval_script(r, keys: list[str], args: list) -> list:
    """Execute SCRIPT using EVALSHA with automatic SCRIPT LOAD caching and fallback."""
    global _SCRIPT_SHA
    if _SCRIPT_SHA is None:
        try:
            _SCRIPT_SHA = r.script_load(SCRIPT)
        except Exception:
            return r.eval(SCRIPT, len(keys), *keys, *args)
    try:
        return r.evalsha(_SCRIPT_SHA, len(keys), *keys, *args)
    except Exception:
        try:
            _SCRIPT_SHA = r.script_load(SCRIPT)
            return r.evalsha(_SCRIPT_SHA, len(keys), *keys, *args)
        except Exception:
            return r.eval(SCRIPT, len(keys), *keys, *args)


def update(r, t: dict) -> dict[str, float] | None:
    """Atomically update Redis state for transaction t and compute all 19 canonical features.

    Returns:
        dict[str, float] with 19 canonical features in exact D.FEATURE_LIST order,
        or None if transaction has already been processed (idempotency guard).

    Raises:
        ValueError: If transaction arrives out-of-order or has invalid fields.
    """
    dt_utc, _ = _parse_timestamp(t)
    account_id = _parse_id(t, ("account_id",), required=True)

    keys = keys_for(t)
    args = args_for(t)

    res = _eval_script(r, keys, args)
    if not res:
        raise RuntimeError("Redis Lua script returned empty response")

    status = res[0].decode("utf-8") if isinstance(res[0], bytes) else str(res[0])
    if status == "dup":
        return None
    if status == "out_of_order":
        err_msg = (
            res[1].decode("utf-8")
            if len(res) > 1 and isinstance(res[1], bytes)
            else (str(res[1]) if len(res) > 1 else "Transaction arrived out of chronological order")
        )
        raise ValueError(f"Out-of-order transaction: {err_msg}")
    if status != "ok":
        raise RuntimeError(f"Unexpected Redis script status: {status}")

    # Extract 14 Lua outputs
    raw_lua_values = res[1:]
    if len(raw_lua_values) != len(D.LUA_OUTPUTS):
        raise RuntimeError(
            f"Expected {len(D.LUA_OUTPUTS)} values from Lua, got {len(raw_lua_values)}"
        )

    lua_features: dict[str, float] = {}
    for name, raw_v in zip(D.LUA_OUTPUTS, raw_lua_values):
        s_val = raw_v.decode("utf-8") if isinstance(raw_v, bytes) else str(raw_v)
        lua_features[name] = float(s_val)

    # Calendar features (2)
    cal_features = D.calendar(dt_utc)

    # Graph features (3)
    graph_features = read_graph(r, account_id)

    # Assemble all 19 features in canonical order
    features: dict[str, float] = {}
    for f in D.FEATURE_LIST:
        if f in lua_features:
            features[f] = lua_features[f]
        elif f in cal_features:
            features[f] = cal_features[f]
        elif f in graph_features:
            features[f] = graph_features[f]
        else:
            features[f] = D.BY_NAME[f].default

    return features

