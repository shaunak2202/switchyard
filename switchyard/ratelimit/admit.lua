-- Authenticate an API key and charge its two token buckets, atomically.
--
-- KEYS[1]  key metadata hash  (rpm, tpm, disabled)
-- KEYS[2]  requests bucket    (tokens, ts)
-- KEYS[3]  tokens bucket      (tokens, ts)
-- ARGV[1]  token cost of this request (estimated)
--
-- Returns {status, retry_after_ms, rpm_limit, rpm_remaining, tpm_limit, tpm_remaining}
--   status  1 admitted
--           0 rate limited (nothing is charged)
--          -1 unknown key
--          -2 key disabled
--          -3 request can never fit: cost exceeds the per-minute token limit
--
-- Each bucket holds up to `limit` tokens and refills continuously at limit/60 per second, so a
-- key can burst up to one minute's allowance and then settles at its steady rate. The request
-- is admitted only if BOTH buckets can pay; otherwise neither is charged. Redis runs a script
-- to completion with no other command interleaved, which makes check-and-decrement atomic
-- across every gateway process. The clock is Redis's own (TIME), so gateway replicas with
-- skewed clocks cannot over-refill a bucket.

local meta = redis.call('HMGET', KEYS[1], 'rpm', 'tpm', 'disabled')
if not meta[1] then
  return {-1, 0, 0, 0, 0, 0}
end
if meta[3] == '1' then
  return {-2, 0, 0, 0, 0, 0}
end

local rpm = tonumber(meta[1])
local tpm = tonumber(meta[2])
local cost = tonumber(ARGV[1])
if cost > tpm then
  return {-3, 0, rpm, 0, tpm, 0}
end

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local window_ms = 60000

local function level(key, capacity)
  local b = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(b[1])
  if tokens == nil then
    return capacity
  end
  local elapsed = math.max(0, now_ms - tonumber(b[2]))
  return math.min(capacity, tokens + elapsed * capacity / window_ms)
end

local requests = level(KEYS[2], rpm)
local tokens = level(KEYS[3], tpm)

if requests >= 1 and tokens >= cost then
  requests = requests - 1
  tokens = tokens - cost
  redis.call('HSET', KEYS[2], 'tokens', tostring(requests), 'ts', now_ms)
  redis.call('HSET', KEYS[3], 'tokens', tostring(tokens), 'ts', now_ms)
  -- An idle bucket is full again after one window, so its state can expire then.
  redis.call('PEXPIRE', KEYS[2], window_ms)
  redis.call('PEXPIRE', KEYS[3], window_ms)
  return {1, 0, rpm, math.floor(requests), tpm, math.floor(tokens)}
end

local wait_ms = 0
if requests < 1 then
  wait_ms = math.max(wait_ms, (1 - requests) * window_ms / rpm)
end
if tokens < cost then
  wait_ms = math.max(wait_ms, (cost - tokens) * window_ms / tpm)
end
return {0, math.ceil(wait_ms), rpm, math.floor(requests), tpm, math.floor(tokens)}
