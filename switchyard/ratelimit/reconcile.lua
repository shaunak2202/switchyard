-- Correct a key's token bucket once the real token usage is known.
--
-- KEYS[1]  tokens bucket (tokens, ts)
-- ARGV[1]  delta: estimated - actual (positive refunds, negative charges more)
-- ARGV[2]  bucket capacity (the key's tpm)
--
-- The balance may go negative when a response used more than estimated. That debt is repaid
-- by refill, so a key that under-declares max_tokens still converges on its tpm limit.

local capacity = tonumber(ARGV[2])
local delta = tonumber(ARGV[1])
local window_ms = 60000

local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)

local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(b[1])
if tokens == nil then
  tokens = capacity
else
  tokens = math.min(capacity, tokens + math.max(0, now_ms - tonumber(b[2])) * capacity / window_ms)
end

tokens = math.min(capacity, tokens + delta)
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', now_ms)
-- A bucket in debt needs longer than one window to refill.
local ttl = window_ms
if tokens < 0 then
  ttl = window_ms + math.ceil(-tokens * window_ms / capacity)
end
redis.call('PEXPIRE', KEYS[1], ttl)
return math.floor(tokens)
