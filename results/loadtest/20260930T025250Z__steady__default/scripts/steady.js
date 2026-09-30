// Scenario (b): steady load with a realistic cache-hit mix.
//
// Traffic model (deterministic, seeded):
//   * POPULAR_SHARE of requests ask one of POOL_SIZE questions, chosen with a Zipf(ZIPF_S)
//     popularity distribution: a few questions are asked constantly, most rarely.
//   * Of those, REPHRASE_SHARE are re-typed (lower-cased, punctuation dropped) so they miss the
//     exact cache and exercise the semantic tier.
//   * The rest are unique one-off prompts that can never hit.
//   * STREAM_SHARE of requests stream. All use temperature 0 (cache-eligible).

import { chat, report, TREND_STATS, writeSummary } from './lib.js';

const RATE = Number(__ENV.RATE || 200);
const DURATION = __ENV.DURATION || '180s';
const POOL_SIZE = Number(__ENV.POOL_SIZE || 1000);
const ZIPF_S = Number(__ENV.ZIPF_S || 1.1);
const POPULAR_SHARE = Number(__ENV.POPULAR_SHARE || 0.8);
const REPHRASE_SHARE = Number(__ENV.REPHRASE_SHARE || 0.15);
const STREAM_SHARE = Number(__ENV.STREAM_SHARE || 0.5);

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: DURATION,
      preAllocatedVUs: RATE,
      maxVUs: 3000,
    },
  },
  thresholds: Object.fromEntries(
    ['hit-exact', 'hit-semantic', 'miss', 'bypass']
      .flatMap((c) => [`cache_outcomes{cache:${c}}`, `latency_by_cache{cache:${c}}`,
        `stream_ttfb{cache:${c}}`])
      .concat(['http_req_failed', 'request_failures'])
      .map((m) => [m, report(m)])
  ),
  summaryTrendStats: TREND_STATS,
  discardResponseBodies: true,
};

// Seeded PRNG (mulberry32) so the request mix is identical run to run.
function rng(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 20 templates x 50 topics = 1000 distinct, natural questions (no ids or tags in the text:
// they measurably change how the semantic verifier scores a pair; see ADR-016).
const TEMPLATES = [
  'What is the main trade-off in %s?', 'Give three best practices for %s.',
  'Explain a common failure mode in %s.', 'How would you benchmark %s?',
  'What is a good first book on %s?', 'How do I get started with %s?',
  'What are the most common mistakes in %s?', 'Explain %s to a new engineer.',
  'What changed in %s over the last decade?', 'How do I debug problems in %s?',
  'What metrics matter most for %s?', 'Compare two popular approaches to %s.',
  'What interview questions are asked about %s?', 'How does %s work at a high level?',
  'What are the security risks in %s?', 'How do I test code that uses %s?',
  'What is the history of %s?', 'When should I avoid %s?',
  'What tools are used for %s?', 'Summarise the key ideas of %s.',
];
const TOPICS = ['relational databases', 'TCP congestion control', 'garbage collection',
  'HTTP caching', 'message queues', 'TLS certificates', 'distributed tracing', 'unit testing',
  'container orchestration', 'Raft consensus', 'rate limiting', 'load balancing', 'DNS',
  'B-tree indexes', 'event sourcing', 'CRDTs', 'feature flags', 'blue-green deployments',
  'circuit breakers', 'connection pooling', 'vector databases', 'stream processing',
  'OAuth 2.0', 'password hashing', 'memory allocators', 'JIT compilation', 'type inference',
  'regular expressions', 'git internals', 'CI pipelines', 'infrastructure as code',
  'service meshes', 'gRPC', 'GraphQL', 'WebSockets', 'CDNs', 'object storage',
  'write-ahead logging', 'MVCC', 'sharding', 'leader election', 'backpressure',
  'idempotency keys', 'exponential backoff', 'bloom filters', 'consistent hashing',
  'time-series databases', 'column stores', 'data lakes', 'schema migrations'];
function question(i) {
  const template = TEMPLATES[i % TEMPLATES.length];
  return template.replace('%s', TOPICS[Math.floor(i / TEMPLATES.length) % TOPICS.length]);
}

// Cumulative Zipf distribution over the pool, computed once per VU.
const weights = Array.from({ length: POOL_SIZE }, (_, k) => 1 / Math.pow(k + 1, ZIPF_S));
const total = weights.reduce((a, b) => a + b, 0);
const cdf = [];
weights.reduce((acc, w, i) => (cdf[i] = acc + w / total), 0);
function zipf(u) {
  let lo = 0, hi = cdf.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
  return lo;
}

export default function () {
  // One stream of randomness per request, derived from VU and iteration: reproducible.
  const r = rng(__VU * 1_000_003 + __ITER);
  let content;
  if (r() < POPULAR_SHARE) {
    content = question(zipf(r()));
    // The same question typed differently: lower-case, no final punctuation.
    if (r() < REPHRASE_SHARE) content = content.toLowerCase().replace(/[?.]$/, '');
  } else {
    content = `One-off question ${__VU}-${__ITER}-${Math.floor(r() * 1e9)}: explain retries.`;
  }
  chat({
    model: 'mock',
    temperature: 0,
    stream: r() < STREAM_SHARE,
    messages: [{ role: 'user', content }],
  });
}

export const handleSummary = writeSummary;
