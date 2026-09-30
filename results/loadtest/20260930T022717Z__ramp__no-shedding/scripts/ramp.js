// Scenario (a): step load to find maximum throughput and where latency breaks down.
//
// Open model (constant-arrival-rate): each step offers a fixed request rate regardless of how
// slow the server gets, which is what real traffic does. A closed model (fixed VUs) would slow
// its own arrival rate as latency rises and hide the knee. Each step is tagged `step:<rate>` so
// the summary carries per-step statistics.
//
// Env: STEPS (comma-separated req/s), STEP_SECONDS, GAP_SECONDS, STREAM (0/1),
//      TARGET (gateway, or a mock for the baseline run).

import { chat, report, TREND_STATS, writeSummary } from './lib.js';

const STEPS = (__ENV.STEPS || '50,100,200,300,400,500,600,800').split(',').map(Number);
const STEP_SECONDS = Number(__ENV.STEP_SECONDS || 30);
const GAP_SECONDS = Number(__ENV.GAP_SECONDS || 5);
const STREAM = __ENV.STREAM === '1';

const scenarios = {};
const thresholds = {};
STEPS.forEach((rate, i) => {
  const step = String(rate).padStart(5, '0');
  scenarios[`step_${step}`] = {
    executor: 'constant-arrival-rate',
    rate,
    timeUnit: '1s',
    duration: `${STEP_SECONDS}s`,
    startTime: `${i * (STEP_SECONDS + GAP_SECONDS)}s`,
    preAllocatedVUs: Math.min(rate, 2000),
    maxVUs: 4000,
    tags: { step },
  };
  // Always-true thresholds: their only purpose is to make k6 report per-step sub-metrics.
  const metrics = [
    ...['http_req_duration', 'http_req_waiting', 'http_reqs', 'http_req_failed',
      'dropped_iterations'].map((m) => `${m}{step:${step}}`),
    `http_req_duration{step:${step},expected_response:true}`,
    ...['200', '429', '502', '503', '504', '0'].map((c) => `status_codes{step:${step},code:${c}}`),
  ];
  for (const m of metrics) thresholds[m] = report(m);
});

export const options = {
  scenarios,
  thresholds,
  summaryTrendStats: TREND_STATS,
  discardResponseBodies: true,
};

// No temperature: every request is a cache bypass, so this measures the proxy path itself.
const body = {
  model: __ENV.MODEL || 'mock',
  stream: STREAM,
  messages: [{ role: 'user', content: 'Summarise the benefits of connection pooling.' }],
};

export default function () {
  chat(body, {}, {});
}

export const handleSummary = writeSummary;
