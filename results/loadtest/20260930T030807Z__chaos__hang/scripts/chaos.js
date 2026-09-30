// Scenario (c): the primary provider starts failing mid-test, then recovers.
//
// Steady traffic at RATE for DURATION. At FAULT_AT the controller scenario switches
// mock-primary into FAULT mode through its admin API; at RECOVER_AT it restores it. Every
// request is recorded (CSV output) so the runner can rebuild a per-second timeline: error rate,
// latency and which provider served the traffic.
//
// FAULT: "errors" (every call returns 500), "hang" (every call hangs), or "drop" (streams are
// cut after a few tokens).

import http from 'k6/http';
import { chat, TREND_STATS, writeSummary } from './lib.js';

const RATE = Number(__ENV.RATE || 150);
const DURATION = Number(__ENV.DURATION_SECONDS || 120);
const FAULT_AT = Number(__ENV.FAULT_AT || 40);
const RECOVER_AT = Number(__ENV.RECOVER_AT || 80);
const FAULT = __ENV.FAULT || 'errors';
const MOCK_ADMIN = __ENV.MOCK_ADMIN || 'http://mock-primary:9000';

const FAULTS = {
  errors: { error_rate: 1.0, error_status: 500 },
  hang: { timeout_rate: 1.0 },
  drop: { stream_abort_rate: 1.0, fail_after_tokens: 3 },
};
const HEALTHY = { error_rate: 0, timeout_rate: 0, stream_abort_rate: 0 };

export const options = {
  scenarios: {
    traffic: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: `${DURATION}s`,
      preAllocatedVUs: RATE * 2,
      maxVUs: 3000,
      exec: 'traffic',
    },
    inject_fault: { executor: 'shared-iterations', iterations: 1, vus: 1,
      startTime: `${FAULT_AT}s`, exec: 'injectFault' },
    recover: { executor: 'shared-iterations', iterations: 1, vus: 1,
      startTime: `${RECOVER_AT}s`, exec: 'recover' },
  },
  thresholds: { http_req_failed: ['rate>=0'] },
  summaryTrendStats: TREND_STATS,
  discardResponseBodies: true,
};

export function traffic() {
  // Half streaming: a "drop" fault only affects streams, and TTFB matters for both.
  chat({
    model: 'mock',
    stream: Math.random() < 0.5,
    messages: [{ role: 'user', content: 'Chaos test: explain circuit breakers.' }],
  }, {}, { phase: 'traffic' });
}

function setMock(settings, label) {
  const res = http.patch(`${MOCK_ADMIN}/admin/config`, JSON.stringify(settings), {
    headers: { 'content-type': 'application/json' },
    tags: { phase: 'control' },
  });
  console.log(`CONTROL ${label} at ${Date.now()} status=${res.status}`);
}

export function injectFault() { setMock(FAULTS[FAULT], `fault:${FAULT}`); }
export function recover() { setMock(HEALTHY, 'recover'); }

export const handleSummary = writeSummary;
