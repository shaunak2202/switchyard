// Scenario (d): rate-limit burst.
//
// One API key with RPM requests/minute. Phase 1 fires BURST requests as fast as BURST_VUS
// connections allow; phase 2 holds SUSTAIN_RATE req/s (well above the refill rate) for
// SUSTAIN_SECONDS. A token bucket should admit exactly its capacity in the burst, then settle
// at RPM/60 per second. Every request is recorded (CSV) so the runner can compare admitted
// requests against the theoretical bound over time.

import { chat, report, TREND_STATS, writeSummary } from './lib.js';

const BURST = Number(__ENV.BURST || 2000);
const BURST_VUS = Number(__ENV.BURST_VUS || 200);
const SUSTAIN_RATE = Number(__ENV.SUSTAIN_RATE || 50);
const SUSTAIN_SECONDS = Number(__ENV.SUSTAIN_SECONDS || 60);

export const options = {
  scenarios: {
    burst: {
      executor: 'shared-iterations',
      iterations: BURST,
      vus: BURST_VUS,
      maxDuration: '60s',
      tags: { phase: 'burst' },
    },
    sustain: {
      executor: 'constant-arrival-rate',
      rate: SUSTAIN_RATE,
      timeUnit: '1s',
      duration: `${SUSTAIN_SECONDS}s`,
      startTime: '10s',
      preAllocatedVUs: SUSTAIN_RATE,
      tags: { phase: 'sustain' },
    },
  },
  thresholds: Object.fromEntries(
    ['burst', 'sustain']
      .flatMap((p) => [`status_codes{phase:${p},code:200}`, `status_codes{phase:${p},code:429}`])
      .map((m) => [m, report(m)])
  ),
  summaryTrendStats: TREND_STATS,
  discardResponseBodies: true,
};

export default function () {
  chat({ model: 'mock', messages: [{ role: 'user', content: 'burst' }], max_tokens: 16 });
}

export const handleSummary = writeSummary;
