// Shared helpers for the Switchyard k6 scenarios.
//
// Every scenario writes its complete k6 summary (all metrics, all tags) to /out/summary.json;
// the runner (loadtest/run.py) mounts the run's results directory at /out.

import http from 'k6/http';
import { Counter, Trend } from 'k6/metrics';

export const GATEWAY = __ENV.TARGET || 'http://gateway:8000';
export const API_KEY = __ENV.API_KEY || 'unused';
export const TREND_STATS = ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max', 'count'];

// Facts only known after the response arrives. k6 can't tag the built-in http_req_duration
// with them, so they get their own metrics.
export const statusCodes = new Counter('status_codes');
export const cacheOutcomes = new Counter('cache_outcomes');
export const latencyByCache = new Trend('latency_by_cache', true);
export const streamTtfb = new Trend('stream_ttfb', true);
export const failures = new Counter('request_failures');
// One sample per request, value = duration, tagged with everything the timeline analyses need
// (ok, code, provider, cache, stream). Scenarios that record CSV output read this metric.
export const requestLog = new Trend('request_log', true);

// k6 only reports a tagged sub-metric in the summary if a threshold references it. These are
// always true: they exist purely to make the per-tag statistics appear.
const TRENDS = new Set(['http_req_duration', 'http_req_waiting', 'latency_by_cache',
  'stream_ttfb', 'request_log']);
const RATES = new Set(['http_req_failed']);
export function report(metric) {
  const name = metric.split('{')[0];
  if (TRENDS.has(name)) return ['max>=0'];
  if (RATES.has(name)) return ['rate>=0'];
  return ['count>=0'];
}

export function chat(body, extraHeaders = {}, tags = {}) {
  const res = http.post(`${GATEWAY}/v1/chat/completions`, JSON.stringify(body), {
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${API_KEY}`,
      ...extraHeaders,
    },
    timeout: '30s',
    tags,
    // A stream that fails after the first byte still has status 200. Only the body shows it
    // (an error event and no [DONE]), so streaming bodies are always read.
    responseType: body.stream ? 'text' : 'none',
  });
  const cache = res.headers['X-Switchyard-Cache'] || 'none';
  const provider = res.headers['X-Switchyard-Provider'] || 'none';
  const ok = res.status === 200 && (!body.stream || (res.body || '').includes('data: [DONE]'));
  const stream = body.stream ? 'true' : 'false';

  statusCodes.add(1, { ...tags, code: String(res.status) });
  cacheOutcomes.add(1, { ...tags, cache });
  requestLog.add(res.timings.duration, {
    ...tags, ok: String(ok), code: String(res.status), provider, cache, stream,
  });
  if (!ok) failures.add(1, { ...tags, code: String(res.status), stream });
  if (ok) {
    latencyByCache.add(res.timings.duration, { ...tags, cache });
    if (body.stream) streamTtfb.add(res.timings.waiting, { ...tags, cache });
  }
  return res;
}

export function writeSummary(data) {
  return {
    '/out/summary.json': JSON.stringify(data, null, 2),
    stdout: `requests=${data.metrics.http_reqs.values.count} ` +
      `rate=${data.metrics.http_reqs.values.rate.toFixed(1)}/s\n`,
  };
}
