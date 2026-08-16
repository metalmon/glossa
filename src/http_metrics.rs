//! Hand-rolled Prometheus HTTP request metrics for the streamable-http server: request/response
//! counters, an in-flight gauge, a fixed-bucket latency histogram, and an auth-rejection counter.
//! Shared (behind an `Arc`) between the axum middleware that records each request and
//! `GlossaServer::metrics_text`, which renders these next to the existing index/graph gauges. Kept
//! dependency-free and in the same exposition style as the rest of `/metrics`.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::fmt::Write as _;

/// Upper bounds (seconds) of the cumulative latency histogram buckets. A `+Inf` bucket (= total
/// count) is appended at render time.
const DURATION_BUCKETS_SECS: [f64; 11] =
    [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

#[derive(Debug)]
pub struct HttpMetrics {
    requests_total: AtomicU64,
    responses_2xx: AtomicU64,
    responses_3xx: AtomicU64,
    responses_4xx: AtomicU64,
    responses_5xx: AtomicU64,
    in_flight: AtomicI64,
    auth_rejected: AtomicU64,
    /// Cumulative bucket counts: `duration_buckets[i]` counts requests with latency ≤
    /// `DURATION_BUCKETS_SECS[i]`. Stored cumulative so `record` bumps every bucket a sample falls
    /// under (matching Prometheus histogram semantics directly).
    duration_buckets: [AtomicU64; DURATION_BUCKETS_SECS.len()],
    /// Sum of observed latencies in milliseconds (integer accumulation; rendered as seconds).
    duration_sum_ms: AtomicU64,
    duration_count: AtomicU64,
}

impl Default for HttpMetrics {
    fn default() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            responses_2xx: AtomicU64::new(0),
            responses_3xx: AtomicU64::new(0),
            responses_4xx: AtomicU64::new(0),
            responses_5xx: AtomicU64::new(0),
            in_flight: AtomicI64::new(0),
            auth_rejected: AtomicU64::new(0),
            duration_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            duration_sum_ms: AtomicU64::new(0),
            duration_count: AtomicU64::new(0),
        }
    }
}

impl HttpMetrics {
    pub fn inc_in_flight(&self) {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_in_flight(&self) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn inc_auth_rejected(&self) {
        self.auth_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a finished request: bump the total, the status-class counter, and the latency
    /// histogram. `status` is the HTTP status code; `duration_secs` the wall-clock time served.
    pub fn record(&self, status: u16, duration_secs: f64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        let class = match status / 100 {
            2 => &self.responses_2xx,
            3 => &self.responses_3xx,
            4 => &self.responses_4xx,
            _ => &self.responses_5xx, // 5xx and any non-standard code
        };
        class.fetch_add(1, Ordering::Relaxed);
        for (i, le) in DURATION_BUCKETS_SECS.iter().enumerate() {
            if duration_secs <= *le {
                self.duration_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        self.duration_sum_ms
            .fetch_add((duration_secs * 1000.0) as u64, Ordering::Relaxed);
        self.duration_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Prometheus text-exposition for these metrics (no trailing gaps; caller concatenates).
    pub fn render(&self) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mut s = String::new();
        let _ = write!(
            s,
            "# HELP glossa_http_requests_total HTTP requests received\n\
             # TYPE glossa_http_requests_total counter\n\
             glossa_http_requests_total {}\n\
             # HELP glossa_http_responses_total HTTP responses by status class\n\
             # TYPE glossa_http_responses_total counter\n\
             glossa_http_responses_total{{class=\"2xx\"}} {}\n\
             glossa_http_responses_total{{class=\"3xx\"}} {}\n\
             glossa_http_responses_total{{class=\"4xx\"}} {}\n\
             glossa_http_responses_total{{class=\"5xx\"}} {}\n\
             # HELP glossa_http_requests_in_flight HTTP requests currently being served\n\
             # TYPE glossa_http_requests_in_flight gauge\n\
             glossa_http_requests_in_flight {}\n\
             # HELP glossa_mcp_auth_rejected_total /mcp requests rejected with 401 (missing/invalid token)\n\
             # TYPE glossa_mcp_auth_rejected_total counter\n\
             glossa_mcp_auth_rejected_total {}\n",
            g(&self.requests_total),
            g(&self.responses_2xx),
            g(&self.responses_3xx),
            g(&self.responses_4xx),
            g(&self.responses_5xx),
            self.in_flight.load(Ordering::Relaxed),
            g(&self.auth_rejected),
        );
        s.push_str(
            "# HELP glossa_http_request_duration_seconds HTTP request latency\n\
             # TYPE glossa_http_request_duration_seconds histogram\n",
        );
        for (i, le) in DURATION_BUCKETS_SECS.iter().enumerate() {
            let _ = writeln!(
                s,
                "glossa_http_request_duration_seconds_bucket{{le=\"{le}\"}} {}",
                g(&self.duration_buckets[i])
            );
        }
        let count = g(&self.duration_count);
        let _ = writeln!(
            s,
            "glossa_http_request_duration_seconds_bucket{{le=\"+Inf\"}} {count}"
        );
        let _ = writeln!(
            s,
            "glossa_http_request_duration_seconds_sum {:.3}",
            g(&self.duration_sum_ms) as f64 / 1000.0
        );
        let _ = writeln!(s, "glossa_http_request_duration_seconds_count {count}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_counts_classes_and_histogram() {
        let m = HttpMetrics::default();
        m.record(200, 0.003); // 2xx, falls in the smallest bucket
        m.record(200, 0.4); // 2xx, ≤ 0.5
        m.record(401, 0.001); // 4xx
        m.record(503, 12.0); // 5xx, only in +Inf
        m.inc_auth_rejected();
        let out = m.render();

        assert!(out.contains("glossa_http_requests_total 4"));
        assert!(out.contains("glossa_http_responses_total{class=\"2xx\"} 2"));
        assert!(out.contains("glossa_http_responses_total{class=\"4xx\"} 1"));
        assert!(out.contains("glossa_http_responses_total{class=\"5xx\"} 1"));
        assert!(out.contains("glossa_mcp_auth_rejected_total 1"));
        // Cumulative histogram: the 0.005 bucket holds the two ≤5ms samples (0.003, 0.001).
        assert!(out.contains("le=\"0.005\"} 2"));
        // ≤0.5 holds three (0.003, 0.001, 0.4); the 12s sample only reaches +Inf.
        assert!(out.contains("le=\"0.5\"} 3"));
        assert!(out.contains("le=\"+Inf\"} 4"));
        assert!(out.contains("glossa_http_request_duration_seconds_count 4"));
    }

    #[test]
    fn in_flight_tracks_up_and_down() {
        let m = HttpMetrics::default();
        m.inc_in_flight();
        m.inc_in_flight();
        m.dec_in_flight();
        assert!(m.render().contains("glossa_http_requests_in_flight 1"));
    }
}
