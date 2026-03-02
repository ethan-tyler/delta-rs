#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct SampleStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub stddev_ms: f64,
    pub cv_pct: Option<f64>,
}

pub fn compute_stats(samples_ms: &[f64]) -> Option<SampleStats> {
    if samples_ms.is_empty() {
        return None;
    }
    if samples_ms.iter().any(|value| value.is_nan()) {
        return None;
    }

    let mut values = samples_ms.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let len = values.len();
    let sum: f64 = values.iter().sum();
    let mean_ms = sum / (len as f64);
    let median_ms = if len.is_multiple_of(2) {
        (values[(len / 2) - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    };
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean_ms;
            delta * delta
        })
        .sum::<f64>()
        / (len as f64);
    let stddev_ms = variance.sqrt();
    let cv_pct = if mean_ms.abs() > f64::EPSILON {
        Some((stddev_ms / mean_ms.abs()) * 100.0)
    } else {
        None
    };

    Some(SampleStats {
        min_ms: *values.first().unwrap_or(&0.0),
        max_ms: *values.last().unwrap_or(&0.0),
        mean_ms,
        median_ms,
        stddev_ms,
        cv_pct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(compute_stats(&[]), None);
    }

    #[test]
    fn single_element() {
        let stats = compute_stats(&[42.0]).unwrap();
        assert_eq!(stats.min_ms, 42.0);
        assert_eq!(stats.max_ms, 42.0);
        assert_eq!(stats.mean_ms, 42.0);
        assert_eq!(stats.median_ms, 42.0);
        assert_eq!(stats.stddev_ms, 0.0);
        assert_eq!(stats.cv_pct, Some(0.0));
    }

    #[test]
    fn two_elements_even_median() {
        let stats = compute_stats(&[10.0, 20.0]).unwrap();
        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.max_ms, 20.0);
        assert_eq!(stats.mean_ms, 15.0);
        assert_eq!(stats.median_ms, 15.0);
        assert!(stats.stddev_ms > 0.0);
        assert!(stats.cv_pct.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn odd_count_picks_middle() {
        let stats = compute_stats(&[5.0, 1.0, 3.0]).unwrap();
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, 5.0);
        assert_eq!(stats.mean_ms, 3.0);
        assert_eq!(stats.median_ms, 3.0);
    }

    #[test]
    fn unsorted_input_is_handled() {
        let stats = compute_stats(&[50.0, 10.0, 30.0, 20.0, 40.0]).unwrap();
        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.max_ms, 50.0);
        assert_eq!(stats.mean_ms, 30.0);
        assert_eq!(stats.median_ms, 30.0);
    }

    #[test]
    fn four_elements_even_median() {
        let stats = compute_stats(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(stats.median_ms, 2.5);
    }

    #[test]
    fn nan_input_returns_none() {
        assert_eq!(compute_stats(&[f64::NAN, 1.0, 2.0]), None);
    }

    #[test]
    fn infinity_input_handled() {
        let stats = compute_stats(&[1.0, f64::INFINITY]).unwrap();
        assert_eq!(stats.min_ms, 1.0);
        assert_eq!(stats.max_ms, f64::INFINITY);
    }
}
