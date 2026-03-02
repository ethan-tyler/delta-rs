use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::datasets::NarrowSaleRow;

const REGIONS: [&str; 6] = ["us", "eu", "apac", "latam", "mea", "ca"];

pub fn generate_narrow_sales_rows(seed: u64, rows: usize) -> Vec<NarrowSaleRow> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(rows);
    let start_ts: i64 = 1_700_000_000_000;

    for id in 0..rows {
        let region_idx = rng.gen_range(0..REGIONS.len());
        let skew = (region_idx as i64) * 7;
        let value_i64 = rng.gen_range(-5_000..50_000) + skew;
        let flag = rng.gen_bool(0.35);
        out.push(NarrowSaleRow {
            id: id as i64,
            ts_ms: start_ts + (id as i64 * 60_000),
            region: REGIONS[region_idx].to_string(),
            value_i64,
            flag,
        });
    }

    out
}
