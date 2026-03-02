use clap::Parser;
use delta_bench::cli::{Args, Command, RunnerMode};

#[test]
fn run_command_accepts_new_selector_flags() {
    let args = Args::parse_from([
        "delta-bench",
        "--backend-profile",
        "s3_locking_vultr",
        "run",
        "--dataset-id",
        "small_files",
        "--target",
        "all",
        "--case-filter",
        "merge_upsert",
        "--runner",
        "all",
    ]);

    assert_eq!(args.backend_profile.as_deref(), Some("s3_locking_vultr"));

    match args.command {
        Command::Run {
            dataset_id,
            case_filter,
            runner,
            ..
        } => {
            assert_eq!(dataset_id.as_deref(), Some("small_files"));
            assert_eq!(case_filter.as_deref(), Some("merge_upsert"));
            assert_eq!(runner, RunnerMode::All);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn data_command_accepts_dataset_id() {
    let args = Args::parse_from([
        "delta-bench",
        "data",
        "--dataset-id",
        "tiny_smoke",
        "--seed",
        "42",
    ]);

    match args.command {
        Command::Data { dataset_id, .. } => {
            assert_eq!(dataset_id.as_deref(), Some("tiny_smoke"));
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn run_command_accepts_tpcds_duckdb_dataset_id() {
    let args = Args::parse_from([
        "delta-bench",
        "run",
        "--dataset-id",
        "tpcds_duckdb",
        "--target",
        "tpcds",
    ]);

    match args.command {
        Command::Run {
            dataset_id, target, ..
        } => {
            assert_eq!(dataset_id.as_deref(), Some("tpcds_duckdb"));
            assert_eq!(target, "tpcds");
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn run_command_defaults_runner_to_all() {
    let args = Args::parse_from(["delta-bench", "run"]);
    match args.command {
        Command::Run { runner, .. } => assert_eq!(runner, RunnerMode::All),
        other => panic!("unexpected command: {other:?}"),
    }
}
