#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpcdsQuerySpec {
    pub id: &'static str,
    pub sql_file: &'static str,
    pub enabled: bool,
    pub skip_reason: Option<&'static str>,
}

pub fn phase1_query_catalog() -> Vec<TpcdsQuerySpec> {
    vec![
        TpcdsQuerySpec {
            id: "q03",
            sql_file: "q03.sql",
            enabled: true,
            skip_reason: None,
        },
        TpcdsQuerySpec {
            id: "q07",
            sql_file: "q07.sql",
            enabled: true,
            skip_reason: None,
        },
        TpcdsQuerySpec {
            id: "q64",
            sql_file: "q64.sql",
            enabled: true,
            skip_reason: None,
        },
        TpcdsQuerySpec {
            id: "q72",
            sql_file: "q72.sql",
            enabled: false,
            skip_reason: Some(
                "blocked pending DataFusion issue-tracker parity for TPC-DS q72 semantics",
            ),
        },
    ]
}
