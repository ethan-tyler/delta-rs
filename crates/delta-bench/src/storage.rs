use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use deltalake_core::{open_table, open_table_with_storage_options, DeltaTable};
use url::Url;

use crate::cli::StorageBackend;
use crate::error::{BenchError, BenchResult};

pub const TABLE_ROOT_KEY: &str = "table_root";
static ISOLATION_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct StorageConfig {
    backend: StorageBackend,
    options: HashMap<String, String>,
    table_root: Option<Url>,
}

impl StorageConfig {
    pub fn local() -> Self {
        Self {
            backend: StorageBackend::Local,
            options: HashMap::new(),
            table_root: None,
        }
    }

    pub fn new(backend: StorageBackend, options: HashMap<String, String>) -> BenchResult<Self> {
        let table_root = if matches!(backend, StorageBackend::Local) {
            None
        } else {
            let root = options.get(TABLE_ROOT_KEY).ok_or_else(|| {
                BenchError::InvalidArgument(format!(
                    "storage option '{TABLE_ROOT_KEY}=<uri>' is required when backend is not local"
                ))
            })?;
            let parsed = Url::parse(root).map_err(|e| {
                BenchError::InvalidArgument(format!("invalid table_root URI '{root}': {e}"))
            })?;
            validate_table_root_scheme(backend, &parsed)?;
            Some(parsed)
        };

        Ok(Self {
            backend,
            options,
            table_root,
        })
    }

    pub fn backend(&self) -> StorageBackend {
        self.backend
    }

    pub fn is_local(&self) -> bool {
        matches!(self.backend, StorageBackend::Local)
    }

    pub fn object_store_options(&self) -> HashMap<String, String> {
        let mut out = self.options.clone();
        out.remove(TABLE_ROOT_KEY);
        out
    }

    pub fn fixture_table_url(&self, scale: &str, table_name: &str) -> BenchResult<Url> {
        let mut root = self.table_root.clone().ok_or_else(|| {
            BenchError::InvalidArgument(
                "fixture_table_url requires a non-local storage backend".to_string(),
            )
        })?;

        let base_path = root.path().trim_end_matches('/');
        let joined = if base_path.is_empty() || base_path == "/" {
            format!("/{scale}/{table_name}")
        } else {
            format!("{base_path}/{scale}/{table_name}")
        };
        root.set_path(&joined);
        Ok(root)
    }

    pub fn isolated_table_url(
        &self,
        scale: &str,
        base_table_name: &str,
        isolation_key: &str,
    ) -> BenchResult<Url> {
        if self.is_local() {
            return Err(BenchError::InvalidArgument(
                "isolated_table_url requires a non-local storage backend".to_string(),
            ));
        }

        let sanitized_key = sanitize_path_component(isolation_key);
        let table_name = format!(
            "{base_table_name}__isolated__{sanitized_key}__{}",
            next_isolation_suffix()
        );
        self.fixture_table_url(scale, &table_name)
    }

    pub fn table_url_for(
        &self,
        local_table_path: &Path,
        scale: &str,
        table_name: &str,
    ) -> BenchResult<Url> {
        if self.is_local() {
            let absolute_path = if local_table_path.is_absolute() {
                local_table_path.to_path_buf()
            } else {
                std::env::current_dir()?.join(local_table_path)
            };
            Url::from_directory_path(&absolute_path).map_err(|()| {
                BenchError::InvalidArgument(format!(
                    "failed to create table URL for {}",
                    absolute_path.display()
                ))
            })
        } else {
            self.fixture_table_url(scale, table_name)
        }
    }

    pub async fn open_table(&self, table_url: Url) -> BenchResult<DeltaTable> {
        let options = self.object_store_options();
        if options.is_empty() {
            Ok(open_table(table_url).await?)
        } else {
            Ok(open_table_with_storage_options(table_url, options).await?)
        }
    }

    pub async fn try_from_url_for_write(&self, table_url: Url) -> BenchResult<DeltaTable> {
        let options = self.object_store_options();
        if options.is_empty() {
            Ok(DeltaTable::try_from_url(table_url).await?)
        } else {
            Ok(DeltaTable::try_from_url_with_storage_options(table_url, options).await?)
        }
    }
}

pub fn load_backend_profile_options(profile: Option<&str>) -> BenchResult<HashMap<String, String>> {
    load_backend_profile_options_from_root(profile, Path::new("."))
}

pub fn load_backend_profile_options_from_root(
    profile: Option<&str>,
    root: &Path,
) -> BenchResult<HashMap<String, String>> {
    let Some(profile) = profile.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(HashMap::new());
    };
    if profile == "local" {
        return Ok(HashMap::new());
    }
    validate_backend_profile_name(profile)?;

    let file = root
        .join("backends")
        .join(format!("{profile}.env"))
        .to_path_buf();
    if !file.exists() {
        return Err(BenchError::InvalidArgument(format!(
            "backend profile '{profile}' was requested, but profile file is missing: {}",
            file.display()
        )));
    }

    parse_profile_file(&file)
}

fn validate_backend_profile_name(profile: &str) -> BenchResult<()> {
    if !profile
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
    {
        return Err(BenchError::InvalidArgument(format!(
            "invalid backend profile '{profile}'; allowed characters: [A-Za-z0-9._-]"
        )));
    }
    Ok(())
}

fn parse_profile_file(file: &Path) -> BenchResult<HashMap<String, String>> {
    let mut options = HashMap::new();
    let content = fs::read_to_string(file)?;
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            return Err(BenchError::InvalidArgument(format!(
                "invalid backend profile line {} in '{}': expected KEY=VALUE",
                line_no + 1,
                file.display()
            )));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(BenchError::InvalidArgument(format!(
                "invalid backend profile line {} in '{}': key must not be empty",
                line_no + 1,
                file.display()
            )));
        }
        options.insert(key.to_string(), value.trim().to_string());
    }
    Ok(options)
}

fn validate_table_root_scheme(backend: StorageBackend, table_root: &Url) -> BenchResult<()> {
    let expected: &[&str] = match backend {
        StorageBackend::Local => return Ok(()),
        StorageBackend::S3 => &["s3"],
    };

    if expected.iter().any(|scheme| *scheme == table_root.scheme()) {
        return Ok(());
    }

    let expected_display = expected
        .iter()
        .map(|scheme| format!("{scheme}://"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(BenchError::InvalidArgument(format!(
        "table_root '{}' is incompatible with backend {:?}; expected scheme one of: {}",
        table_root, backend, expected_display
    )))
}

fn sanitize_path_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "table".to_string()
    } else {
        trimmed.to_string()
    }
}

fn next_isolation_suffix() -> String {
    let counter = ISOLATION_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos}-{counter}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_alphanumeric_unchanged() {
        assert_eq!(
            sanitize_path_component("hello-world_v2.0"),
            "hello-world_v2.0"
        );
    }

    #[test]
    fn sanitize_special_chars_replaced() {
        assert_eq!(sanitize_path_component("a/b\\c:d"), "a_b_c_d");
    }

    #[test]
    fn sanitize_empty_returns_table() {
        assert_eq!(sanitize_path_component(""), "table");
    }

    #[test]
    fn sanitize_all_special_returns_table() {
        assert_eq!(sanitize_path_component("///"), "table");
    }

    #[test]
    fn sanitize_unicode_replaced() {
        // Trailing non-ASCII char becomes underscore, then gets trimmed
        assert_eq!(sanitize_path_component("caf\u{00e9}"), "caf");
        // Non-ASCII in the middle stays as underscore
        assert_eq!(sanitize_path_component("a\u{00e9}b"), "a_b");
    }

    #[test]
    fn sanitize_leading_trailing_underscores_trimmed() {
        assert_eq!(sanitize_path_component("__name__"), "name");
    }

    #[test]
    fn validate_mismatched_scheme_rejected() {
        let url = Url::parse("gs://bucket/path").unwrap();
        let result = validate_table_root_scheme(StorageBackend::S3, &url);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("incompatible"));
    }

    #[test]
    fn validate_matching_scheme_accepted() {
        let url = Url::parse("s3://bucket/path").unwrap();
        assert!(validate_table_root_scheme(StorageBackend::S3, &url).is_ok());
    }
}
