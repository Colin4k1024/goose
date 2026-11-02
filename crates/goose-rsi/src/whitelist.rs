//! Frozen mutation-surface whitelist (fail-closed).
//!
//! Issue #1 defines the mutation surface the RSI engine may ever touch:
//!
//! * allowed file surfaces: recipe files (`recipe/*.yaml`, the actual user
//!   directory is `recipes/`), sub-recipe DAGs nested under `recipes/**`, and
//!   prompt templates under `prompts/**`;
//! * allowed config surfaces: non-security config keys (config keys are
//!   allowed unless they match a deny pattern);
//! * denied surfaces: `secrets*.yaml*` (including `.lock` files), provider
//!   credentials, `GOOSE_MODE` / permission configuration, and MCP server
//!   binaries / launch configuration.
//!
//! Everything here is a frozen `const`. There is deliberately no API to
//! extend, replace or override these patterns at runtime: the RSI engine must
//! never be able to widen its own mutation surface.

use crate::proposal::MutationTarget;
use serde::{Deserialize, Serialize};

/// Glob patterns (POSIX, config-dir relative) for file paths the RSI engine
/// may mutate. Anything not matching one of these patterns is rejected.
///
/// Glob syntax: `*` matches any run of characters within one path segment,
/// `**` as a whole segment matches any number of segments.
pub const WHITELISTED_PATH_PATTERNS: &[&str] = &[
    "recipe/*.yaml",
    "recipes/*.yaml",
    // sub-recipe DAG entries nested in subdirectories
    "recipes/**/*.yaml",
    // prompt templates
    "prompts/**",
];

/// Glob patterns for file paths that are always denied, even if they would
/// match a whitelisted pattern. Kept as an independent frozen guard
/// (defense in depth): every pattern here is a security boundary.
pub const DENIED_PATH_PATTERNS: &[&str] = &[
    // secret stores, including lock files and nested copies
    "secrets*.yaml*",
    "**/secrets*.yaml*",
    // MCP server binaries and manifests under the extensions directory
    "extensions/**",
    "**/extensions/**",
];

/// Dotted-key glob patterns for config keys that are always denied. Config
/// keys not matching any of these patterns are considered non-security keys
/// and are allowed.
///
/// Same glob syntax as path patterns, with `.` as the separator.
pub const DENIED_CONFIG_KEY_PATTERNS: &[&str] = &[
    // runtime mode / permission switches (GOOSE_MODE and permission config)
    "GOOSE_MODE",
    "**.GOOSE_MODE",
    "permissions",
    "permissions.**",
    "**.permissions",
    "**.permissions.**",
    // MCP server launch configuration (binaries, args, env)
    "extensions",
    "extensions.**",
    "**.extensions",
    "**.extensions.**",
    // provider selection and credentials: switching providers changes which
    // model executes instructions, and provider auth material must never be
    // writable by the engine itself
    "GOOSE_PROVIDER",
    "GOOSE_PROVIDER.*",
    "**.GOOSE_PROVIDER",
    "**.GOOSE_PROVIDER.*",
    "**.api_key",
    "**.api_key.*",
    "**.api_key.**",
    "**.secret_key",
    "**.secret_key.*",
    "**.credentials",
    "**.credentials.**",
    "**.token",
    "**.token.**",
];

/// Why a mutation target was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum WhitelistCheck {
    /// The target hits the frozen deny list (security boundary).
    #[error("mutation target denied: {target}")]
    Denied { target: String },
    /// The target is not on the frozen whitelist (fail-closed default).
    #[error("mutation target not whitelisted: {target}")]
    NotWhitelisted { target: String },
}

/// Error returned by whitelist validation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(transparent)]
#[error("{check}")]
pub struct ValidationError {
    pub check: WhitelistCheck,
}

impl From<WhitelistCheck> for ValidationError {
    fn from(check: WhitelistCheck) -> Self {
        ValidationError { check }
    }
}

/// Validate a complete target set. Fails closed: an empty target set is a
/// proposal that mutates nothing and is rejected as invalid rather than
/// silently accepted.
pub fn validate_targets(targets: &[MutationTarget]) -> Result<(), ValidationError> {
    if targets.is_empty() {
        return Err(WhitelistCheck::NotWhitelisted {
            target: "<empty target set>".to_string(),
        }
        .into());
    }
    for target in targets {
        validate_target(target)?;
    }
    Ok(())
}

/// Validate a single mutation target against the frozen patterns.
///
/// File paths must be config-dir relative POSIX paths without `.`/`..`
/// components (path traversal cannot smuggle a target past the deny list).
/// This function performs no filesystem I/O.
pub fn validate_target(target: &MutationTarget) -> Result<(), ValidationError> {
    match target {
        MutationTarget::FilePath { path } => {
            let normalized = normalize_relative_path(path)?;
            if matches_any(DENIED_PATH_PATTERNS, &normalized, '/') {
                return Err(WhitelistCheck::Denied {
                    target: target.display(),
                }
                .into());
            }
            if !matches_any(WHITELISTED_PATH_PATTERNS, &normalized, '/') {
                return Err(WhitelistCheck::NotWhitelisted {
                    target: target.display(),
                }
                .into());
            }
            Ok(())
        }
        MutationTarget::ConfigKey { key } => {
            normalize_config_key(key)?;
            if matches_any(DENIED_CONFIG_KEY_PATTERNS, key, '.') {
                return Err(WhitelistCheck::Denied {
                    target: target.display(),
                }
                .into());
            }
            Ok(())
        }
    }
}

fn normalize_relative_path(path: &str) -> Result<String, ValidationError> {
    let invalid = |detail: &str| ValidationError {
        check: WhitelistCheck::NotWhitelisted {
            target: format!("file:{path} ({detail})"),
        },
    };
    if path.is_empty() {
        return Err(invalid("empty path"));
    }
    if path.starts_with('/') {
        return Err(invalid("absolute paths are not allowed"));
    }
    if path.contains('\\') {
        return Err(invalid("backslash separators are not allowed"));
    }
    for segment in path.split('/') {
        if segment.is_empty() {
            return Err(invalid("empty path segment"));
        }
        if segment == "." || segment == ".." {
            return Err(invalid("relative path components are not allowed"));
        }
    }
    Ok(path.to_string())
}

fn normalize_config_key(key: &str) -> Result<String, ValidationError> {
    let invalid = |detail: &str| ValidationError {
        check: WhitelistCheck::NotWhitelisted {
            target: format!("config:{key} ({detail})"),
        },
    };
    if key.is_empty() {
        return Err(invalid("empty key"));
    }
    if key.starts_with('.') || key.ends_with('.') {
        return Err(invalid("leading or trailing dot"));
    }
    for segment in key.split('.') {
        if segment.is_empty() {
            return Err(invalid("empty key segment"));
        }
    }
    Ok(key.to_string())
}

fn matches_any(patterns: &[&str], text: &str, separator: char) -> bool {
    patterns
        .iter()
        .any(|pattern| glob_match(pattern, text, separator))
}

/// Glob matching with `*` (within one segment) and `**` (a whole segment
/// matching any number of segments, including zero).
pub(crate) fn glob_match(pattern: &str, text: &str, separator: char) -> bool {
    let pattern_segments: Vec<&str> = pattern.split(separator).collect();
    let text_segments: Vec<&str> = text.split(separator).collect();
    segments_match(&pattern_segments, &text_segments)
}

fn segments_match(pattern: &[&str], text: &[&str]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(&"**") => {
            segments_match(&pattern[1..], text)
                || (!text.is_empty() && segments_match(pattern, &text[1..]))
        }
        Some(segment) => match text.first() {
            Some(candidate) if segment_glob(segment, candidate) => {
                segments_match(&pattern[1..], &text[1..])
            }
            _ => false,
        },
    }
}

fn segment_glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_chars(&p, &t)
}

fn glob_chars(pattern: &[char], text: &[char]) -> bool {
    match pattern.split_first() {
        None => text.is_empty(),
        Some(('*', rest)) => {
            glob_chars(rest, text)
                || match text.split_first() {
                    Some((_, tail)) => glob_chars(pattern, tail),
                    None => false,
                }
        }
        Some((expected, rest)) => match text.split_first() {
            Some((actual, tail)) if expected == actual => glob_chars(rest, tail),
            _ => false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_star_within_segment() {
        assert!(glob_match("secrets*.yaml*", "secrets.yaml", '/'));
        assert!(glob_match("secrets*.yaml*", "secrets.yaml.lock", '/'));
        assert!(!glob_match("secrets*.yaml*", "secret.yaml", '/'));
    }

    #[test]
    fn glob_double_star_spans_segments() {
        assert!(glob_match(
            "**/secrets*.yaml*",
            "a/b/secrets.yaml.lock",
            '/'
        ));
        assert!(glob_match("**/secrets*.yaml*", "secrets.yaml", '/'));
        assert!(glob_match("recipes/**/*.yaml", "recipes/a/b/c.yaml", '/'));
        assert!(glob_match("recipes/**/*.yaml", "recipes/c.yaml", '/'));
        assert!(!glob_match("recipes/**/*.yaml", "recipes/c.yml", '/'));
        assert!(glob_match("prompts/**", "prompts/system.md", '/'));
        assert!(glob_match("**.permissions.**", "a.permissions.b.c", '.'));
        assert!(glob_match("**.api_key", "providers.openai.api_key", '.'));
    }

    #[test]
    fn glob_star_does_not_cross_separator() {
        assert!(!glob_match("recipe/*.yaml", "recipe/a/b.yaml", '/'));
        assert!(!glob_match("**.api_key", "a.api_key.b", '.'));
    }

    #[test]
    fn whitelist_patterns_are_frozen() {
        // Snapshot test: the mutation surface is pinned. Changing any of
        // these constants must be a deliberate, reviewed decision.
        assert_eq!(
            WHITELISTED_PATH_PATTERNS,
            &[
                "recipe/*.yaml",
                "recipes/*.yaml",
                "recipes/**/*.yaml",
                "prompts/**",
            ]
        );
        assert_eq!(
            DENIED_PATH_PATTERNS,
            &[
                "secrets*.yaml*",
                "**/secrets*.yaml*",
                "extensions/**",
                "**/extensions/**",
            ]
        );
        assert!(DENIED_CONFIG_KEY_PATTERNS.contains(&"GOOSE_MODE"));
        assert!(DENIED_CONFIG_KEY_PATTERNS.contains(&"**.api_key"));
        assert!(DENIED_CONFIG_KEY_PATTERNS.contains(&"extensions.**"));
    }
}
