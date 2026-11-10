use crate::config::paths::Paths;
use crate::conversation::Conversation;
use crate::providers::{self, base::Provider};
use anyhow::{anyhow, ensure, Result};
use fs2::FileExt;
use goose_providers::conversation::message::Message;
use goose_providers::model::ModelConfig;
use goose_rsi::apply::{apply, recover, ApplyConfig, Change};
use goose_rsi::fitness::FitnessReport;
use goose_rsi::proposal::{
    FitnessExpectation, MutationProposal, MutationTarget, PayloadChange, ProposalSource,
};
use rmcp::model::Role;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TARGET: &str = "prompts/learned-behavior.md";
const MAX_RULES_BYTES: usize = 8_192;
const COOLDOWN: Duration = Duration::from_secs(300);
const MIN_IMPROVEMENT: f64 = 0.1;
const HOLDOUTS: [(&str, &str); 4] = [
    ("Reply with exactly READY and nothing else.", "READY"),
    ("What is 2 + 2? Reply with only the numeral.", "4"),
    ("请只回复“收到”，不要添加其他内容。", "收到"),
    ("Repeat exactly this word: BLUE", "BLUE"),
];

#[derive(Serialize, Deserialize)]
struct AttemptMarker {
    at_secs: u64,
    feedback_id: String,
}

#[derive(Serialize, Deserialize)]
struct BehaviorEvent {
    at: String,
    outcome: String,
    baseline: f64,
    candidate: f64,
    baseline_holdout: f64,
    candidate_holdout: f64,
    backup_version: Option<u32>,
}

pub fn is_explicit_correction(text: &str) -> bool {
    let lowered = text.to_lowercase();
    [
        "不是我想要",
        "你理解错",
        "你没有按",
        "请改正",
        "更正一下",
        "这不对",
        "不要再",
        "以后请",
        "下次请",
        "还是没有",
        "仍然没有",
        "没看到",
        "没做到",
        "你说的不对",
        "结果不对",
        "回答不对",
        "i meant",
        "you misunderstood",
        "that's not what i",
        "please correct",
        "don't do that again",
        "doesn't work",
        "still missing",
    ]
    .iter()
    .any(|marker| lowered.contains(marker))
}

pub fn previous_assistant_reply(conversation: &Conversation) -> Option<String> {
    conversation
        .messages()
        .iter()
        .rev()
        .find(|message| {
            message.is_user_visible() && !message.user_visible_content().as_concat_text().is_empty()
        })
        .filter(|message| message.role == Role::Assistant)
        .map(|message| message.user_visible_content().as_concat_text())
}

pub fn load_rules() -> Option<String> {
    load_rules_from(&Paths::config_dir())
}

fn load_rules_from(root: &Path) -> Option<String> {
    let contents = fs::read_to_string(root.join(TARGET)).ok()?;
    (!contents.trim().is_empty() && contents.len() <= MAX_RULES_BYTES).then_some(contents)
}

pub fn status_line(root: &Path) -> String {
    let rules = load_rules_from(root);
    let count = rules
        .as_deref()
        .map(|text| text.lines().filter(|line| line.starts_with("- ")).count())
        .unwrap_or(0);
    let events = read_events(root);
    let last = events.last().map_or("none", |event| event.outcome.as_str());
    let mut status = format!(
        "  Automatic behavior evolution: enabled (after explicit corrections)\n  Learned rules: {count} (last attempt: {last})"
    );
    if let Some(rules) = rules {
        status.push_str(&format!("\n{rules}"));
    }
    status
}

pub fn history_lines(root: &Path, limit: usize) -> String {
    let events = read_events(root);
    events
        .iter()
        .rev()
        .take(limit)
        .map(|event| {
            if event.outcome != "applied" && event.outcome != "rejected" {
                return format!("  {} — behavior {}", event.at, event.outcome);
            }
            format!(
                "  {} — behavior {} (training {:.2} → {:.2}; holdout {:.2} → {:.2}){}",
                event.at,
                event.outcome,
                event.baseline,
                event.candidate,
                event.baseline_holdout,
                event.candidate_holdout,
                event
                    .backup_version
                    .map_or(String::new(), |version| format!(" backup v{version}"))
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub async fn evolve_from_correction(
    correction: String,
    previous_reply: String,
    provider_name: String,
    mut model: ModelConfig,
) -> Result<()> {
    let root = Paths::config_dir();
    let rsi_dir = root.join("rsi");
    fs::create_dir_all(&rsi_dir)?;
    let lock = OpenOptions::new()
        .create(true)
        .write(true)
        .open(rsi_dir.join("behavior.lock"))?;
    if lock.try_lock_exclusive().is_err() {
        return Ok(());
    }

    let feedback_id = Sha256::digest(correction.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let marker_path = rsi_dir.join("behavior-last-attempt.json");
    if let Some(marker) = fs::read_to_string(&marker_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<AttemptMarker>(&raw).ok())
    {
        if now.saturating_sub(marker.at_secs) < COOLDOWN.as_secs()
            || (marker.feedback_id == feedback_id
                && now.saturating_sub(marker.at_secs) < 24 * 60 * 60)
        {
            return Ok(());
        }
    }
    fs::write(
        &marker_path,
        serde_json::to_vec(&AttemptMarker {
            at_secs: now,
            feedback_id,
        })?,
    )?;

    let config = ApplyConfig::new(&root);
    recover(&config)?;
    let before = load_rules_from(&root).unwrap_or_default();
    model.max_tokens = Some(512);
    let provider = providers::create(&provider_name, vec![]).await?;
    provider.apply_model_selection(&model).await?;

    let proposal_prompt = format!(
        "Previous assistant response:\n{}\n\nUser correction:\n{}\n\nExisting learned rules:\n{}\n\nExtract ONE short, generalizable behavior rule for future conversations. Do not copy names, paths, credentials, or private details. Do not restate an existing rule. Reply only with JSON: {{\"rule\":\"...\"}}. If there is no reusable rule, return {{\"rule\":\"\"}}.",
        truncate(&previous_reply, 4_000),
        truncate(&correction, 4_000),
        before
    );
    let response = complete_text(
        &provider,
        &model,
        "You identify reusable improvements to an AI assistant's behavior.",
        &proposal_prompt,
    )
    .await?;
    let rule = parse_rule(&response)?;
    if rule.is_empty() || before.lines().any(|line| line == format!("- {rule}")) {
        record_outcome("skipped")?;
        return Ok(());
    }
    let after = if before.trim().is_empty() {
        format!("- {rule}\n")
    } else {
        format!("{}\n- {rule}\n", before.trim_end())
    };
    ensure!(
        after.len() <= MAX_RULES_BYTES,
        "learned behavior file is full"
    );

    let started = Instant::now();
    let replay = format!(
        "The assistant previously replied:\n{}\n\nThe user corrected it:\n{}\n\nRespond to the user's correction.",
        truncate(&previous_reply, 4_000),
        truncate(&correction, 4_000)
    );
    let baseline_answer = answer(&provider, &model, &before, &replay).await?;
    let candidate_answer = answer(&provider, &model, &after, &replay).await?;
    let (baseline, candidate) = judge(
        &provider,
        &model,
        &correction,
        &baseline_answer,
        &candidate_answer,
    )
    .await?;

    let mut baseline_holdout = 0.0;
    let mut candidate_holdout = 0.0;
    for (task, expected) in HOLDOUTS {
        baseline_holdout +=
            f64::from(answer(&provider, &model, &before, task).await?.trim() == expected);
        candidate_holdout +=
            f64::from(answer(&provider, &model, &after, task).await?.trim() == expected);
    }
    baseline_holdout /= HOLDOUTS.len() as f64;
    candidate_holdout /= HOLDOUTS.len() as f64;
    let passed = passes_behavior_gate(baseline, candidate, baseline_holdout, candidate_holdout);

    let proposal = MutationProposal::new(
        "Behavior rule learned from explicit user correction",
        ProposalSource::Session,
        vec![MutationTarget::FilePath {
            path: TARGET.to_string(),
        }],
        FitnessExpectation {
            baseline_score: baseline,
            minimum_score: baseline + MIN_IMPROVEMENT,
        },
        vec![PayloadChange::FileChange {
            path: TARGET.to_string(),
            content: after.clone(),
        }],
    );
    proposal.validate()?;
    let report = FitnessReport {
        proposal_id: proposal.id.to_string(),
        passed,
        score: candidate,
        failed_phases: if passed {
            vec![]
        } else {
            vec!["behavior-gate".to_string()]
        },
        duration: started.elapsed(),
        raw_log_ref: rsi_dir.join("behavior-events.jsonl"),
        timestamp: SystemTime::now(),
    };
    let backup_version = if passed {
        Some(
            apply(
                &config,
                &proposal,
                &report,
                &[Change::FileChange {
                    path: TARGET.to_string(),
                    content: after,
                }],
                None,
            )?
            .backup_version,
        )
    } else {
        None
    };
    append_event(
        &rsi_dir.join("behavior-events.jsonl"),
        &BehaviorEvent {
            at: chrono::Utc::now().to_rfc3339(),
            outcome: if passed { "applied" } else { "rejected" }.to_string(),
            baseline,
            candidate,
            baseline_holdout,
            candidate_holdout,
            backup_version,
        },
    )?;
    Ok(())
}

pub fn record_failure() -> Result<()> {
    record_outcome("error")
}

fn record_outcome(outcome: &str) -> Result<()> {
    let rsi_dir = Paths::config_dir().join("rsi");
    fs::create_dir_all(&rsi_dir)?;
    append_event(
        &rsi_dir.join("behavior-events.jsonl"),
        &BehaviorEvent {
            at: chrono::Utc::now().to_rfc3339(),
            outcome: outcome.to_string(),
            baseline: 0.0,
            candidate: 0.0,
            baseline_holdout: 0.0,
            candidate_holdout: 0.0,
            backup_version: None,
        },
    )
}

async fn answer(
    provider: &Arc<dyn Provider>,
    model: &ModelConfig,
    rules: &str,
    prompt: &str,
) -> Result<String> {
    complete_text(
        provider,
        model,
        &format!("You are a helpful assistant. Follow the user's instructions.\nLearned behavior rules:\n{rules}"),
        prompt,
    )
    .await
}

async fn complete_text(
    provider: &Arc<dyn Provider>,
    model: &ModelConfig,
    system: &str,
    prompt: &str,
) -> Result<String> {
    let (response, _) = provider
        .complete(model, system, &[Message::user().with_text(prompt)], &[])
        .await?;
    response
        .content
        .iter()
        .find_map(|content| content.as_text().map(str::to_string))
        .ok_or_else(|| anyhow!("model returned no text"))
}

async fn judge(
    provider: &Arc<dyn Provider>,
    model: &ModelConfig,
    correction: &str,
    baseline: &str,
    candidate: &str,
) -> Result<(f64, f64)> {
    let prompt = format!(
        "User correction:\n{}\n\nAnswer A:\n{}\n\nAnswer B:\n{}\n\nRate each answer from 0 to 1 for following the user's correction, without rewarding verbosity. Return only JSON: {{\"a\":0.0,\"b\":0.0}}.",
        truncate(correction, 4_000), truncate(baseline, 4_000), truncate(candidate, 4_000)
    );
    let result = complete_text(
        provider,
        model,
        "You are an impartial evaluator. Answer A and B are unlabeled alternatives.",
        &prompt,
    )
    .await?;
    let parsed = parse_json(&result)?;
    let a = parsed
        .get("a")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow!("missing baseline score"))?;
    let b = parsed
        .get("b")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| anyhow!("missing candidate score"))?;
    ensure!(
        (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b),
        "invalid behavior scores"
    );
    Ok((a, b))
}

fn parse_rule(response: &str) -> Result<String> {
    let value = parse_json(response)?;
    let rule = value
        .get("rule")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("missing behavior rule"))?
        .trim();
    ensure!(
        rule.len() <= 300 && !rule.contains('\n') && !rule.contains("http"),
        "invalid behavior rule"
    );
    Ok(rule.to_string())
}

fn parse_json(text: &str) -> Result<serde_json::Value> {
    let start = text
        .find('{')
        .ok_or_else(|| anyhow!("model returned no JSON"))?;
    let end = text
        .rfind('}')
        .ok_or_else(|| anyhow!("model returned no JSON"))?;
    Ok(serde_json::from_str(&text[start..=end])?)
}

fn truncate(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn passes_behavior_gate(
    baseline: f64,
    candidate: f64,
    baseline_holdout: f64,
    candidate_holdout: f64,
) -> bool {
    candidate > baseline + MIN_IMPROVEMENT
        && candidate_holdout >= 0.75
        && candidate_holdout >= baseline_holdout
}

fn append_event(path: &Path, event: &BehaviorEvent) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(event)?)?;
    Ok(())
}

fn read_events(root: &Path) -> Vec<BehaviorEvent> {
    fs::read_to_string(root.join("rsi/behavior-events.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_trigger_is_specific() {
        assert!(is_explicit_correction("这不是我想要的结果，以后请先验证"));
        assert!(is_explicit_correction("I meant the installed app"));
        assert!(is_explicit_correction("这个安装包左侧还是没有 RSI 进化"));
        assert!(!is_explicit_correction("请帮我总结今天的会议"));
    }

    #[test]
    fn only_the_immediate_previous_visible_message_can_trigger_evolution() {
        let assistant = Message::assistant().with_text("The app is ready.");
        let user = Message::user().with_text("Another request");
        let conversation = Conversation::new_unvalidated([assistant.clone()]);
        assert_eq!(
            previous_assistant_reply(&conversation).as_deref(),
            Some("The app is ready.")
        );
        let conversation = Conversation::new_unvalidated([assistant, user]);
        assert!(previous_assistant_reply(&conversation).is_none());
    }

    #[test]
    fn malformed_or_oversized_rules_are_rejected() {
        assert!(parse_rule("{\"rule\":\"Check the result before claiming success.\"}").is_ok());
        assert!(parse_rule("{\"rule\":\"first\\nsecond\"}").is_err());
        assert!(parse_rule("{\"rule\":\"http://example.com\"}").is_err());
    }

    #[test]
    fn behavior_gate_requires_improvement_without_holdout_regression() {
        assert!(passes_behavior_gate(0.4, 0.8, 0.5, 1.0));
        assert!(!passes_behavior_gate(0.4, 0.45, 1.0, 1.0));
        assert!(!passes_behavior_gate(0.4, 0.8, 0.5, 0.5));
        assert!(!passes_behavior_gate(0.4, 0.4, 1.0, 1.0));
        assert!(!passes_behavior_gate(0.4, 0.8, 1.0, 0.5));
        assert!(!passes_behavior_gate(0.4, 0.8, 0.0, 0.0));
    }

    #[test]
    fn accepted_rule_reaches_the_next_system_prompt() {
        let root = tempfile::tempdir().unwrap();
        let root_string = root.path().display().to_string();
        let _guard = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(root_string.as_str()))]);
        let config_dir = root.path().join("config");
        fs::create_dir_all(config_dir.join("prompts")).unwrap();
        fs::write(
            config_dir.join(TARGET),
            "- Verify the result before claiming success.\n",
        )
        .unwrap();

        let prompt = crate::agents::PromptManager::new()
            .builder()
            .with_goose_mode(crate::config::GooseMode::Auto)
            .build()
            .unwrap();
        assert!(prompt.contains("Verify the result before claiming success."));
    }
}
