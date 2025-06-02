use std::sync::Arc;

use anyhow::Result;
use forge_domain::{
    extract_tag_content, Agent, ChatCompletionMessage, Compact, CompactionService, Context,
    ContextMessage, ProviderService, Role, TemplateService,
};
use futures::StreamExt;
use tracing::{debug, info};

/// Handles the compaction of conversation contexts to manage token usage
#[derive(Clone)]
pub struct ForgeCompactionService<T, P> {
    template: Arc<T>,
    provider: Arc<P>,
}

impl<T: TemplateService, P: ProviderService> ForgeCompactionService<T, P> {
    /// Creates a new ContextCompactor instance
    pub fn new(template: Arc<T>, provider: Arc<P>) -> Self {
        Self { template, provider }
    }

    /// Apply compaction to the context if requested
    pub async fn compact_context(&self, agent: &Agent, context: Context) -> Result<Context> {
        // Return early if agent doesn't have compaction configured
        if let Some(ref compact) = agent.compact {
            debug!(agent_id = %agent.id, "Context compaction triggered");

            // Identify and compress the first compressible sequence
            // Get all compressible sequences, considering the preservation window
            match find_sequence(&context, compact.retention_window)
                .into_iter()
                .next()
            {
                Some(sequence) => {
                    debug!(agent_id = %agent.id, "Compressing sequence");
                    self.compress_single_sequence(compact, context, sequence)
                        .await
                }
                None => {
                    debug!(agent_id = %agent.id, "No compressible sequences found");
                    Ok(context)
                }
            }
        } else {
            Ok(context)
        }
    }

    /// Compress a single identified sequence of assistant messages
    async fn compress_single_sequence(
        &self,
        compact: &Compact,
        mut context: Context,
        sequence: (usize, usize),
    ) -> Result<Context> {
        let (start, end) = sequence;

        // Extract the sequence to summarize
        let sequence_messages = &context.messages[start..=end].to_vec();

        // Generate summary for this sequence
        let summary = self
            .generate_summary_for_sequence(compact, sequence_messages)
            .await?;

        // Log the summary for debugging
        info!(
            summary = %summary,
            sequence_start = sequence.0,
            sequence_end = sequence.1,
            sequence_length = sequence_messages.len(),
            "Created context compaction summary"
        );

        let summary = format!(
            r#"Continuing from a prior analysis. Below is a compacted summary of the ongoing session. Use this summary as authoritative context for your reasoning and decision-making. You do not need to repeat or reanalyze it unless specifically asked: <summary>{summary}</summary> Proceed based on this context.
        "#
        );

        // Replace the sequence with a single summary message using splice
        // This removes the sequence and inserts the summary message in-place
        context.messages.splice(
            start..=end,
            std::iter::once(ContextMessage::assistant(summary, None)),
        );

        Ok(context)
    }

    /// Generate a summary for a specific sequence of assistant messages
    async fn generate_summary_for_sequence(
        &self,
        compact: &Compact,
        messages: &[ContextMessage],
    ) -> Result<String> {
        // Create a temporary context with just the sequence for summarization
        let sequence_context = messages
            .iter()
            .fold(Context::default(), |ctx, msg| ctx.add_message(msg.clone()));

        // Render the summarization prompt
        let summary_tag = compact.summary_tag.as_ref().cloned().unwrap_or_default();
        let ctx = serde_json::json!({
            "context": sequence_context.to_text(),
            "summary_tag": summary_tag
        });

        let prompt = self.template.render(
            compact
                .prompt
                .as_deref()
                .unwrap_or("{{> system-prompt-context-summarizer.hbs}}"),
            &ctx,
        )?;

        // Create a new context
        let mut context = Context::default()
            .add_message(ContextMessage::user(prompt, compact.model.clone().into()));

        // Set max_tokens for summary
        if let Some(max_token) = compact.max_tokens {
            context = context.max_tokens(max_token);
        }

        // Get summary from the provider
        let response = self.provider.chat(&compact.model, context).await?;

        self.collect_completion_stream_content(compact, response)
            .await
    }

    /// Collects the content from a streaming ChatCompletionMessage response
    /// and extracts text within the configured tag if present
    async fn collect_completion_stream_content<F>(
        &self,
        compact: &Compact,
        mut stream: F,
    ) -> Result<String>
    where
        F: futures::Stream<Item = Result<ChatCompletionMessage>> + Unpin,
    {
        let mut result_content = String::new();

        while let Some(message_result) = stream.next().await {
            let message = message_result?;
            if let Some(content) = message.content {
                result_content.push_str(content.as_str());
            }
        }

        // Extract content from within configured tags if present and if tag is
        // configured
        if let Some(extracted) = extract_tag_content(
            &result_content,
            compact
                .summary_tag
                .as_ref()
                .cloned()
                .unwrap_or_default()
                .as_str(),
        ) {
            return Ok(extracted.to_string());
        }

        // If no tag extraction performed, return the original content
        Ok(result_content)
    }
}

/// Finds a sequence in the context for compaction, starting from the first
/// assistant message and including all messages up to the last possible message
/// (respecting preservation window)
fn find_sequence(context: &Context, preserve_last_n: usize) -> Option<(usize, usize)> {
    let messages = &context.messages;
    if messages.is_empty() {
        return None;
    }

    // len will be always > 0
    let length = messages.len();

    // Find the first assistant message index
    let start = messages
        .iter()
        .enumerate()
        .find(|(_, message)| !message.has_role(Role::System))
        .map(|(index, _)| index)?;

    // Don't compact if there's no assistant message
    if start >= length {
        return None;
    }

    // Calculate the end index based on preservation window
    // If we need to preserve all or more messages than we have, there's nothing to
    // compact
    if preserve_last_n >= length {
        return None;
    }

    // Use saturating subtraction to prevent potential overflow
    let end = length.saturating_sub(preserve_last_n).saturating_sub(1);

    // Ensure we have at least two messages to create a meaningful summary
    // If start > end or end is invalid, don't compact
    if start > end || end >= length || end.saturating_sub(start) < 1 {
        return None;
    }

    // Don't break between a tool call and its result
    if messages.get(end).is_some_and(|msg| msg.has_tool_call()) {
        // If the last message has a tool call, adjust end to include the tool result
        // This means either not compacting at all, or reducing the end by 1
        if end == start {
            // If start == end and it has a tool call, don't compact
            return None;
        } else {
            // Otherwise reduce end by 1
            return Some((start, end.saturating_sub(1)));
        }
    }

    // Return the sequence only if it has at least one message
    if end >= start {
        Some((start, end))
    } else {
        None
    }
}

#[async_trait::async_trait]
impl<T: TemplateService, P: ProviderService> CompactionService for ForgeCompactionService<T, P> {
    async fn compact_context(&self, agent: &Agent, context: Context) -> anyhow::Result<Context> {
        // Call the compact_context method without passing prompt_tokens
        // since the decision logic has been moved to the orchestrator
        self.compact_context(agent, context).await
    }
}

#[cfg(test)]
mod tests {
    use forge_domain::{ModelId, ToolCallFull, ToolCallId, ToolName, ToolResult};
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    fn seq(pattern: impl ToString, preserve_last_n: usize) -> String {
        let model_id = ModelId::new("gpt-4");
        let pattern = pattern.to_string();

        let tool_call = ToolCallFull {
            name: ToolName::new("forge_tool_fs_read"),
            call_id: Some(ToolCallId::new("call_123")),
            arguments: json!({"path": "/test/path"}),
        };

        let tool_result = ToolResult::new(ToolName::new("forge_tool_fs_read"))
            .call_id(ToolCallId::new("call_123"))
            .success(json!({"content": "File content"}).to_string());

        let mut context = Context::default();

        for c in pattern.chars() {
            match c {
                's' => context = context.add_message(ContextMessage::system("System message")),
                'u' => {
                    context = context.add_message(ContextMessage::user(
                        "User message",
                        model_id.clone().into(),
                    ))
                }
                'a' => {
                    context =
                        context.add_message(ContextMessage::assistant("Assistant message", None))
                }
                't' => {
                    context = context.add_message(ContextMessage::assistant(
                        "Assistant message with tool call",
                        Some(vec![tool_call.clone()]),
                    ))
                }
                'r' => {
                    context = context.add_message(ContextMessage::tool_result(tool_result.clone()))
                }
                _ => panic!("Invalid character in test pattern: {c}"),
            }
        }

        let sequence = find_sequence(&context, preserve_last_n);

        let mut result = pattern.clone();
        if let Some((start, end)) = sequence {
            result.insert(start, '[');
            result.insert(end + 2, ']');
        }

        result
    }

    #[test]
    fn test_sequence_finding() {
        // Basic compaction scenarios
        let actual = seq("suaaau", 0);
        let expected = "s[uaaau]";
        assert_eq!(actual, expected);

        let actual = seq("sua", 0);
        let expected = "s[ua]";
        assert_eq!(actual, expected);

        let actual = seq("suauaa", 0);
        let expected = "s[uauaa]";
        assert_eq!(actual, expected);

        // Tool call scenarios
        let actual = seq("suttu", 0);
        let expected = "s[uttu]";
        assert_eq!(actual, expected);

        let actual = seq("sutraau", 0);
        let expected = "s[utraau]";
        assert_eq!(actual, expected);

        let actual = seq("utrutru", 0);
        let expected = "[utrutru]";
        assert_eq!(actual, expected);

        let actual = seq("uttarru", 0);
        let expected = "[uttarru]";
        assert_eq!(actual, expected);

        let actual = seq("urru", 0);
        let expected = "[urru]";
        assert_eq!(actual, expected);

        let actual = seq("uturu", 0);
        let expected = "[uturu]";
        assert_eq!(actual, expected);

        // Preservation window scenarios
        let actual = seq("suaaaauaa", 0);
        let expected = "s[uaaaauaa]";
        assert_eq!(actual, expected);

        let actual = seq("suaaaauaa", 3);
        let expected = "s[uaaaa]uaa";
        assert_eq!(actual, expected);

        let actual = seq("suaaaauaa", 5);
        let expected = "s[uaa]aauaa";
        assert_eq!(actual, expected);

        let actual = seq("suaaaauaa", 8);
        let expected = "suaaaauaa";
        assert_eq!(actual, expected);

        let actual = seq("suauaaa", 0);
        let expected = "s[uauaaa]";
        assert_eq!(actual, expected);

        let actual = seq("suauaaa", 2);
        let expected = "s[uaua]aa";
        assert_eq!(actual, expected);

        let actual = seq("suauaaa", 1);
        let expected = "s[uauaa]a";
        assert_eq!(actual, expected);

        // Tool call atomicity preservation
        let actual = seq("sutrtrtra", 0);
        let expected = "s[utrtrtra]";
        assert_eq!(actual, expected);

        let actual = seq("sutrtrtra", 1);
        let expected = "s[utrtrtr]a";
        assert_eq!(actual, expected);

        let actual = seq("sutrtrtra", 2);
        let expected = "s[utrtr]tra";
        assert_eq!(actual, expected);

        // Conversation patterns
        let actual = seq("suauauaua", 0);
        let expected = "s[uauauaua]";
        assert_eq!(actual, expected);

        let actual = seq("suauauaua", 2);
        let expected = "s[uauaua]ua";
        assert_eq!(actual, expected);

        let actual = seq("suauauaua", 6);
        let expected = "s[ua]uauaua";
        assert_eq!(actual, expected);

        let actual = seq("sutruaua", 0);
        let expected = "s[utruaua]";
        assert_eq!(actual, expected);

        let actual = seq("sutruaua", 3);
        let expected = "s[utru]aua";
        assert_eq!(actual, expected);

        // Special cases
        let actual = seq("saua", 0);
        let expected = "s[aua]";
        assert_eq!(actual, expected);

        let actual = seq("suaut", 0);
        let expected = "s[uau]t";
        assert_eq!(actual, expected);

        // Edge cases
        let actual = seq("", 0);
        let expected = "";
        assert_eq!(actual, expected);

        let actual = seq("s", 0);
        let expected = "s";
        assert_eq!(actual, expected);

        let actual = seq("sua", 3);
        let expected = "sua";
        assert_eq!(actual, expected);

        let actual = seq("ut", 0);
        let expected = "[u]t";
        assert_eq!(actual, expected);

        let actual = seq("suuu", 0);
        let expected = "s[uuu]";
        assert_eq!(actual, expected);

        let actual = seq("ut", 1);
        let expected = "ut";
        assert_eq!(actual, expected);

        let actual = seq("ua", 0);
        let expected = "[ua]";
        assert_eq!(actual, expected);
    }
}
