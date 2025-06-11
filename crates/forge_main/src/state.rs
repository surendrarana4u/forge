use derive_setters::Setters;
use forge_api::{AgentId, ConversationId, ModelId, Provider, Usage, Workflow};

use crate::prompt::ForgePrompt;

//TODO: UIState and ForgePrompt seem like the same thing and can be merged
/// State information for the UI
#[derive(Debug, Default, Clone, Setters)]
#[setters(strip_option)]
pub struct UIState {
    pub conversation_id: Option<ConversationId>,
    pub usage: Usage,
    pub operating_agent: Option<AgentId>,
    pub is_first: bool,
    pub model: Option<ModelId>,
    pub provider: Option<Provider>,
}

impl UIState {
    pub fn new(workflow: Workflow) -> Self {
        let operating_agent = workflow
            .variables
            .get("operating_agent")
            .and_then(|value| value.as_str())
            .map(AgentId::new)
            .or_else(|| workflow.agents.first().map(|agent| agent.id.clone()));

        Self {
            conversation_id: Default::default(),
            usage: Default::default(),
            is_first: true,
            model: workflow.model,
            operating_agent,
            provider: Default::default(),
        }
    }
}

impl From<UIState> for ForgePrompt {
    fn from(state: UIState) -> Self {
        ForgePrompt {
            usage: Some(state.usage),
            model: state.model,
            agent_id: state.operating_agent.unwrap_or(AgentId::new("act")),
        }
    }
}
