//! OpenAI-compatible HTTP surface (PH-501, band B5). B0 pins the route constants
//! so the server, client and UI agree on paths before the axum handler exists.

pub const PATH_MODELS: &str = "/v1/models";
pub const PATH_CHAT_COMPLETIONS: &str = "/v1/chat/completions";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_are_openai_shaped() {
        assert_eq!(PATH_MODELS, "/v1/models");
        assert_eq!(PATH_CHAT_COMPLETIONS, "/v1/chat/completions");
    }
}
