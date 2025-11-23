"""
Test for Gemini thought signatures in function calling.

Validates that thought signatures are preserved through the roundtrip:
- Gemini response → items → messages
"""

from __future__ import annotations

from openai.types.chat.chat_completion_message_tool_call import Function

from agents.extensions.models.litellm_model import InternalChatCompletionMessage, InternalToolCall
from agents.models.chatcmpl_converter import Converter


def test_thought_signature_roundtrip():
    """Test that thought signatures are preserved from Gemini responses to messages."""

    # Create mock Gemini response with thought signature in new extra_content structure
    class MockToolCall(InternalToolCall):
        def __init__(self):
            super().__init__(
                id="call_123",
                type="function",
                function=Function(name="get_weather", arguments='{"city": "Paris"}'),
                extra_content={"google": {"thought_signature": "test_signature_abc"}},
            )

    message = InternalChatCompletionMessage(
        role="assistant",
        content="I'll check the weather.",
        reasoning_content="",
        tool_calls=[MockToolCall()],
    )

    # Step 1: Convert to items
    items = Converter.message_to_output_items(message)
    func_calls = [item for item in items if hasattr(item, "type") and item.type == "function_call"]
    assert len(func_calls) == 1

    # Verify thought_signature is stored in items with new structure
    func_call_dict = func_calls[0].model_dump()
    assert (
        func_call_dict["provider_specific_fields"]["google"]["thought_signature"]
        == "test_signature_abc"
    )

    # Step 2: Convert back to messages
    items_as_dicts = [item.model_dump() for item in items]
    messages = Converter.items_to_messages([{"role": "user", "content": "test"}] + items_as_dicts)

    # Verify thought_signature is restored in extra_content format
    assistant_msg = [msg for msg in messages if msg.get("role") == "assistant"][0]
    tool_call = assistant_msg["tool_calls"][0]  # type: ignore[index, typeddict-item]
    assert tool_call["extra_content"]["google"]["thought_signature"] == "test_signature_abc"
