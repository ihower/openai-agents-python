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
    """Test that thought signatures are preserved from Gemini responses to messages via reasoning items."""
    import json

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

    # Verify thought_signature is stored in items
    func_call_dict = func_calls[0].model_dump()
    assert func_call_dict["thought_signature"] == "test_signature_abc"

    # Step 2: For the signature to be restored in messages, we need a reasoning item with functionCall
    # This simulates the full roundtrip through Gemini's reasoning format
    items_with_reasoning = [
        {"role": "user", "content": "test"},
        {
            "type": "reasoning",
            "id": "fake_id",
            "summary": [],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": json.dumps({"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}),
                }
            ],
            "encrypted_content": "test_signature_abc",
        },
    ] + [item.model_dump() for item in items if hasattr(item, "type") and item.type == "function_call"]

    messages = Converter.items_to_messages(items_with_reasoning, preserve_thinking_blocks=True)  # type: ignore[arg-type]

    # Verify thought_signature is restored in extra_content format
    assistant_msg = [msg for msg in messages if msg.get("role") == "assistant"][0]
    tool_call = assistant_msg["tool_calls"][0]
    assert tool_call["extra_content"]["google"]["thought_signature"] == "test_signature_abc"


def test_gemini_function_call_reasoning_item_with_signature():
    """Test that Gemini functionCall reasoning items with signatures are properly converted."""
    import json

    # Simulate items from Gemini response with functionCall reasoning item
    items = [
        {"role": "user", "content": "What's the weather?"},
        # Gemini reasoning item with functionCall in the text
        {
            "type": "reasoning",
            "id": "fake_id",
            "summary": [],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": json.dumps({"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}),
                }
            ],
            "encrypted_content": "signature_xyz_123",
        },
        # The actual function call
        {
            "type": "function_call",
            "id": "fake_id",
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": '{"city": "Tokyo"}',
        },
    ]

    # Convert to messages with preserve_thinking_blocks=True
    messages = Converter.items_to_messages(items, preserve_thinking_blocks=True)  # type: ignore[arg-type]

    # Find the assistant message with tool calls
    assistant_msg = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("tool_calls")][0]

    # Verify the signature from reasoning item is applied to the function call
    tool_call = assistant_msg["tool_calls"][0]
    assert "extra_content" in tool_call
    assert tool_call["extra_content"]["google"]["thought_signature"] == "signature_xyz_123"


def test_gemini_text_reasoning_item_without_function_call():
    """Test that Gemini text reasoning items (without functionCall) use original logic."""
    import json

    # Simulate items with text reasoning item (not functionCall)
    items = [
        {"role": "user", "content": "Think about this"},
        # Gemini reasoning item with text only
        {
            "type": "reasoning",
            "id": "fake_id",
            "summary": [],
            "content": [
                {
                    "type": "reasoning_text",
                    "text": json.dumps({"text": "I'm thinking about the weather..."}),
                }
            ],
            "encrypted_content": "signature_text_456",
        },
        # Some message content
        {
            "type": "message",
            "id": "fake_id",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Let me help you."}],
        },
    ]

    # Convert to messages with preserve_thinking_blocks=True
    messages = Converter.items_to_messages(items, preserve_thinking_blocks=True)  # type: ignore[arg-type]

    # Should have user message and assistant message
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Let me help you."
