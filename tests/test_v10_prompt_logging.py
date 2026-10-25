import src.generator as generator


class _FakeLlama:
    def tokenize(self, data, add_bos=False, special=False):
        return list(data)

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": "Credit Hour is 1.5."}}], "usage": {}}


def test_exact_messages_are_captured_only_inside_the_block(monkeypatch):
    monkeypatch.setattr(generator, "_get_generator", lambda *a, **k: _FakeLlama())
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    generator._invoke_messages(messages, "model", 16, "outside")
    sink = []
    with generator.capture_generation_prompts(sink):
        answer = generator._invoke_messages(messages, "model", 16, "direct_generation")
    generator._invoke_messages(messages, "model", 16, "after")
    assert answer == "Credit Hour is 1.5."
    assert sink == [{"stage": "direct_generation", "max_tokens": 16, "messages": messages}]
    assert sink[0]["messages"] is not messages
