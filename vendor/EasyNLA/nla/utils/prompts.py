"""AV prompt construction shared by the RL trainers (and eval/probe scripts)."""


def build_prompt_text(prompt_msgs, inject_char, tokenizer, enable_thinking=False):
    """Apply the chat template; substitute the <INJECT> placeholder with inject_char.

    enable_thinking=False by default: Qwen3-family templates otherwise open a
    bare <think> block at the generation prompt — NOT the format the NLA AVs
    were trained in (the qwen3.6 av_base card explicitly says to apply the
    template with enable_thinking=False). In think-mode the AV confabulates
    and IGNORES the injected vector (an1 smoke 2026-08-14: 6/6 byte-identical
    explanations across 6 different activations, steering mechanically
    verified). Templates without the variable simply ignore the kwarg."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)
