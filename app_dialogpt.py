#! /usr/bin/env python

# Import necessary libraries
import gradio as gr # Gradio for UI
import torch # PyTorch
from transformers import AutoModelForCausalLM, AutoTokenizer 

MODEL_ID = "microsoft/DialoGPT-small"  # medium also works
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()

# Important: set pad token consistently to avoid attention-mask ambiguity
tok.pad_token = tok.eos_token
model.config.pad_token_id = model.config.eos_token_id

CTX = getattr(model.config, 'n_positions', 1024)

def build_history_text(history):
    if not history:
        return ""
    pieces = []
    eos = tok.eos_token
    for u, a in history:
        if u:
            pieces.append(f"{u}{eos}")
        if a:
            pieces.append(f"{a}{eos}")
    return "".join(pieces)



with gr.Blocks(title="DialoGPT Chatbot") as demo:
    gr.Markdown("# 💬 Simple DialoGPT Chatbot")
    chat = gr.Chatbot(height=350)
    msg = gr.Textbox(placeholder="Say something…")
    state = gr.State([])

    def clear():
        return [], []

    # Set model to evaluation mode
    @torch.inference_mode()
    def respond(user_msg, history):
        history = history or []
        if not user_msg or not user_msg.strip():
            return history, history

        # Compose flat text with EOS separators
        new_input = build_history_text(history) + user_msg + tok.eos_token

        # Tokenize without extra specials; we control EOS ourselves
        input_ids = tok.encode(new_input, return_tensors="pt", add_special_tokens=False)

        # Keep last N tokens (leave a margin for generation)
        max_ctx = max(CTX - 2, 8)
        input_ids = input_ids[:, -max_ctx:]

        # Since we're not padding, attention is simply all ones
        attention_mask = torch.ones_like(input_ids)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            pad_token_id=tok.eos_token_id,   # be explicit
            eos_token_id=tok.eos_token_id,
        )

        generated = output_ids[0, input_ids.shape[-1]:]
        reply = tok.decode(generated, skip_special_tokens=True).strip()

        # Fallback in rare cases where decode is empty
        if not reply:
            reply = "(no response generated—try again or rephrase)"

        history = history + [[user_msg, reply]]
        return history, history

    msg.submit(respond, [msg, state], [chat, state])
    msg.submit(lambda: "", None, msg)
    gr.Button("Clear").click(clear, None, [chat, state])

if __name__ == "__main__":
    demo.launch()