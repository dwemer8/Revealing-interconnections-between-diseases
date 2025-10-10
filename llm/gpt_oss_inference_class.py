import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort
)
from enum import Enum
from typing import Iterable, Optional, List

class GPT_OSS_MODEL_SIZE(Enum):
    SMALL = 20
    BIG = 120

class GPT_OSS_Inference:
    def __init__(
        self,
        model_size: GPT_OSS_MODEL_SIZE = GPT_OSS_MODEL_SIZE.SMALL,
        generate_args: dict = {},
        reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
        conversation_start_date: str = "2025-06-28",
        developer_message: str = "",
        device_map: str = "auto" #"cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.set_system_message(reasoning_effort=reasoning_effort, conversation_start_date=conversation_start_date)
        self.set_developer_message(message=developer_message)
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        self.generate_args = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        self.generate_args.update(generate_args)

        MODEL_ID = f"openai/gpt-oss-{model_size.value}b"
        # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )

    def set_system_message(self, reasoning_effort: ReasoningEffort, conversation_start_date: str):
        self.system_message = (
            SystemContent.new()
                .with_reasoning_effort(reasoning_effort)
                .with_conversation_start_date(conversation_start_date)
        )

    def set_developer_message(self, message: str = ""):
        self.developer_message = (
            DeveloperContent.new()
            .with_instructions(message)
        )

    def __call__(self, query: str) -> List[Message]:
        convo = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, self.system_message),
                Message.from_role_and_content(Role.DEVELOPER, self.developer_message),
                Message.from_role_and_content(Role.USER, query),
            ]
        )
        
        tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        # 3) 'tokens' comes from: encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
        #    It should be a list[int] (or similar) of input token ids for the *assistant turn*.
        if isinstance(tokens, list):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.model.device)
        elif hasattr(tokens, "ids") or hasattr(tokens, "input_ids"):
            # Support common container shapes from encoders
            raw = getattr(tokens, "ids", None) or getattr(tokens, "input_ids", None)
            input_ids = torch.tensor([raw], dtype=torch.long, device=self.model.device)
        else:
            raise TypeError("Unexpected 'tokens' format. Expected list[int] or an object with .input_ids/.ids")
        
        gen_out = self.model.generate(
            input_ids=input_ids,
            eos_token_id=self.model.config.eos_token_id,  # model’s own EOS
            pad_token_id=self.model.config.eos_token_id,
            **self.generate_args,
        )

        new_tokens_tensor = gen_out[0, input_ids.shape[1]:]
        new_tokens = new_tokens_tensor.tolist() 

        # 6) Harmony often defines its own special stop token(s).
        #    If your encoding exposes a stop id, strip it before parsing.
        stop_ids = []
        for attr in ("stop_token_id", "stop_id", "special_stop_id", "stop"):
            if hasattr(self.encoding, attr):
                sid = getattr(self.encoding, attr)
                if isinstance(sid, int):
                    stop_ids = [sid]
                    break
                if isinstance(sid, (list, tuple)) and all(isinstance(x, int) for x in sid):
                    stop_ids = list(sid)
                    break

        # Remove trailing Harmony stop if present
        def strip_trailing_stops(seq, stops):
            if not stops or not seq:
                return seq
            while seq and seq[-1] in stops:
                seq.pop()
            return seq

        new_tokens = strip_trailing_stops(new_tokens, stop_ids)

        # After receiving a token response
        # Do not pass in the stop token
        parsed_response = self.encoding.parse_messages_from_completion_tokens(new_tokens, Role.ASSISTANT)
        # rendered_response = self.render_messages(parsed_response)
        return parsed_response
    
    @staticmethod
    def extract_text(msg) -> str:
        # Extract text pieces from content list
        text_bits = []
        for part in getattr(msg, "content", []) or []:
            # Many content types expose `.text`; fall back to str(part)
            text = getattr(part, "text", None)
            if text is None:
                text = str(part)
            text_bits.append(text)

        text_joined = "\n".join(text_bits).strip()
        return text_joined

    @staticmethod
    def render_messages(msgs: List[Message]):
        output = []
        for i, m in enumerate(msgs, 1):
            output.append({
                "index": i,
                "role": getattr(getattr(m, "author", None), "role", None),
                "channel": getattr(m, "channel", None),
                "recipient": getattr(m, "recipient", None),
                "text": GPT_OSS_Inference.extract_text(m),
                
            })
        return output
    
    @staticmethod
    def get_final_answer(msgs: List[Message]) -> Optional[str]:
        for m in msgs[::-1]:
            if getattr(m, "channel", None) == "final":
                return GPT_OSS_Inference.extract_text(m)
        return None