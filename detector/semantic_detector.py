import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


class SemanticDetector:

    def __init__(self):

        print("Loading semantic detector...")

        # -----------------------------
        # Paths
        # -----------------------------

        self.base_model_name = "Qwen/Qwen2-0.5B-Instruct"

        self.adapter_path = "models/qwen-prompt-detector"

        # -----------------------------
        # Device
        # -----------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # -----------------------------
        # Tokenizer
        # -----------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name
        )

        # -----------------------------
        # Load base model
        # -----------------------------

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name
        )

        # -----------------------------
        # Load LoRA adapter
        # -----------------------------

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path
        )

        # -----------------------------
        # Move to device
        # -----------------------------

        self.model = self.model.to(self.device)

        self.model.eval()

        # -----------------------------
        # Labels
        # -----------------------------

        self.labels = [
            "SAFE",
            "SUSPICIOUS",
            "MALICIOUS"
        ]

    def detect(self, text):

        prompt = f"""
You are a cybersecurity prompt injection classifier.

Classify the INPUT as exactly one of:
SAFE
SUSPICIOUS
MALICIOUS

INPUT:
{text}

ANSWER:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

        logits = outputs.logits[0, -1]

        # -----------------------------
        # Score labels
        # -----------------------------

        scores = {}

        for label in self.labels:

            token_id = self.tokenizer.encode(
                label,
                add_special_tokens=False
            )[0]

            scores[label] = logits[token_id].item()

        # -----------------------------
        # Best prediction
        # -----------------------------

        prediction = max(
            scores,
            key=scores.get
        )

        probs = torch.softmax(
            torch.tensor(list(scores.values())),
            dim=0
        )

        confidence = probs.max().item()

        # -----------------------------
        # Risk mapping
        # -----------------------------

        if prediction == "MALICIOUS":

            score = 0.95
            suspicious = True

        elif prediction == "SUSPICIOUS":

            score = 0.7
            suspicious = True

        else:

            score = 0.1
            suspicious = False

        return {

            "semantic_score": score,

            "is_suspicious": suspicious,

            "classification": prediction,

            "confidence": round(confidence, 3),

            "raw_scores": scores
        }