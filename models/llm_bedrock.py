from __future__ import annotations
from typing import Optional


class BedrockGenerator:
    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None, profile: Optional[str] = None) -> None:
        import boto3  # type: ignore  # lazy import to keep optional
        self.model_id = model_id or "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.region = region
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
            self.client = session.client("bedrock-runtime")
        else:
            self.client = boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
        # Use Converse API for normalized interaction
        resp = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": max_new_tokens,
                "temperature": temperature,
            },
        )
        # Extract first text output
        msg = resp.get("output", {}).get("message", {})
        contents = msg.get("content", [])
        if contents and "text" in contents[0]:
            return contents[0]["text"].strip()
        # Fallback for provider variations
        for c in contents:
            if isinstance(c, dict) and c.get("text"):
                return c["text"].strip()
        return ""


