import os
import torch
import logging
import anthropic
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(self, provider: str, model_name: str, **kwargs):
        """Initialize LLM client based on provider type"""
        self.provider = provider
        self.model_name = model_name

        try:
            if provider == "anthropic":
                self.client = anthropic.Anthropic(
                    api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("Initialized Anthropic client")

            elif provider == "openai":
                self.client = OpenAI(
                    api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
                )
                logger.info("Initialized OpenAI client")

            elif provider == "huggingface":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {self.device}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                    if self.device == "cuda"
                    else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logger.info(f"Loaded HuggingFace model: {model_name}")

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"Error initializing {provider} client: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: int = 5000, temperature: float = 0.0):
        """Generate text using the specified LLM provider"""
        logger.debug(
            f"Generating with {self.provider}, max_tokens={max_tokens}, temperature={temperature}"
        )

        try:
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ],
                )
                return message.content[0].text, message.stop_reason

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (
                    response.choices[0].message.content,
                    response.choices[0].finish_reason,
                )

            elif self.provider == "huggingface":
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response[len(prompt) :].strip(), "stop"

        except Exception as e:
            logger.error(f"Error generating response with {self.provider}: {str(e)}")
            return "", "error"
