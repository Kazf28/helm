from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Any, Dict, List, Optional, TypedDict

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog, hwarn
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
)
from helm.proxy.retry import NonRetriableException
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_sequence
from helm.tokenizers.huggingface_tokenizer import HuggingFaceTokenizer, WrappedPreTrainedTokenizer
from threading import Lock


class StopAtSpecificTokenCriteria(StoppingCriteria):
    def __init__(self, stop_sequence: List[int]):
        super().__init__()
        self.stop_sequence = stop_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Create a tensor from the stop_sequence
        stop_sequence_tensor = torch.tensor(self.stop_sequence, device=input_ids.device, dtype=input_ids.dtype)

        # Check if the current sequence ends with the stop_sequence
        current_sequence = input_ids[:, -len(self.stop_sequence) :]
        return bool(torch.all(current_sequence == stop_sequence_tensor).item())


class HuggingFaceRequest(TypedDict):
    """Data passed between make_request and serve_request. Used as the cache key."""

    engine: str
    prompt: str
    temperature: float
    num_return_sequences: int
    max_new_tokens: int
    top_p: float
    echo_prompt: bool
    top_k_per_token: int
    stop_sequences: List


class HuggingFaceServer:
    """A thin wrapper around a Hugging Face AutoModelForCausalLM for HuggingFaceClient to call."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        **kwargs,
    ):
        self.device: Optional[str]
        if "device_map" in kwargs:
            if "device" in kwargs:
                raise ValueError("At most one of one of `device` and `device_map` may be specified.")
            try:
                import accelerate  # noqa: F401
            except ModuleNotFoundError as e:
                handle_module_not_found_error(e, ["accelerate"])
            hlog(f'Hugging Face device_map set to "{kwargs["device_map"]}" from kwargs.')
            self.device = None
        elif "device" in kwargs:
            if "device_map" in kwargs:
                raise ValueError("At most one of one of `device` and `device_map` may be specified.")
            hlog(f'Hugging Face device set to "{kwargs["device"]}" from kwargs.')
            self.device = kwargs.pop("device")
        elif torch.cuda.is_available():
            hlog('Hugging Face device set to "cuda:0" because CUDA is available.')
            self.device = "cuda:0"
        else:
            hlog('Hugging Face device set to "cpu" because CUDA is unavailable.')
            self.device = "cpu"

        # Security issue: currently we trust remote code by default.
        # We retain this temporarily to maintain reverse compatibility.
        # TODO: Delete if-else and don't set trust_remote_code=True
        if "trust_remote_code" not in kwargs:
            kwargs["trust_remote_code"] = True

        with htrack_block(f"Loading Hugging Face model {pretrained_model_name_or_path}"):
            # WARNING this may fail if your GPU does not have enough memory
            if self.device is None:
                # kwargs contains device_map=auto
                # Do not call to() because accelerate will take care of model device placement.
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs).to(
                    self.device
                )
        self.wrapped_tokenizer = wrapped_tokenizer

    def serve_request(self, raw_request: HuggingFaceRequest) -> Dict:
        with self.wrapped_tokenizer as tokenizer:
            encoded_input = tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
                0 if self.device is None else self.device
            )
        stopping_criteria: Optional[StoppingCriteriaList] = None
        optional_args = {}
        if len(raw_request["stop_sequences"]) > 0:
            with self.wrapped_tokenizer as tokenizer:
                stop_sequence_ids = tokenizer(
                    raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
                )
            if len(stop_sequence_ids.input_ids) == 1 and len(stop_sequence_ids.input_ids[0]) == 1:
                optional_args["eos_token_id"] = stop_sequence_ids.input_ids[0][0]
            else:
                stopping_criteria = StoppingCriteriaList()
                for stop_sequence_input_ids in stop_sequence_ids.input_ids:
                    stopping_criteria.append(StopAtSpecificTokenCriteria(stop_sequence=stop_sequence_input_ids))

        # Check if we need to compute the perplexity of the prompt (#1497)
        compute_logprobs_only = (
            raw_request["max_new_tokens"] == 0
            and raw_request["num_return_sequences"] == 1
            and raw_request["echo_prompt"]
        )

        # Use HuggingFace's `generate` method.
        if compute_logprobs_only:
            with torch.no_grad():
                output = self.model(encoded_input["input_ids"])
            sequences = encoded_input["input_ids"]
            scores = output.logits
        else:
            output = self.model.generate(
                **encoded_input,
                temperature=raw_request["temperature"],
                num_return_sequences=raw_request["num_return_sequences"],
                max_new_tokens=raw_request["max_new_tokens"],
                top_p=raw_request["top_p"],
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                **optional_args,
                stopping_criteria=stopping_criteria,
            )
            sequences = output.sequences
            scores = output.scores

        prompt_tokens_logprobs = []
        if compute_logprobs_only:
            # Append the logprob of the first token of the prompt.
            prompt_tokens_logprobs.append(0.0)

            # Compute logprobs of prompt tokens.
            for completion_id in range(raw_request["num_return_sequences"]):
                for i in range(len(sequences[completion_id]) - 1):
                    logprobs = torch.nn.functional.log_softmax(scores[completion_id][i], dim=0)
                    prompt_tokens_logprobs.append(logprobs[sequences[completion_id][i + 1]].item())

        # Compute logprobs of generated tokens for each completed sequence.
        all_generated_tokens_logprobs = []
        for completion_id in range(raw_request["num_return_sequences"]):
            generated_tokens_logprobs = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)
                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                generated_tokens_logprobs.append(logprobs[sequences[completion_id][j]].item())
            all_generated_tokens_logprobs.append(generated_tokens_logprobs)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        with self.wrapped_tokenizer as tokenizer:
            all_tokens = [[tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, generated_tokens_logprobs in zip(
            all_decoded_text, all_tokens, all_generated_tokens_logprobs
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": generated_tokens_logprobs,
                    "prompt_logprobs": prompt_tokens_logprobs,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


class HuggingFaceServerFactory:
    """A factory that creates and caches HuggingFaceServer objects."""

    _servers: Dict[str, HuggingFaceServer] = {}
    _servers_lock: Lock = Lock()

    @staticmethod
    def get_server(
        helm_model_name: str,
        pretrained_model_name_or_path: str,
        wrapped_tokenizer: WrappedPreTrainedTokenizer,
        **kwargs,
    ) -> Any:
        """
        Checks if the desired HuggingFaceModel is cached. Creates the HuggingFaceModel if it's not cached.
        Returns the HuggingFaceModel.
        """
        with HuggingFaceServerFactory._servers_lock:
            if helm_model_name not in HuggingFaceServerFactory._servers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM model {helm_model_name} with Hugging Face Transformers"
                ):
                    HuggingFaceServerFactory._servers[helm_model_name] = HuggingFaceServer(
                        pretrained_model_name_or_path, wrapped_tokenizer, **kwargs
                    )

        return HuggingFaceServerFactory._servers[helm_model_name]


TORCH_DTYPE_KEY = "torch_dtype"
TORCH_DTYPE_VALUE_PREFIX = "torch."


def _process_huggingface_client_kwargs(raw_kwargs: Dict[str, Any]):
    """Process the kwargs for HuggingFaceClient.

    The kwargs passed to HuggingFaceClient will eventually be passed to AutoModel.from_pretrained().
    Since the kwargs from HuggingFaceClient may be derived from configuration YAML,
    they may contain primitive types instead of the unserializable types that
    AutoModel.from_pretrained() expects (e.g. torch_dtype). This function converts values of
    primitive types to values of the unserializable types."""
    processed_kwargs = deepcopy(raw_kwargs)

    # Convert torch_dtype string value to actual dtypes
    # e.g. the string "torch.bfloat16" is converted to torch.bfloat16
    torch_dtype = processed_kwargs.get(TORCH_DTYPE_KEY)
    if torch_dtype and isinstance(torch_dtype, str):
        if torch_dtype.startswith(TORCH_DTYPE_VALUE_PREFIX):
            processed_kwargs[TORCH_DTYPE_KEY] = getattr(torch, torch_dtype[len(TORCH_DTYPE_VALUE_PREFIX) :])

    return processed_kwargs


class HuggingFaceClient(CachingClient):
    def __init__(
        self,
        cache_config: CacheConfig,
        tokenizer: Tokenizer,
        pretrained_model_name_or_path: Optional[str] = None,
        end_of_text_token: Optional[str] = None,
        apply_chat_template: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(cache_config=cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        if not isinstance(tokenizer, HuggingFaceTokenizer):
            raise ValueError(
                f"Tokenizer for Hugging Face model {pretrained_model_name_or_path} must be a HuggingFaceTokenizer, "
                "but instead it is {tokenizer}"
            )
        self._wrapped_tokenizer: WrappedPreTrainedTokenizer = tokenizer.get_wrapped_tokenizer()
        self._kwargs = _process_huggingface_client_kwargs(kwargs)
        self._end_of_text_token = end_of_text_token
        # If the user did not explicitly configure whether the model is a chat model with `apply_chat_template` arg,
        # auto-infer if the model is a chat model based on whether the tokenizer has a chat template.
        # Note: Auto-inference is incorrect for some non-chat models that still have chat templates
        # e.g. Qwen2, Qwen 2.5.
        # For these models, the `apply_chat_template` arg should be explicitly set to false.
        if apply_chat_template is not None:
            self._apply_chat_template = apply_chat_template
        else:
            with self._wrapped_tokenizer as hf_tokenizer:
                self._apply_chat_template = bool(hf_tokenizer.chat_template)
                hwarn(
                    f"Automatically set `apply_chat_template` to {self._apply_chat_template} based on "
                    "whether the tokenizer has a chat template. "
                    "If this is incorrect, please explicitly set `apply_chat_template`."
                )

    def get_prompt(self, request: Request) -> str:
        if request.prompt and request.messages:
            raise NonRetriableException(f"More than one of `prompt` and `messages` was set in request: {request}")
        # Chat model expects a list of messages as input
        if self._apply_chat_template:
            with self._wrapped_tokenizer as tokenizer:
                if request.messages:
                    prompt = tokenizer.apply_chat_template(request.messages, tokenize=False)
                    assert isinstance(prompt, str)
                    return prompt
                else:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": request.prompt}], tokenize=False
                    )
                    assert isinstance(prompt, str)
                    return prompt
        # Base non-chat model expects a string as input
        else:
            if request.messages:
                raise NonRetriableException("Chat mesages not supported by non-chat model")
            else:
                return request.prompt

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request: HuggingFaceRequest = {
            "engine": request.model_engine,
            "prompt": self.get_prompt(request),
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        pretrained_model_name_or_path = (
            self._pretrained_model_name_or_path if self._pretrained_model_name_or_path else request.model
        )
        huggingface_model: HuggingFaceServer = HuggingFaceServerFactory.get_server(
            helm_model_name=request.model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            wrapped_tokenizer=self._wrapped_tokenizer,
            **self._kwargs,
        )

        try:

            def do_it() -> Dict[str, Any]:
                return huggingface_model.serve_request(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            error: str = f"HuggingFace error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                if raw_completion.get("prompt_logprobs"):
                    for token_text, logprob in zip(
                        raw_completion["tokens"][: response["input_length"]],
                        raw_completion["prompt_logprobs"][: response["input_length"]],
                    ):
                        tokens.append(Token(text=token_text, logprob=logprob))
                        sequence_logprob += logprob
                else:
                    for token_text in raw_completion["tokens"][: response["input_length"]]:
                        tokens.append(Token(text=token_text, logprob=0.0))

            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob in zip(generated_tokens, raw_completion["logprobs"]):
                tokens.append(Token(text=token_text, logprob=logprob))
                sequence_logprob += logprob

            completion = GeneratedOutput(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request, end_of_text_token=self._end_of_text_token)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )
