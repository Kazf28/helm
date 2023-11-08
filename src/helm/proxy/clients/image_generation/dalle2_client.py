from typing import Any, Dict, List, Optional
import base64

from helm.common.cache import CacheConfig, Cache
from helm.common.general import hlog
from helm.common.file_caches.file_cache import FileCache
from helm.common.media_object import MultimediaObject
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import Request, RequestResult, Sequence, wrap_request_time
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from helm.proxy.clients.moderation_api_client import ModerationAPIClient
from helm.proxy.clients.client import Client, CachingClient
from .image_generation_client_utils import get_single_image_multimedia_object

try:
    import openai
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["openai"])


class DALLE2Client(Client):
    MAX_PROMPT_LENGTH: int = 1000
    VALID_IMAGE_DIMENSIONS: List[int] = [256, 512, 1024]
    DEFAULT_IMAGE_SIZE_STR: str = "512x512"
    CONTENT_POLICY_VIOLATED: str = (
        "The prompt violates OpenAI's content policy. "
        "See https://labs.openai.com/policies/content-policy for more information."
    )

    # DALL-E 2 will respond with the following error messages (or even a substring of the message)
    # if it has any issues generating images for a particular prompt
    PROMPT_FLAGGED_ERROR: str = (
        "Your request was rejected as a result of our safety system. "
        "Your prompt may contain text that is not allowed by our safety system."
    )
    PROMPT_FLAGGED_ERROR2: str = (
        "Something went wrong with your generation. You may try again or ask for a different prompt"
    )
    PROMPT_FLAGGED_ERROR3: str = (
        "The server had an error while processing your request. Sorry about that! You can retry your request, "
        "or contact us through our help center at help.openai.com if the error persists."
    )

    def __init__(
        self,
        api_key: str,
        cache_config: CacheConfig,
        file_cache: FileCache,
        moderation_api_client: ModerationAPIClient,
        org_id: Optional[str] = None,
    ):
        self.file_cache: FileCache = file_cache
        self._cache = Cache(cache_config)

        self.moderation_api_client: ModerationAPIClient = moderation_api_client

        self.org_id: Optional[str] = org_id
        self.api_key: Optional[str] = api_key
        self.api_base: str = "https://api.openai.com/v1"

    def make_request(self, request: Request) -> RequestResult:
        def get_content_policy_violated_result():
            no_image = Sequence(
                text="",
                logprob=0,
                tokens=[],
                multimodal_content=MultimediaObject(),
                finish_reason={"reason": self.CONTENT_POLICY_VIOLATED},
            )
            return RequestResult(
                success=True,
                cached=False,
                request_time=0,
                completions=[no_image] * request.num_completions,
                embedding=[],
            )

        def get_size_str(w: Optional[int], h: Optional[int]) -> str:
            if w is None or h is None:
                return self.DEFAULT_IMAGE_SIZE_STR

            assert w == h, "The DALL-E 2 API only supports generating square images."
            assert w in self.VALID_IMAGE_DIMENSIONS, "Valid dimensions are 256x256, 512x512, or 1024x1024 pixels."
            return f"{w}x{h}"

        if len(request.prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError("The maximum length of the prompt is 1000 characters.")
        if request.num_completions < 1 or request.num_completions > 10:
            raise ValueError("`num_completions` must be between 1 and 10.")

        # Use the Moderation API to check if the prompt violates OpenAI's content policy before generating images
        if self.moderation_api_client.will_be_flagged(request.prompt):
            return get_content_policy_violated_result()

        # https://beta.openai.com/docs/api-reference/images/create#images/create-response_format
        assert request.image_generation_parameters is not None
        raw_request: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": request.num_completions,
            "size": get_size_str(
                request.image_generation_parameters.output_image_width,
                request.image_generation_parameters.output_image_height,
            ),
            "response_format": "b64_json",  # Always set to b64_json as URLs are only valid for an hour
        }

        try:

            def do_it():
                openai.organization = self.org_id
                openai.api_key = self.api_key
                openai.api_base = self.api_base
                result = openai.Image.create(**raw_request)
                assert "data" in result, f"Invalid response: {result} from prompt: {request.prompt}"

                for image in result["data"]:
                    # Write out the image to a file and save the path
                    image["file_path"] = self.file_cache.store(lambda: base64.b64decode(image["b64_json"]))
                    image.pop("b64_json", None)
                return result

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self._cache.get(cache_key, wrap_request_time(do_it))
        except openai.error.OpenAIError as e:
            if (
                str(e) in self.PROMPT_FLAGGED_ERROR
                or self.PROMPT_FLAGGED_ERROR2 in str(e)
                or self.PROMPT_FLAGGED_ERROR3 in str(e)
            ):
                # Some requests fail even if we check the prompt against the moderation API.
                # For example, "black" in Spanish (negro) causes requests to DALL-E to fail even
                # though the prompt does not get flagged by the Moderation API.
                hlog(f"Failed safety check: {request.prompt}")
                return get_content_policy_violated_result()
            else:
                error: str = f"DALL-E 2 error: {e}"
                return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[Sequence] = [
            Sequence(
                text="",
                logprob=0,
                tokens=[],
                multimodal_content=get_single_image_multimedia_object(generated_image["file_path"]),
            )
            for generated_image in response["data"]
        ]
        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        raise NotImplementedError("This client does not support tokenizing.")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        raise NotImplementedError("This client does not support decoding.")
