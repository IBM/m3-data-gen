import os
from langchain_openai import ChatOpenAI

class ChatRits(ChatOpenAI):
    """RITS chat model integration using langchain-openai."""

    def __init__(self, **config):
        # Set model with model or model_name
        model_name = config.get("model_name", "mistralai/mixtral-8x22B-instruct-v0.1")  # mistralai/mixtral-8x22B-instruct-v0.1
        end_point = config.get("end_point", "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01")
        rits_api_key = os.getenv("RITS_API_KEY", config.get("rits_api_key"))
        timeout = config.get("timeout")
        if rits_api_key is None:
            raise ValueError("rits_api_key is required")
        params = config.get("params", {})
        # Set default values for overriding fields
        rits_config = {}
        rits_config.setdefault("model_name", model_name)
        rits_config.setdefault("api_key", "/")
        rits_config.setdefault("default_headers", {"RITS_API_KEY": os.getenv("RITS_API_KEY", rits_api_key)})
        rits_config.setdefault("base_url", end_point + "/v1")
        rits_config.setdefault("timeout", timeout)
        rits_config.update(params)
        super().__init__(**rits_config)
