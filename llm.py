from langchain_ibm import ChatWatsonx
from langchain_openai import ChatOpenAI
from typing import List, Dict, Tuple,Optional
import copy, os

class ChatRits(ChatOpenAI):
    """RITS chat model integration using langchain-openai."""

    def __init__(self, config):
        # Set model with model or model_name
        model_name = config.get("model_name", "mistralai/mixtral-8x22B-instruct-v0.1")
        end_point = config.get("end_point", "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01")
        rits_api_key = os.getenv("RITS_API_KEY")
        if rits_api_key is None:
            raise ValueError("rits_api_key is required")
        params = config.get("params", {})
        # Set default values for overriding fields
        rits_config = {}
        rits_config.setdefault("model_name", model_name)
        rits_config.setdefault("api_key", "/")
        if rits_api_key is not None:
            rits_config.setdefault("default_headers", {"RITS_API_KEY": rits_api_key})
        else:
            print("RITS KEY IN GB NOT OKAY?",os.getenv("RITS_API_KEY") is None or os.getenv("RITS_API_KEY")=="")
            rits_config.setdefault("default_headers", {"RITS_API_KEY": os.getenv("RITS_API_KEY")})
        rits_config.setdefault("base_url", end_point + "/v1")
        rits_config.update(params)
        super().__init__(**rits_config)


# Temporary wrapper for RITS model as langchain output - str basically.
class Langchain_RITS:
    def __init__(self, llm: ChatRits):
        self.llm = llm

    def invoke(self, prompt: str) -> str:
        res = self.llm.invoke(prompt).content
        return res

    def generate(self, prompt: str) -> List[str]:
        res = self.llm.generate([prompt])
        return [g.text.strip() for g in res.generations[0]]


def load_model(provider: str, config: Dict):
    if provider == "watson_x":
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
        model_params = {
            GenTextParamsMetaNames.MAX_NEW_TOKENS: config["params"][

                "max_token"
            ],
            GenTextParamsMetaNames.MIN_NEW_TOKENS: config["params"][
                "min_new_tokens"
            ],
            GenTextParamsMetaNames.DECODING_METHOD: config["params"][
                "decoding_method"
            ],
            GenTextParamsMetaNames.REPETITION_PENALTY: config["params"][
                "repetition_penalty"
            ],
        }
        _watsonx_valid_model_param_keys = []
        for prop in GenTextParamsMetaNames._meta_props_definitions:
            _watsonx_valid_model_param_keys.append(prop.key)
        _watsonx_valid_model_param_keys = set(_watsonx_valid_model_param_keys)

        watsonx_model_params = {}
        for param_name, param_value in model_params.items():
            if param_name in _watsonx_valid_model_param_keys:
                watsonx_model_params[param_name] = param_value
            else:
                raise Exception(
                    f"Model parameter {param_name} is not a valid parameter for the IBM WatsonX LLM API. Ignoring."
                )

        llm = ChatWatsonx(
            model_id=config["model_id_wx"].lower(),
            url=config["url"],
            apikey=config["apikey"],
            params=watsonx_model_params,
            streaming=False,
            project_id=config["project_id"],
        )
        langchain_llm = llm.watsonx_model.to_langchain()
        return langchain_llm

    elif provider == "rits":
        llm = ChatRits(config)
        return Langchain_RITS(llm)