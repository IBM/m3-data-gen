mistral_config = {
        "gb": {
        "model_name": "mistralai/mistral-large-instruct-2411",
        "end_point": "https://restricted-mixtral-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mistral-large-instruct-2411/",
        "params": {
            "max_tokens": 16384,
            "temperature": 0
        },
        "tokenizer_id_hf": "mistralai/mistral-large-instruct-2411"
        },
        "local": {
        "model_name": "mistralai/mixtral-8x22B-instruct-v0.1",
        "end_point": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01",
        "params": {
            "max_tokens": 16384,
            "temperature": 0
        },    
	}
}
llama_config = {}