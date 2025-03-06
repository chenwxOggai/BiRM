from openai import OpenAI


def send_questions_to_vllm(
    questions,
    temperature,
    max_tokens,
    stop_tokens,
    n_response,
    vllm_url,
    model_name_or_path,
):
    """
    Send questions to vLLM API in batch and retrieve answers
    """
    openai_api_key = "EMPTY"
    openai_api_base = vllm_url

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    completion = client.completions.create(
        model=model_name_or_path,
        prompt=questions,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_tokens,
        n=n_response,  # for greedy eval, set seed = 42
    )

    return completion
