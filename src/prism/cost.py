"""What an LLM call costs, from the REAL token counts (never estimated: prototype mistake #8)."""


def model_prices(settings):
    """The price line for the model being used. No price line → error (never a silent $0)."""
    if settings.llm_model not in settings.llm_prices:
        raise ValueError(f"No prices for {settings.llm_model} in llm_prices (configs/default.yaml)")
    return settings.llm_prices[settings.llm_model]


def llm_call_cost(
    prompt_tokens: int, cached_tokens: int, completion_tokens: int, settings
) -> float:
    """Cost of one LLM call in dollars.

    prompt_tokens includes the cached ones, so: normal input = prompt − cached (billed at full price),
    cached input = billed at the cheaper cached price, output = billed at the output price.
    (For reasoning models like gpt-5, the "thinking" is included in completion_tokens.)
    """
    prices = model_prices(settings)
    normal_input = prompt_tokens - cached_tokens
    cost = 0.0
    cost += normal_input * prices.input / 1_000_000
    cost += cached_tokens * prices.cached_input / 1_000_000
    cost += completion_tokens * prices.output / 1_000_000
    return cost
