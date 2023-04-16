import tiktoken

MODEL = "gpt-3.5-turbo"
MAX_TOKENS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0301": 4096,
        "gpt-4": 8192,
        "gpt-4-0314": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0314": 32768,
        "text-davinci-003": 4097,
        "text-davinci-002": 4097,
        "code-davinci-002": 8001,
        "text-curie-001": 2049,
        "text-babbage-001": 2049,
        "text-ada-001": 2049,
        "davinci": 2049,
        "curie": 2049,
        "babbage": 2049,
        "ada": 2049 }

def get_tokens(text: str) -> int:
    enc = tiktoken.encoding_for_model(MODEL)
    tokens = len(enc.encode(text))
    return tokens

# tokens = TOKENS[MODEL]
def toksplit(text: str, tokens: int) -> list:
    enc = tiktoken.encoding_for_model(MODEL)
    encoding = enc.encode(text)

    chunks = [encoding[i:i + tokens] for i in range(0, len(encoding), tokens)]
    decoded = [enc.decode(chunk) for chunk in chunks]

    return decoded



