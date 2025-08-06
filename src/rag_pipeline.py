from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from .config import PROMPT, MODEL_NAME, TEMPERATURE, MODEL_PROVIDER


llm = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER, temperature=TEMPERATURE)

def answer_question(query: str, contexts: list[str]) -> str:
    """
    Answer a question using the provided context.

    Args:
        query: The query to answer
        contexts: The context to use for answering the question

    Returns:
        The answer to the question
    """
    prompt = PROMPT.format(contexts=contexts, query=query)
    human_message = HumanMessage(content=prompt)

    response = llm.invoke([human_message])
    return response.content
