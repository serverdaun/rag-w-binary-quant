from typing import Any, Generator
import numpy as np


# Helper function for batching
def batch_iterate(items: Any, batch_size: int) -> Generator[Any, None, None]:
    """
    Iterate over items in batches.

    Args:
        items: The items to iterate over
        batch_size: The size of the batches

    Returns:
        A generator of batches
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def generate_document_embeddings(documents: list[str], embedding_model: Any) -> list[bytes]:
    """
    Generate document embeddings.

    Args:
        documents: The documents to generate embeddings for
        embedding_model: The embedding model to use

    Returns:
        A list of document embeddings
    """
    binary_embeddings = []

    try:
        for context in batch_iterate(documents, batch_size=512):
            # generate float32 embeddings
            batch_embeddings = embedding_model.get_text_embedding_batch(context)

            # convert float32 to binary vectors
            embeds_array = np.array(batch_embeddings)
            binary_embeds = np.where(embeds_array > 0, 1, 0).astype(np.uint8)

            # convert to bytes array
            packed_embeds = np.packbits(binary_embeds, axis=1)

            binary_embeddings.extend(packed_embeds)
        return binary_embeddings
    except Exception as e:
        print(f"Error generating document embeddings: {e}")
        return []

def generate_query_embeddings(query: str, embdding_model: Any) -> bytes:
    """
    Generate query embeddings.

    Args:
        query: The query to generate embeddings for
        embdding_model: The embedding model to use

    Returns:
        A bytes array of query embeddings
    """
    try:
        # generate float32 embeddings
        query_embedding = embdding_model.get_text_embedding(query)

        # convert float32 to binary vector
        binary_vector = np.where(np.array(query_embedding) > 0, 1, 0).astype(np.uint8)

        # convert to bytes array
        packed_vector = np.packbits(binary_vector, axis=0)
        return packed_vector.tobytes()
    except Exception as e:
        print(f"Error generating query embeddings: {e}")
        return None

