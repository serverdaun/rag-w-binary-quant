from llama_index.core import SimpleDirectoryReader


def load_data(data_dir: str) -> list:
    """
    Load a data from a directory

    Args:
        data_dir: The directory to load the data from

    Returns:
        A list of documents
    """
    try:
        loader = SimpleDirectoryReader(
            input_dir=data_dir,
            required_exts=[".pdf", ".txt", ".md", ".docx", ".doc"],
            recursive=True,
        )
        docs = loader.load_data()
        return docs
    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        return []
