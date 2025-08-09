from pymilvus import DataType, MilvusClient


def get_milvus_client(db_path: str) -> MilvusClient:
    """
    Get a Milvus client.

    Args:
        db_path: The path to the Milvus database

    Returns:
        A Milvus client
    """
    try:
        client = MilvusClient(db_path)
        return client

    except Exception as e:
        print(f"Error getting Milvus client: {e}")
        return None


def create_collection_if_not_exists(
    client: MilvusClient, collection_name: str, dim: int
) -> None:
    """
    Create a collection in Milvus if it does not exist.

    Args:
        client: The Milvus client
        collection_name: The name of the collection to create
        dim: The dimension of the binary vector
    """
    try:
        # Create collection only if it does not exist
        if not client.has_collection(collection_name):
            print(f"Collection {collection_name} not found. Creating it...")

            schema = client.create_schema(
                auto_id=True,
                enable_dynamic_fields=True,
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            )
            schema.add_field(
                field_name="context",
                datatype=DataType.VARCHAR,
                max_length=65535,
            )
            schema.add_field(
                field_name="binary_vector",
                datatype=DataType.BINARY_VECTOR,
                dim=dim,
            )

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="binary_vector",
                index_name="binary_vector_index",
                index_type="BIN_FLAT",
                metric_type="HAMMING",
            )

            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            print(f"Collection {collection_name} created successfully.")
        else:
            print(f"Collection {collection_name} already exists. Skipping creation.")
    except Exception as e:
        print(f"Error creating collection: {e}")
        return None


def insert_data(client: MilvusClient, collection_name: str, data: list[dict]):
    """
    Insert data into a collection in Milvus.

    Args:
        client: The Milvus client
        collection_name: The name of the collection to insert data into
        data: The data to insert
    """
    try:
        client.insert(
            collection_name=collection_name,
            data=data,
        )
    except Exception as e:
        print(f"Error inserting data: {e}")


def search(
    client: MilvusClient, collection_name: str, binary_query: bytes, limit: int = 5
):
    """
    Search for data in a collection in Milvus.
    """
    try:
        # Search for data
        results = client.search(
            collection_name=collection_name,
            data=[binary_query],
            anns_field="binary_vector",
            search_params={
                "metric_type": "HAMMING",
            },
            output_fields=["context"],
            limit=limit,
        )

        if not results:
            print("No search results found")
            return []

        contexts = [res.entity.context for res in results[0]]
        return contexts

    except Exception as e:
        print(f"Error searching for data: {e}")
        return []
