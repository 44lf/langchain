from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingUtils:
    @staticmethod
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese"
        )

    def embed_documents(self, texts: list[str]):
        embeddings = self.get_embeddings()
        vectors = embeddings.embed_documents(texts)
        return vectors

    def embed_query(self, text: str):
        embeddings = self.get_embeddings()
        vector = embeddings.embed_query(text)
        return vector

