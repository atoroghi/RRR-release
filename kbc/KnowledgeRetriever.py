# import typing_extensions
# from importlib import reload
# reload(typing_extensions)
from sentence_transformers import SentenceTransformer, util


class KnowledgeRetriever:
    def __init__(self, model_dir):
        self.mpnet = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            cache_folder=f"{model_dir}/sentence_transformers",
        )

    def __call__(self, question, facts, k_facts=10):
        question_embedding = self.encode([question])
        facts_embeddings = self.encode(facts)

        results = util.semantic_search(
            question_embedding, facts_embeddings, score_function=util.dot_score, top_k=k_facts
        )[0]

        return [facts[result["corpus_id"]] for result in results]

    def encode(self, s):
        return self.mpnet.encode(s, normalize_embeddings=True, convert_to_tensor=True)