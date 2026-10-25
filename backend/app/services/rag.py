"""
BovineVision AI - RAG (Retrieval-Augmented Generation) Service
Maintains the verified cattle breed knowledge base, semantic chunking,
retrieval indexing, and citation-grounded synthesis.
Strictly decoupled from classification.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional


class RAGService:
    """
    RAG engine for BovineVision AI.
    Loads authoritative breed profiles from ICAR-NBAGR and national registries,
    indexes semantic passages, and executes grounded retrieval.
    """
    def __init__(self, kb_dir: str = "G:/Projektz/backend/knowledge_base"):
        self.kb_dir = kb_dir
        self.documents: List[Dict[str, Any]] = []
        self.breed_dossiers: Dict[str, Dict[str, Any]] = {}
        self.vocab: Dict[str, int] = {}
        self.doc_vectors: List[Dict[str, float]] = []
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        stopwords = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 'to', 'for', 'of', 'with',
            'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'or', 'has', 'have', 'were'
        }
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def _load_and_index(self):
        if not os.path.exists(self.kb_dir):
            print(f"[RAG] Warning: Knowledge base directory {self.kb_dir} does not exist.")
            return

        for fname in os.listdir(self.kb_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.kb_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        breed_key = data["breed_name"].lower()
                        self.breed_dossiers[breed_key] = data

                        # Semantic chunking of the breed profile
                        # Chunk 1: Identity & Classification
                        sci = data.get("scientific_classification", {})
                        chunk1 = {
                            "chunk_id": f"{breed_key}_classification",
                            "breed_id": data["id"],
                            "breed_name": data["breed_name"],
                            "section": "Classification & Registration",
                            "text": (
                                f"{data['breed_name']} is an indigenous Indian cattle breed belonging to {sci.get('species', 'Bos indicus')}. "
                                f"Official ICAR-NBAGR Accession Number: {sci.get('nbagr_accession', 'N/A')}. "
                                f"Functional Category: {sci.get('category', 'N/A')}. Conservation Status: {data.get('conservation_status', 'N/A')}."
                            ),
                            "citations": data.get("citations", [])
                        }
                        self.documents.append(chunk1)

                        # Chunk 2: Origin & Breeding Tract
                        orig = data.get("origin_and_geography", {})
                        chunk2 = {
                            "chunk_id": f"{breed_key}_geography",
                            "breed_id": data["id"],
                            "breed_name": data["breed_name"],
                            "section": "Origin & Geographic Distribution",
                            "text": (
                                f"Native breeding tract of {data['breed_name']} cattle: {orig.get('native_tract', 'N/A')}. "
                                f"Agro-climatic ecosystem: {orig.get('agro_climatic_zone', 'N/A')}. "
                                f"Historical background and pastoral heritage: {orig.get('historical_background', 'N/A')}."
                            ),
                            "citations": data.get("citations", [])
                        }
                        self.documents.append(chunk2)

                        # Chunk 3: Morphological Features
                        morph = data.get("morphological_traits", {})
                        chunk3 = {
                            "chunk_id": f"{breed_key}_morphology",
                            "breed_id": data["id"],
                            "breed_name": data["breed_name"],
                            "section": "Physical Characteristics & Morphology",
                            "text": (
                                f"Morphology of {data['breed_name']}: "
                                f"Horns: {morph.get('horns', 'N/A')}. "
                                f"Forehead: {morph.get('forehead', 'N/A')}. "
                                f"Coat color: {morph.get('coat_color', 'N/A')}. "
                                f"Ears: {morph.get('ears', 'N/A')}. "
                                f"Hump and dewlap: {morph.get('hump_and_dewlap', 'N/A')}. "
                                f"Body conformation: {morph.get('body_conformation', 'N/A')}."
                            ),
                            "citations": data.get("citations", [])
                        }
                        self.documents.append(chunk3)

                        # Chunk 4: Production & Utility
                        prod = data.get("production_and_utility", {})
                        chunk4 = {
                            "chunk_id": f"{breed_key}_production",
                            "breed_id": data["id"],
                            "breed_name": data["breed_name"],
                            "section": "Production & Utility",
                            "text": (
                                f"Production metrics for {data['breed_name']}: "
                                f"Primary utility: {prod.get('primary_utility', 'N/A')}. "
                                f"Average lactation milk yield: {prod.get('average_lactation_yield_kg', 'N/A')} over a lactation period of {prod.get('lactation_length_days', 'N/A')}. "
                                f"Milk fat percentage: {prod.get('fat_percentage', 'N/A')}. "
                                f"Draught power and work capacity: {prod.get('draught_capacity', 'N/A')}."
                            ),
                            "citations": data.get("citations", [])
                        }
                        self.documents.append(chunk4)

                        # Chunk 5: Management & Climate Resilience
                        mgmt = data.get("management_and_resilience", {})
                        chunk5 = {
                            "chunk_id": f"{breed_key}_resilience",
                            "breed_id": data["id"],
                            "breed_name": data["breed_name"],
                            "section": "Husbandry, Feeding & Resilience",
                            "text": (
                                f"Husbandry and adaptability of {data['breed_name']}: "
                                f"Diet and forage habits: {mgmt.get('diet_and_forage', 'N/A')}. "
                                f"Disease and climate resistance: {mgmt.get('disease_resistance', 'N/A')}. "
                                f"Temperament: {mgmt.get('temperament', 'N/A')}."
                            ),
                            "citations": data.get("citations", [])
                        }
                        self.documents.append(chunk5)

                except Exception as e:
                    print(f"[RAG] Error reading {fpath}: {e}")

        # Build TF-IDF vectors
        doc_count = len(self.documents)
        df = {}
        doc_token_counts = []
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            doc_token_counts.append(counts)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self.doc_vectors = []
        for counts in doc_token_counts:
            vec = {}
            total_tokens = sum(counts.values()) or 1
            for t, cnt in counts.items():
                tf = cnt / total_tokens
                idf = math.log((1 + doc_count) / (1 + df.get(t, 0))) + 1
                vec[t] = tf * idf
            # Normalize vector
            norm = math.sqrt(sum(v**2 for v in vec.values())) or 1.0
            vec = {t: v / norm for t, v in vec.items()}
            self.doc_vectors.append(vec)

        print(f"[RAG] Successfully indexed {len(self.breed_dossiers)} breeds and {len(self.documents)} semantic chunks.")

    def get_breed_dossier(self, breed_name: str, is_rejected: bool = False) -> Dict[str, Any]:
        """
        Retrieves the structured verified profile for an accepted cattle breed.
        Strictly raises ValueError if prediction was rejected.
        """
        if is_rejected:
            raise ValueError("RAG retrieval rejected: low confidence detection does not permit RAG activation.")

        key = breed_name.strip().lower()
        dossier = self.breed_dossiers.get(key)
        if not dossier:
            # Check partial match
            for k, d in self.breed_dossiers.items():
                if k in key or key in k:
                    return d
            raise KeyError(f"Breed '{breed_name}' not found in verified knowledge repository.")
        return dossier

    def retrieve(self, query: str, breed_name: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval for a given query with optional breed constraint.
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        # Vectorize query
        q_counts = {}
        for t in q_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1
        q_vec = {}
        for t, cnt in q_counts.items():
            idf = math.log((1 + len(self.documents)) / (1 + self.vocab.get(t, 1))) + 1
            q_vec[t] = (cnt / len(q_tokens)) * idf
        q_norm = math.sqrt(sum(v**2 for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        scores = []
        breed_filter = breed_name.strip().lower() if breed_name else None

        for idx, (doc, d_vec) in enumerate(zip(self.documents, self.doc_vectors)):
            if breed_filter and doc["breed_name"].lower() != breed_filter:
                continue

            dot = sum(q_vec.get(t, 0.0) * d_val for t, d_val in d_vec.items())
            if dot > 0.01:
                scores.append((dot, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, doc in scores[:top_k]:
            results.append({
                "chunk_id": doc["chunk_id"],
                "breed_name": doc["breed_name"],
                "section": doc["section"],
                "score": round(float(s), 4),
                "text": doc["text"],
                "citations": doc["citations"]
            })
        return results

    def answer_question(self, breed_name: str, question: str, is_rejected: bool = False) -> Dict[str, Any]:
        """
        Grounded QA: Retrieves verified passages and generates an evidence-backed answer.
        """
        if is_rejected:
            return {
                "allowed": False,
                "error": "RAG responses are prohibited for rejected predictions."
            }

        retrieved_chunks = self.retrieve(f"{breed_name} {question}", breed_name=breed_name, top_k=3)
        if not retrieved_chunks:
            # Fallback to breed profile
            dossier = self.get_breed_dossier(breed_name, is_rejected=False)
            return {
                "allowed": True,
                "question": question,
                "breed": breed_name,
                "answer": (
                    f"According to ICAR-NBAGR and national livestock documentation, {breed_name} is an indigenous {dossier['scientific_classification']['category']} "
                    f"native to {dossier['origin_and_geography']['native_tract']}. "
                    f"Average milk yield: {dossier['production_and_utility']['average_lactation_yield_kg']}, "
                    f"fat: {dossier['production_and_utility']['fat_percentage']}. "
                    f"Draught traits: {dossier['production_and_utility']['draught_capacity']}."
                ),
                "supporting_evidence": [],
                "citations": dossier.get("citations", [])
            }

        primary_chunk = retrieved_chunks[0]
        # Construct synthesis from retrieved evidence
        answer_text = f"Based on verified records ({primary_chunk['section']}):\n{primary_chunk['text']}"
        if len(retrieved_chunks) > 1 and retrieved_chunks[1]['score'] > 0.15:
            answer_text += f"\n\nAdditionally ({retrieved_chunks[1]['section']}):\n{retrieved_chunks[1]['text']}"

        return {
            "allowed": True,
            "question": question,
            "breed": breed_name,
            "answer": answer_text,
            "supporting_evidence": retrieved_chunks,
            "citations": primary_chunk.get("citations", [])
        }
