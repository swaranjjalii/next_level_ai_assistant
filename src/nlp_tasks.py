from .llm_abstraction import GeminiLLM
from .retrieval import VectorRetriever

class NLPTasks:
    def __init__(self):
        self.llm = GeminiLLM()
        self.retriever = VectorRetriever()

    def summarize(self, text: str) -> str:
        prompt = f"Summarize this text concisely:\n{text}"
        return self.llm.generate(prompt)

    def sentiment_analysis(self, text: str) -> str:
        prompt = f"Classify sentiment (positive/negative/neutral) for:\n{text}"
        return self.llm.generate(prompt)

    def named_entity_recognition(self, text: str) -> str:
        prompt = (
            "Extract named entities (people, organizations, locations, dates) from this text. "
            "Format the output as JSON with entity types as keys and lists of entities as values:\n" + text
        )
        return self.llm.generate(prompt)

    def question_answering(self, question: str, context: str = None) -> str:
        if context:
            prompt = f"Answer this question based on the provided context.\nQuestion: {question}\nContext: {context}"
        else:
            # Use RAG to find relevant context if not provided
            relevant_docs = self.retriever.search(question, top_k=2)
            if relevant_docs:
                contexts = [self.retriever.get_document_text(doc_id) for doc_id, _ in relevant_docs]
                context_text = "\n".join(contexts)
                prompt = f"Answer this question based on the provided context.\nQuestion: {question}\nContext: {context_text}"
            else:
                prompt = f"Answer this question concisely: {question}"
        
        return self.llm.generate(prompt)

    def code_generation(self, problem_statement: str) -> str:
        prompt = (
            f"Generate a Python code solution for the following problem:\n{problem_statement}\n\n"
            "Provide the solution with proper documentation and explanation."
        )
        return self.llm.generate(prompt)
    
    def code_review(self, code: str) -> str:
        prompt = (
            f"Review the following Python code for potential issues, bugs, or improvements:\n```python\n{code}\n```\n\n"
            "Provide specific feedback and suggestions for improvement."
        )
        return self.llm.generate(prompt)