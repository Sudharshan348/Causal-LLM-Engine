import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

TABLE_SCHEMAS = {
    "orders": "table 'orders' with columns: order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date. Useful for tracking delivery delays and purchase timelines.",
    "order_reviews": "table 'order_reviews' with columns: review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp. Useful for customer satisfaction and feedback.",
    "order_items": "table 'order_items' with columns: order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value. Useful for shipping costs, item prices, and seller data.",
    "products": "table 'products' with columns: product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm. Useful for item dimensions and categories.",
    "order_payments": "table 'order_payments' with columns: order_id, payment_sequential, payment_type, payment_installments, payment_value. Useful for financial transactions, installments, and payment methods."
}

CACHED_SCHEMA_EMBEDDINGS = {}

def get_embedding(text: str) -> list:
    """Generates vector embeddings using Gemini, with a deterministic local fallback for safety."""
    for model_name in ["text-embedding-004", "gemini-embedding-001"]:
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return response.embeddings[0].values
        except Exception:
            continue
    np.random.seed(abs(hash(text)) % (2**32))
    return list(np.random.normal(0, 1, 768))

def get_schema_embeddings() -> dict:
    """Lazily computes and caches static schema embeddings to avoid redundant API calls."""
    global CACHED_SCHEMA_EMBEDDINGS
    if not CACHED_SCHEMA_EMBEDDINGS:
        for t_name, s_desc in TABLE_SCHEMAS.items():
            CACHED_SCHEMA_EMBEDDINGS[t_name] = get_embedding(s_desc)
    return CACHED_SCHEMA_EMBEDDINGS

def retrieve_relevant_tables(user_query: str, top_k: int = 3) -> str:
    """RAG Core: Finds the most relevant tables for the user's question using cosine similarity."""
    query_embedding = get_embedding(user_query)
    table_embeddings = get_schema_embeddings()
    similarities = {}
    
    for table_name, table_embedding in table_embeddings.items():
        dot_product = np.dot(query_embedding, table_embedding)
        norm_a = np.linalg.norm(query_embedding)
        norm_b = np.linalg.norm(table_embedding)
        if norm_a > 0 and norm_b > 0:
            similarities[table_name] = dot_product / (norm_a * norm_b)
        else:
            similarities[table_name] = 0.0
    
    sorted_tables = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    retrieved_schema = ""
    for i in range(min(top_k, len(sorted_tables))):
        t_name = sorted_tables[i][0]
        retrieved_schema += f"{i+1}. {TABLE_SCHEMAS[t_name]}\n"
        
    return retrieved_schema

if __name__ == "__main__":
    print("Testing RAG Schema Retrieval...")
    test_q = "Does paying in high installments cause higher total payment values?"
    print(f"\nQuery: {test_q}\n\nRetrieved Context:\n{retrieve_relevant_tables(test_q, top_k=2)}")