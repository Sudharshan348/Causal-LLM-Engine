import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from ai_engine.schema_embedder import retrieve_relevant_tables
except ModuleNotFoundError:
    from schema_embedder import retrieve_relevant_tables

load_dotenv()
client = genai.Client()

OFFLINE_CACHE = {
    "installment": {
        "treatment": "payment_installments",
        "outcome": "payment_value",
        "confounders": ["payment_sequential"],
        "sql_query": "SELECT payment_installments, payment_value, payment_sequential FROM order_payments WHERE payment_installments IS NOT NULL AND payment_value IS NOT NULL",
        "business_hypothesis": "Testing whether paying in a higher number of installments causes higher payment values when controlling for payment sequential."
    },
    "freight": {
        "treatment": "freight_value",
        "outcome": "review_score",
        "confounders": ["price", "product_weight_g"],
        "sql_query": "SELECT oi.freight_value, r.review_score, oi.price, p.product_weight_g FROM order_items oi JOIN order_reviews r ON oi.order_id = r.order_id JOIN products p ON oi.product_id = p.product_id WHERE oi.freight_value IS NOT NULL AND r.review_score IS NOT NULL",
        "business_hypothesis": "Testing whether higher freight costs negatively impact customer satisfaction and lead to lower review scores, after controlling for price and weight."
    },
    "photo": {
        "treatment": "product_photos_qty",
        "outcome": "review_score",
        "confounders": ["price"],
        "sql_query": "SELECT p.product_photos_qty, r.review_score, oi.price FROM order_reviews r JOIN order_items oi ON r.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE p.product_photos_qty IS NOT NULL AND r.review_score IS NOT NULL",
        "business_hypothesis": "Having more product photos increases customer review scores by setting clearer expectations, controlling for the product's price."
    },
    "description": {
        "treatment": "product_description_lenght",
        "outcome": "review_score",
        "confounders": ["price", "product_photos_qty"],
        "sql_query": "SELECT p.product_description_lenght AS product_description_lenght, r.review_score AS review_score, oi.price AS price, p.product_photos_qty AS product_photos_qty FROM order_reviews r JOIN order_items oi ON r.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE p.product_description_lenght IS NOT NULL AND r.review_score IS NOT NULL",
        "business_hypothesis": "Testing whether longer product descriptions lead to higher customer review scores, controlling for item price and photo count."
    },
    "weight": {
        "treatment": "product_weight_g",
        "outcome": "freight_value",
        "confounders": ["price"],
        "sql_query": "SELECT p.product_weight_g, oi.freight_value, oi.price FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.product_weight_g IS NOT NULL AND oi.freight_value IS NOT NULL AND oi.price IS NOT NULL",
        "business_hypothesis": "Heavier products lead to higher freight shipping costs even after controlling for the item's purchase price."
    },
    "length": {
        "treatment": "product_length_cm",
        "outcome": "freight_value",
        "confounders": ["product_weight_g"],
        "sql_query": "SELECT oi.freight_value, p.product_length_cm, p.product_weight_g FROM order_items oi JOIN products p ON oi.product_id = p.product_id WHERE p.product_length_cm IS NOT NULL AND p.product_weight_g IS NOT NULL",
        "business_hypothesis": "Increasing product length leads to higher freight shipping costs even after controlling for product weight."
    },
    "price": {
        "treatment": "price",
        "outcome": "review_score",
        "confounders": ["freight_value"],
        "sql_query": "SELECT oi.price AS price, r.review_score AS review_score, oi.freight_value AS freight_value FROM order_items oi JOIN order_reviews r ON oi.order_id = r.order_id WHERE oi.price IS NOT NULL AND r.review_score IS NOT NULL",
        "business_hypothesis": "Higher product prices lead to lower customer review scores when controlling for shipping fees."
    }
}

def parse_business_question(user_question: str) -> dict:
    """Retrieves relevant schema via RAG and extracts Causal Parameters into JSON with quota fallback."""
    
    rag_context = retrieve_relevant_tables(user_question, top_k=3)
    
    system_prompt = f"""
    You are an expert Data Scientist and Causal Inference Assistant.
    You have access to a relational SQLite database. Based on semantic search, here are the relevant tables for the user's query:
    
    {rag_context}
    
    Your job is to read the user's business question and output a strictly formatted JSON object.
    Do NOT wrap the output in markdown blocks (e.g., no ```json). Return raw JSON only with these exact keys and format:
    {{
        "sql_query": "A valid SQLite SELECT query that joins the necessary tables and returns the columns for treatment, outcome, and confounders.",
        "treatment": "The exact column name representing the cause.",
        "outcome": "The exact column name representing the effect.",
        "confounders": ["Array of exact column names acting as control variables."],
        "business_hypothesis": "A 1-sentence summary of what we are testing."
    }}
    """
    
    full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
    
    max_retries = 2
    response = None
    last_error = None
    models_to_try = ["gemini-3.8-flash"]
    
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                if response:
                    break
            except Exception as e:
                last_error = e
                if "503" in str(e) and attempt < max_retries - 1:
                    time.sleep(2)
                elif "404" in str(e) or "NOT_FOUND" in str(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    break
                else:
                    break
        if response:
            break
                
    if not response:
        # Fallback to local semantic cache if quota exceeded or offline
        q_lower = user_question.lower()
        for key, cached_val in OFFLINE_CACHE.items():
            if key in q_lower:
                print(f"Notice: Using cached semantic query plan for '{key}'.")
                return cached_val
        print(f"API Routing Error: Failed with error {str(last_error)}")
        return {}
        
    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
        
    except json.JSONDecodeError:
        print("Error: Model output was not valid JSON.")
        return {}
    except Exception as e:
        print(f"Unexpected parsing error: {str(e)}")
        return {}

if __name__ == "__main__":
    test_question = "Does shipping delay cause lower review scores, controlling for freight price?"
    print(f"Routing Question through RAG pipeline: '{test_question}'\n")
    
    result = parse_business_question(test_question)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Routing failed.")