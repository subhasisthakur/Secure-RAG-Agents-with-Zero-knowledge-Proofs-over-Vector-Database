import ollama
import json
import os
import random
import pandas as pd 

def generate_qa_with_ollama(text_to_process, model_name="llama3.2"):
    """
    Generates QA pairs from a given text using the Ollama Python library.
    """
    prompt = f"""
    You are an expert QA generator. Based solely on the following context,
    generate only 3 comprehensive question-answer pairs in a JSON array format. Only output the json response.
    Context: \"{text_to_process}\"
    """

    response = ollama.generate(model=model_name, prompt=prompt, stream=False)
    generated_text = response['response']


    return generated_text
