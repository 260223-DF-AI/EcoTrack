"""Refactor my file name later"""
from google import genai
from dotenv import load_dotenv
import os
from graph_of_thoughts import controller, language_models, operations
import json
import re

load_dotenv()
KEY = os.getenv('GEMINI_API_KEY')

IN_CONTEXT = """You are using a ReAct loop and a Graph of Thought paradigm to 
figure out whether or not a given endangered animal is in an unusual location 
based on a description provided. 
The line indicated by '1. ' is the the output format for you to strictly adhere to. In the 'reason' section, document your thoughts there
The line indicated by '2. ' is the name of the animal to evaluate
The line indicated by '3. ' is the description provided by the user possibly containing 
various information on the context the animal was in"""
OUT_FORMAT = "{'unusual_location': True or False, 'reason': '', 'confidence': 0-100}" # desired output format
INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous |prior )?instructions",
    r"you are now",
    r"disregard (all |your )?(previous |prior )?(instructions|context)",
]

MAX_CHARS = 2000

# create gemini client
client = genai.Client(api_key=KEY)

# Have code for input sanitation here and compression, aka preprocessing input before passing it through to the LLM
# can uncomment if you have cuda cores
# from presidio_analyzer import AnalyzerEngine
# from presidio_anonymizer import AnonymizerEngine
# from llmlingua import PromptCompressor

# analyzer = AnalyzerEngine()
# anonymizer = AnonymizerEngine()
# compressor = PromptCompressor()

# def sanitize_input(input: str):
    # # Detect PII entities in the text
    # results = analyzer.analyze(text=input, language="en")

    # # Replace detected PII with placeholder tags
    # anonymized = anonymizer.anonymize(text=input, analyzer_results=results).text

    # compressed = compressor.compress_prompt(anonymized, rate=0.5)
    # return compressed

def preprocess_uinput(raw_input: str) -> str:
    """Preprocesses user provided textual input by validating and sanitizing it before sending it through the LLM

    Args:
        raw_input - str of the raw, unprocessed user text input
    Returns:
        the processed user input as a string
    """
    # validate 
    if len(raw_input) > MAX_CHARS:
        raise # some sort of specific error

    processed_input = raw_input.strip()
    if not processed_input:
        raise # another sort of specific error 

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, processed_input, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected. Request rejected.")
    
    return processed_input

def animal_loc_analysis(class_pred: dict, additional_info: str):
    """Don't forget to write your doc comment, Isabelle"""
    # additional_info = sanitize_input(additional_info) # uncomment if you have cuda cores to sanitize input before passing it to gemini
    additional_info = preprocess_uinput(additional_info)
    prompt = f"{IN_CONTEXT}\n1. {OUT_FORMAT}\n2. {class_pred['species']}\n3. {additional_info}"
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt,
        config={
            'temperature': 0.0
        }
    )
    print(response.text)

if __name__ == '__main__':
    my_input = "Zoo in Sichuan China"
    animal_loc_analysis({'species': 'panda'}, my_input)
