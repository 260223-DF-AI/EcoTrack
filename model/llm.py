"""Refactor my file name later"""
from google import genai
from dotenv import load_dotenv
import os
import graph_of_thoughts
import json

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

def animal_loc_analysis(class_pred: dict, additional_info: str):
    # additional_info = sanatize_input(additional_info) # uncomment if you have cuda cores to sanatize input before passing it to gemini

    prompt = f"{IN_CONTEXT}\n1. {OUT_FORMAT}\n2. {class_pred['species']}\n3. {additional_info}"
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=prompt,
        config={
            'temperature': 0.0
        }
    )
    print(response.text)

# Can I get the LLM to tell me the usual habitat of a species?

if __name__ == '__main__':
    my_input = "Zoo in Sichuan China"
    animal_loc_analysis({'species': 'panda'}, my_input)
