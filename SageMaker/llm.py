from google import genai
from dotenv import load_dotenv
import os
import graph_of_thoughts
import json
import re
"""In order to be able to run this file you have to:
1. Go to google AI studio (you can get started for free)
2a. Navigate to the API Keys tab on the right if not already there
2b. If you don't see a default key or don't want to use that one, click on 'Create API key' and fill out the fields and 'Create key' when done
2c. Copy the API key of your choosing
3a. Create .env file if it doesn't already exist
3b. Add 'GEMINI_API_KEY=<your api key>' line to your .env file
"""

load_dotenv()
KEY = os.getenv('GEMINI_API_KEY')

# list things for the model to consider
IN_CONTEXT = """You are using a ReAct loop using a Graph of Thought paradigm to 
determine if a given endangered animal is in an unusual location based on a description provided."""
OUT_FORMAT = "{'unusual_location': True or False, 'reason': '', 'confidence': 0-100}" # desired output format
RESPONSE_FORMAT = {"type": "object",
                   "properties": {
                        'unusual_location': {
                            "type": "boolean",
                        }, 
                        'reason': {
                            "type": "string",
                        }, 
                        'llm_confidence': {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 100.0,
                        }
                    }, "required": ["unusual_location", "reason", "llm_confidence"]
                }
INJECTION_PATTERNS = [
    r"ignore (all |your )?(previous |prior )?instructions",
    r"you are now",
    r"disregard (all |your )?(previous |prior )?(instructions|context)",
]

MAX_CHARS = 2000

# create gemini client
client = genai.Client(api_key=KEY)

# --------------- Graph of Thought Components -----------------------



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
    prompt = f"""{IN_CONTEXT}

Output as a string enclosed in double quotes only and strictly adhere to this format (put your entire Graph of Thought process into the 'reason' section):
{OUT_FORMAT}

The species detected in the image:
{class_pred['species']}

Additional information/context provided by user:
\"\"\"
{additional_info}
\"\"\"
"""
    
    response = client.models.generate_content(
        # model="gemini-3-flash-preview",
        model="gemini-2.5-flash-lite", 
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_FORMAT,
            'temperature': 0.0,
            'top_p': 0.1
        }
    )
    print(response.text)
    response = json.loads(response.text)
    print(response)

    return response

if __name__ == '__main__':
    my_input = "Zoo in Sichuan China"
    animal_loc_analysis({'species': 'panda'}, my_input)
