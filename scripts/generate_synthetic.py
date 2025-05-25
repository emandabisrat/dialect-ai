import os
import re
import pandas as pd
from mistralai.client import MistralClient
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

prompts_dialect = {
    "Jamaican": (
        "Generate ONLY 50 authentic Jamaican Patois sentences (no numbering, no intros). "
        "Each must contain at least 2 Jamaican words like 'mi', 'deh', 'yuh', 'bwoy'. "
        "Example format:\n"
        "Mi deh yah a chill\n"
        "Wah gwaan wid yuh\n"
        "... [48 more]"
    ),
    "Nigerian": (
        "Generate ONLY 50 Nigerian Pidgin sentences (no explanations). "
        "Each must include 'abeg', 'na', or 'dey'. "
        "Example format:\n"
        "Abeg no vex\n"
        "I dey come now\n"
        "... [48 more]"
    ),
    "Scottish": (
        "Generate ONLY 50 authentic Scottish English sentences (no numbering, no intros). "
        "Each must contain at least 2 Scottish words like 'wee', 'bairn', 'ken'. "
        "Example format:\n"
        "It's a wee bit chilly\n"
        "The bairn is sleeping\n"
        "... [48 more]"
    ),
    "Australian": (
        "Generate ONLY 50 authentic Australian English sentences (no numbering, no intros). "
        "Each must contain at least 2 Australian words like 'mate', 'arvo', 'bush'. "
        "Example format:\n"
        "G'day mate, how's it going?\n"
        "Let's hit the bush this arvo\n"
        "... [48 more]"
    ),
    "Southern US": (
        "Generate ONLY 50 authentic Southern US English sentences (no numbering, no intros). "
        "Each must contain at least 2 Southern words like 'y'all', 'fixin'', 'hushpuppies'. "
        "Example format:\n"
        "Y'all come back now, ya hear?\n"
        "I'm fixin' to go to the store\n"
        "... [48 more]"
    ),        
}

def generate_dialect_samples(dialect: str, n: int = 50):
    prompt = prompts_dialect[dialect]
    
    for _ in range(3):  
        try:
            response = client.chat(
                model="mistral-small",
                messages=[{
                    "role": "user", 
                    "content": prompt + "\nIMPORTANT: RETURN ONLY SENTENCES, NO INTRODUCTORY TEXT"
                }],
                temperature=0.7
            )
            raw_lines = response.choices[0].message.content.split('\n')
            samples = [
                line.strip() for line in raw_lines 
                if line.strip() 
                and not line.startswith(('Here are', 'Example', '---', '1.', '2.', '*'))
                and len(line.split()) >= 3  
            ]
            
            return samples[:n]
            
        except Exception as e:
            print(f"Retrying {dialect}: {str(e)}")
            time.sleep(5)
    
    return []  

os.makedirs("data/synthetic", exist_ok=True)
all_samples = []

for dialect in prompts_dialect:
    print(f"Generating {dialect}...")
    samples = generate_dialect_samples(dialect)
    samples = generate_dialect_samples(dialect) 
        
    df = pd.DataFrame({"text": samples, "label": dialect, "source": "synthetic"})
    df.to_csv(f"data/synthetic/{dialect.lower()}.csv", index=False)
    all_samples.append(df)

full_df = pd.concat(all_samples)
full_df.to_csv("data/synthetic/all_dialects.csv", index=False)
print(f"\n Generated {len(full_df)} valid samples")