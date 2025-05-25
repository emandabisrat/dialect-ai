import praw
import pandas as pd
import os
from dotenv import load_dotenv
import re
from tqdm import tqdm

load_dotenv()
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="dialect-scraper/1.0"
)

def clean_text(text):
    text = re.sub(r'\[.*?\]\(.*?\)', '', text) 
    text = re.sub(r'\bhttps?:\/\/\S+', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

def scrape_subreddit(subreddit: str, label: str, limit: int = 100):
    posts = []
    try:
        for submission in tqdm(reddit.subreddit(subreddit).hot(limit=limit), 
                            desc=f"Scraping r/{subreddit}"):
            if submission.selftext and len(submission.selftext) > 50:
                text = clean_text(f"{submission.title} {submission.selftext}")
                posts.append({
                    "text": text,
                    "label": label,
                    "source": f"reddit/{subreddit}",
                    "length": len(text)
                })
    except Exception as e:
        print(f"Error in r/{subreddit}: {str(e)}")
    return pd.DataFrame(posts)

subreddit_map = {
    "Jamaican": ["Jamaica", "Caribbean"],
    "Nigerian": ["Nigeria", "NigerianFluency"],
    "Scottish": ["Scotland", "ScottishPeopleTwitter"],
    "Australian": ["Australia", "Straya"],
    "Southern US": ["Texas", "Appalachia"]
}

os.makedirs("data/raw", exist_ok=True)

for dialect, subreddits in subreddit_map.items():
    dfs = []
    for subreddit in subreddits:
        df = scrape_subreddit(subreddit, dialect)
        dfs.append(df)
    
    if dfs:
        combined = pd.concat(dfs)
        combined.to_csv(f"data/raw/{dialect.lower().replace(' ', '_')}.csv", index=False)

print("Reddit scraping complete")