# Dialect Classification 

This project aims to build a classification model for 5 unique dialects of English: Scottish, Australian, Jamaican Patois, Nigerian Pidgin, and Southern U.S. 

## Clone the git repository 

To clone this repository, in your terminal run: git clone https://github.com/emandabisrat/dialect-ai.git

## Install Dependencies 

To install all dependencies, in your terminal run: pip install -r requirements.txt

## Run the Pipeline

Run these python files in this specific order:

python scripts/scrape_Reddit.py
python scripts/generate_synthetic.py
python scripts/combine_data.py

## Train Model

Run the notebooks/model.ipynb to train and evaluate the model.




