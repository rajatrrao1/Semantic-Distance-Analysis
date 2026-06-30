This is the repo to compute semantic distance as computed by the following models

	• sentence-transformers/all-MiniLM-L6-v2  – a small, fast model for general-purpose sentence embeddings, suitable for large datasets and interactive use (Reimers & Gurevych, 2019).
	• sentence-transformers/all-mpnet-base-v2  – a higher-capacity encoder with stronger semantic similarity performance than MiniLM, useful when fine‑grained differences matter more than speed (Reimers & Gurevych, 2019).
	• BAAI/bge-large-en-v1.5  – a large English embedding model optimised for retrieval and semantic matching, recommended when you have fewer texts and want more nuanced distances (Xiao et al., 2023).
	• bert-base-uncased  – the original BERT base model; not specifically tuned for sentence similarity, mainly for experimentation and comparison with specialised sentence‑transformer models (Devlin et al., 2019).

If you wish to run it online, please use the online app: 
https://semantic-distance-analysis.streamlit.app/

Instructions to use app:
•	The app opens in your browser.
	•	Upload a CSV file with at least two columns:
	•	one column with group labels (e.g. conditions, artworks),
	•	one column with the texts.
	•	In Settings, select:
	•	the Embedding model,
	•	the Text column,
	•	the Group column,
	•	and the minimum number of texts per group to include.
	•	Click “Run semantic distance analysis” to compute within‑group semantic distances and view/download the results.


If you wish to run it offline, here are the installation instructions:

1. Clone the repository
git clone https://github.com/rajatrrao1/Semantic-Distance-Analysis.git
cd Semantic-Distance-Analysis

2. Create a virtual environment
  python -m venv semantic_distance
  source semantic_distance/bin/activate   # macOS / Linux
  semantic_distance\Scripts\activate    # Windows (PowerShell / CMD)

4. Install dependencies
   pip install -r requirements.txt

5. Run app locally
   streamlit run app.py



Reference
1. All-MiniLM-L6-v2 model card, Hugging Face: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. Pretrained models – SentenceTransformers documentation: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
3. BAAI/bge-large-en-v1.5 model card, Hugging Face: https://huggingface.co/BAAI/bge-large-en-v1.5
4. SentenceTransformers organization page, Hugging Face (embedding models overview): https://huggingface.co/sentence-transformers
5. bert-base-uncased model card, Hugging Face: https://huggingface.co/bert-base-uncased


If you use this tool please cite us as
Ravi Rao, R., mulckhuyse, m., van Elk, M., & Walker, F. (n.d.). Appreciating Ambiguity in Art: The effects of semantic distance and insight confidence on aesthetic appreciation. Retrieved from osf.io/preprints/psyarxiv/h8ztf_v1
