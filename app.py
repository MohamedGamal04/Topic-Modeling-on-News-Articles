import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from nltk.corpus import wordnet
import contractions
from nltk.corpus import stopwords
import re 
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#Preprocess
def preprocess_text(text):
    text = contractions.fix(text)
    text = text.replace('.', ' . ')
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = "".join(num2words(int(word)) if word.isdigit() else word for word in text)
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.tag.pos_tag(text)
    lemmatized_words = []

    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_words)

# Streamlit App
import pickle
@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, nmf = load_model()

# Function to predict topics using NMF
def predict_nmf_topics(text, vectorizer, nmf_model):
    """
    Predict topics for input text using NMF.
    
    Parameters
    ----------
    text : str
        Input text to classify
    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Fitted TF-IDF vectorizer
    nmf_model : sklearn.decomposition.NMF
        Trained NMF model
    
    Returns
    -------
    dict
        Dictionary with 'topic' (dominant topic), 'weights' (topic distribution),
        and 'confidence' (max topic weight)
    """
    # Transform text to TF-IDF
    tfidf = vectorizer.transform([text])
    
    # Get topic distribution (NMF's W matrix reconstruction for this document)
    topic_dist = nmf_model.transform(tfidf)[0]
    
    # Get dominant topic and confidence
    dominant_topic = topic_dist.argmax()
    confidence = topic_dist.max()
    
    return {
        'topic': dominant_topic,
        'weights': topic_dist,
        'confidence': confidence
    }

st.set_page_config(page_title="News Topic Classification", page_icon="📰", layout="wide")

st.title("📰 Topic Classification")
st.markdown("### Classify news articles into different topics using Machine Learning")

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")
    st.info("""
    **Model:** Non-negative Matrix Factorization (NMF)
    
    **Accuracy:** ~80%

    **Dataset:** BBC News Dataset
    """)
    
    st.header("About")
    st.markdown("""
    This app uses a trained Non-negative Matrix Factorization (NMF) model with TF-IDF vectorization
    to predict topics from text.
    
    The preprocessing includes:
    - Contraction expansion
    - HTML tag removal
    - Stopword removal
    - Lemmatization
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Insights"])

with tab1:
    st.header("Single News Article Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter your news article:",
        height=150,
        placeholder="Type or paste your news article here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Classify a News Article", type="primary")
    
    if predict_button and user_input:
        with st.spinner("Processing..."):
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # Predict using NMF
            result = predict_nmf_topics(processed_text, vectorizer, nmf)
            dominant_topic = result['topic']
            confidence = result['confidence']
            topic_weights = result['weights']
            
            st.success("✅ Analysis Complete!")
            
            st.metric(
                    "Dominant Topic",
                    f"Topic {dominant_topic}",
            )
            
            
            # Show topic distribution
            st.markdown("### Topic Distribution")
            topic_df = pd.DataFrame({
                'Topic': [f'Topic {i}' for i in range(len(topic_weights))],
                'Weight': topic_weights
            })
            st.bar_chart(topic_df.set_index('Topic'))
            
            # Show processed text
            with st.expander("🔎 View Preprocessed Text"):
                st.text(processed_text)
    
    elif predict_button and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("Batch News Analysis")

    uploaded_file = st.file_uploader("Upload a CSV file with news articles", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🚀 Classify All News Articles") and df['title'].notna().any() and df['description'].notna().any():
            with st.spinner("Processing batch..."):
                # Process all articles
                df['processed_description'] = df['description'].apply(preprocess_text)
                df['processed_title'] = df['title'].apply(preprocess_text)
                df['text_combined'] = df['title'] + ' ' + df['description']

                # Transform and predict topics for each article
                tfidf = vectorizer.transform(df['text_combined'])
                topic_dist = nmf.transform(tfidf)
                df['dominant_topic'] = topic_dist.argmax(axis=1)
                df['topic_confidence'] = topic_dist.max(axis=1)
                
                st.success("✅ Batch analysis complete!")
                st.dataframe(df[['title', 'dominant_topic', 'topic_confidence']])
                
                # Show statistics
                topic_counts = df['dominant_topic'].value_counts().sort_index()
                st.markdown("### Topic Distribution")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Total Articles", len(df))
                    for topic_id in range(nmf.n_components):
                        count = len(df[df['dominant_topic'] == topic_id])
                        st.metric(f"Topic {topic_id}", count)
                
                with col2:
                    dist_df = pd.DataFrame({
                        'Topic': [f'Topic {i}' for i in topic_counts.index],
                        'Count': topic_counts.values
                    })
                    st.bar_chart(dist_df.set_index('Topic'))
        else:
            st.info("ℹ️ Please upload a CSV file with a 'Title' and 'Description' column to analyze.")

with tab3:
    st.header("Model Insights & Feature Importance")
    
    st.markdown("""
    ### Top Features in Each Topic
    These words have the strongest influence on topic modeling.
    """)
    
    # Example feature importance (replace with actual from your model)
    top_topic_0 = ['say', 'year', 'police', 'old', 'woman', 'find', 'die', 'man', 'murder', 'child', 'two', 'people']
    top_topic_1 = ['world', 'cup', 'final', '2022', 'woman', 'semi', 'qatar', 'france', 'argentina', 'beat', 'reach', '2023']
    top_topic_2 = ['ukraine', 'war', 'russia', 'russian', 'ukrainian', 'putin', 'invasion', 'kyiv', 'say', 'president', 'zelensky', 'attack']
    top_topic_3 = ['rise', 'price', 'rate', 'cost', 'uk', 'energy', 'living', 'bill', 'high', 'inflation', 'interest', 'much']
    top_topic_4 = ['election', 'party', 'minister', 'sunak', 'labour', 'tory', 'rishi', 'pm', 'vote', 'paper', 'uk', 'prime']
    top_topic_5 = ['day', 'seven', 'week', 'past', 'go', 'quiz', 'closely', 'attention', 'pay', 'picture', 'selection', 'take']
    top_topic_6 = ['league', 'city', 'manchester', 'premier', 'win', 'one', 'champion', 'liverpool', 'two', 'arsenal', 'zero', 'title']
    top_topic_7 = ['israel', 'gaza', 'hamas', 'attack', 'israeli', "hostage", "palestinian", "kill", "war", "people", "say", "ceasefire"]
    top_topic_8 = ['england', 'euro', 'test', '2024', 'win', 'wale', 'southgate', 'nation', 'game', 'australia', 'series', 'six']
    top_topic_9 = ['strike', 'train', 'rail', 'worker', 'union', 'pay', 'action', 'disruption', 'walkout', 'affect', 'driver', 'teacher']

    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    with col1:
        st.success("**Top 'topic 0' Words**")
        for i, word in enumerate(top_topic_0, 1):
            st.write(f"{i}. {word}")
    
    with col2:
        st.error("**Top 'topic 1' Words**")
        for i, word in enumerate(top_topic_1, 1):
            st.write(f"{i}. {word}")

    with col3:
        st.warning("**Top 'topic 2' Words**")
        for i, word in enumerate(top_topic_2, 1):
            st.write(f"{i}. {word}")

    with col4:
        st.info("**Top 'topic 3' Words**")
        for i, word in enumerate(top_topic_3, 1):
            st.write(f"{i}. {word}")

    with col5:
        st.success("**Top 'topic 4' Words**")
        for i, word in enumerate(top_topic_4, 1):
            st.write(f"{i}. {word}")

    with col6:
        st.error("**Top 'topic 5' Words**")
        for i, word in enumerate(top_topic_5, 1):
            st.write(f"{i}. {word}")
    
    with col7:
        st.warning("**Top 'topic 6' Words**")
        for i, word in enumerate(top_topic_6, 1):
            st.write(f"{i}. {word}")

    with col8:
        st.info("**Top 'topic 7' Words**")
        for i, word in enumerate(top_topic_7, 1):
            st.write(f"{i}. {word}")
    
    with col9:
        st.success("**Top 'topic 8' Words**")
        for i, word in enumerate(top_topic_8, 1):
            st.write(f"{i}. {word}")

    with col10:
        st.error("**Top 'topic 9' Words**")
        for i, word in enumerate(top_topic_9, 1):
            st.write(f"{i}. {word}")        

    st.markdown("---")
    st.info("""
    **To use your trained model:**
    
    ```
    # Save your model and vectorizer:
    import pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # Load in Streamlit:
    @st.cache_resource
    def load_model():
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    
    vectorizer, clf = load_model()
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Built with Streamlit | Powered by NMF & TF-IDF</p>"
    "</div>",
    unsafe_allow_html=True
)