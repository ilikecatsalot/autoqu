from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Ensure spaCy model is downloaded
def ensure_spacy_model():
    model_name = "en_core_web_md"
    try:
        spacy.load(model_name)
    except OSError:
        print(f"[INFO] Downloading spaCy model '{model_name}'...")
        from spacy.cli import download
        download(model_name)
    return spacy.load(model_name)


# Initialize resources
stop_words = set(stopwords.words('english'))
ner_tagger = ensure_spacy_model() 
vectorizer = TfidfVectorizer()

# Define the API endpoint
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    try:
        # Get JSON input from the request
        data = request.json
        print(f"[DEBUG] Received data: {data}")  # Debugging: Print received data
        initial_input_text = data.get("text", "")
        num_questions = data.get("num_questions", 3)

        # Ensure valid input
        if not initial_input_text:
            print("[ERROR] Missing 'text' field in the request")
            return jsonify({"error": "Missing 'text' field in the request"}), 400

        # Remove bracketed content
        input_text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', initial_input_text)
        print(f"[DEBUG] Cleaned text: {input_text}")  # Debugging: Print cleaned text

        # Perform Named Entity Recognition (NER)
        doc = ner_tagger(input_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"[DEBUG] Extracted Entities: {entities}")  # Debugging: Print extracted entities
        if not entities:
            return jsonify({"error": "No entities found in the input text."}), 400

        # Store valid entities
        valid_entities = [ent for ent in doc.ents]  # Keep all recognized entities
        print(f"[DEBUG] Valid entities: {valid_entities}")  # Debugging: Print valid entities

        # Perform TF-IDF scoring
        sentences = sent_tokenize(input_text)
        filtered_sentences = [
            ' '.join(w for w in word_tokenize(sentence) if w.lower() not in stop_words)
            for sentence in sentences
        ]
        if not filtered_sentences:
            return jsonify({"error": "No valid sentences for question generation."}), 400
        print(f"[DEBUG] Filtered sentences: {filtered_sentences}")  # Debugging: Print filtered sentences

        # Generate TF-IDF vectors
        tf_idf_vector = vectorizer.fit_transform(filtered_sentences)
        feature_names = vectorizer.get_feature_names_out()
        tf_idf_matrix = tf_idf_vector.todense().tolist()

        # Compute word scores
        word_score = {}
        sentence_for_max_word_score = {}
        for i, word in enumerate(feature_names):
            scores = [row[i] for row in tf_idf_matrix]
            word_score[word] = sum(scores) / len(scores)
            max_idx = scores.index(max(scores))
            sentence_for_max_word_score[word] = sentences[max_idx]

        # Rank keywords based on TF-IDF, but use all valid entities
        candidate_keywords = [ent.text for ent in valid_entities]
        candidate_triples = [
            (get_keyword_score(keyword, word_score), keyword, get_corresponding_sentence(keyword, sentence_for_max_word_score))
            for keyword in candidate_keywords
        ]
        candidate_triples.sort(reverse=True, key=lambda x: x[0])
        print(f"[DEBUG] Candidate Triples: {candidate_triples}")  # Debugging: Print candidate triples

        # Generate questions
        questions_dict = {}
        used_sentences = set()
        cntr = 1
        idx = 0
        while cntr <= num_questions and idx < len(candidate_triples):
            _, keyword, sentence = candidate_triples[idx]
            if sentence and sentence not in used_sentences:
                used_sentences.add(sentence)
                question_text = sentence.replace(keyword, "__________")
                answer = keyword

                # Find the entity and label it (ensure entity is a span)
                entity = None
                for ent in doc.ents:
                    if ent.text == keyword:
                        entity = ent
                        break
                label_answer = entity.label_ if entity else None  # Get label of the entity
                print(f"[DEBUG] Entity: {entity}, Label: {label_answer}")  # Debugging: Print entity label

                # Add the question with its entity label and answer
                questions_dict[cntr] = {
                    "question": question_text,
                    "answer": answer,
                    "entity": keyword,  # Return the entity text
                    "label": label_answer  # Return the entity type (label)
                }
                cntr += 1
            idx += 1

        if not questions_dict:
            return jsonify({"error": "Unable to generate questions."}), 400

        return jsonify({"questions": questions_dict})

    except Exception as e:
        print(f"[ERROR] {str(e)}")  # More detailed error message
        return jsonify({"error": "Error generating questions. Please try again."}), 500


def get_keyword_score(keyword, word_score):
    score = sum(word_score.get(word.lower(), 0) for word in word_tokenize(keyword))
    return score


def get_corresponding_sentence(keyword, sentence_for_max_word_score):
    for word in word_tokenize(keyword):
        sentence = sentence_for_max_word_score.get(word.lower(), "")
        if all(w.lower() in sentence.lower() for w in word_tokenize(keyword)):
            return sentence
    return ""


@app.route('/')
def home():
    return send_from_directory('templates', 'index.html')


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
