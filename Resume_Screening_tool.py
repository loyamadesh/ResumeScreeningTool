import streamlit as st
import pdfplumber
import re
import spacy
import pandas as pd
import io # Needed for handling file buffer

# --- Global Configurations and Data ---

# Load the spaCy model once
@st.cache_resource
def load_spacy_model():
    """Load the spaCy model, caching it for efficiency."""
    try:
        # Using a smaller model for faster loading
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        return None

nlp = load_spacy_model()

# A large dictionary of common technical and non-technical skills
SKILLS_LIST = [
    'Python', 'Java', 'C++', 'SQL', 'R', 'JavaScript', 'HTML', 'CSS',
    'Machine Learning', 'Deep Learning', 'NLP', 'Data Science', 'TensorFlow', 
    'PyTorch', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'Git', 'Agile',
    'Communication', 'Leadership', 'Teamwork', 'Pandas', 'NumPy', 'Scikit-learn',
    'Tableau', 'Excel', 'MongoDB', 'PostgreSQL', 'Scrum', 'Figma'
]

EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
# Regex for common phone number formats
PHONE_REGEX = r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10})'

# --- Core Functions ---

def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    text = ""
    try:
        # Use io.BytesIO to handle the uploaded file buffer
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for page in pdf.pages:
                # Add text from each page
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_info(text):
    """Extracts contact info and skills from the cleaned text."""
    if not nlp:
        return {'Error': 'NLP model not loaded.'}
        
    # Extract Contact Information using Regex
    email = re.findall(EMAIL_REGEX, text)
    # The regex for phone returns many matches; we take the first clean 10+ digit number
    phone_matches = re.findall(PHONE_REGEX, text)
    
    # Basic Name Extraction (using spaCy NER)
    doc = nlp(text)
    # Tries to find the first entity labeled as PERSON
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "N/A")
    
    # Skill Extraction (Keyword Matching)
    found_skills = set()
    text_lower = text.lower() 
    for skill in SKILLS_LIST:
        # Check for the skill as a whole word boundary
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
            found_skills.add(skill)
            
    return {
        'Name': name,
        'Email': email[0] if email else 'N/A',
        'Phone': phone_matches[0] if phone_matches else 'N/A',
        'Extracted Skills': sorted(list(found_skills)) # Sort for better display
    }

# --- Streamlit UI Layout ---

st.set_page_config(page_title="ML Resume Screening Demo", layout="centered")

st.title("🤖 ML Resume Screening Tool Demo")
st.markdown("Upload a resume (PDF) to automatically extract key information and skills.")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a PDF file (Resume)", type="pdf")

if uploaded_file is not None:
    # Use a spinner while processing
    with st.spinner('Parsing and analyzing resume...'):
        
        # 1. Text Extraction
        raw_text = extract_text_from_pdf(uploaded_file)
        
        if raw_text:
            # 2. Information Extraction
            extracted_data = extract_info(raw_text)
            
            # 3. Display Results
            st.success("✅ Analysis Complete!")

            # --- FIX: Handle Pandas ValueError by wrapping scalar values in lists ---
            # Create a new dictionary where every value is wrapped in a list
            # This ensures pandas creates a one-row DataFrame correctly.
            data_for_df = {k: [v] for k, v in extracted_data.items()}
            
            # Create the DataFrame, Transpose (.T) so keys become rows, and rename the value column.
            info_df = pd.DataFrame(data_for_df).T.rename(columns={0: "Value"})
            # ------------------------------------------------------------------------

            # Display extracted info as a table
            st.subheader("Extracted Candidate Profile")
            st.dataframe(info_df, use_container_width=True)
            
            # Simple ML Feature Placeholder (e.g., skill count)
            st.subheader("Recruiter Insights (ML Placeholder)")
            skill_count = len(extracted_data.get('Extracted Skills', []))
            st.info(f"*Total Technical Skills Found:* *{skill_count}*")

            # --- Placeholder for ML Ranking ---
            st.markdown("---")
            st.subheader("⭐ Relevance Ranking (Future ML Feature)")
            st.markdown("In a full ML project, a second model would rank this resume against a Job Description (JD) using vector similarity.")
            
            # Input for JD (Placeholder)
            st.text_area("Paste Job Description (JD) Here:", 
                         "E.g., Seeking a Data Scientist with strong Python, SQL, and Deep Learning experience, proficient in PyTorch/TensorFlow.")
            st.metric(label="Relevance Score (Simulated)", value="85%", delta="High Match")


            # Display Raw Text for verification (optional)
            with st.expander("View Raw Extracted Text"):
                st.text(raw_text)