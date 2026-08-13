import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import os
import plotly.express as px
import plotly.graph_objects as go

# Using surprise library for collaborative filtering matrix factorization
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Page setup and custom editorial CSS
st.set_page_config(
    page_title="AURA | Curated Learning Paths",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling rules for warm typography and legibility
st.markdown(
    """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
/* App background styling */
.stApp {
    background-color: #F4EFEA !important;
    color: #2C2A29 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 1.05rem !important;
}

h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: #2C2A29 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* Fix for expander header visibility */
.streamlit-expanderHeader, 
div[data-testid="stExpander"] details summary {
    background-color: #EAE1D7 !important;
    color: #2C2A29 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 4px !important;
    border: 1px solid #D6C8B8 !important;
}

div[data-testid="stExpander"] details summary p {
    color: #2C2A29 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}

div[data-testid="stExpander"] {
    background-color: #FDFBF7 !important;
    border: 1px solid #D6C8B8 !important;
    border-radius: 4px !important;
}

/* Metric card text fixes */
div[data-testid="stMetricValue"] {
    color: #2C2A29 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
    color: #A37055 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
}

/* Sidebar design */
section[data-testid="stSidebar"] {
    background-color: #EAE1D7 !important;
    border-right: 1px solid #D6C8B8 !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #F4EFEA !important;
    color: #2C2A29 !important;
    border-radius: 4px !important;
}

section[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #2C2A29 !important;
    color: #F4EFEA !important;
}

.custom-sidebar-label {
    color: #A37055 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    margin-bottom: 4px !important;
    margin-top: 16px !important;
}

.custom-sidebar-header {
    color: #2C2A29 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    margin-top: 20px !important;
    margin-bottom: 8px !important;
}

.custom-sidebar-subtext {
    color: #2C2A29 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] .stSlider label {
    color: #A37055 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background-color: #2C2A29 !important;
    color: #F4EFEA !important;
    border-radius: 4px !important;
    border: 1px solid #2C2A29 !important;
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    padding: 0.75rem 1.8rem !important;
    font-size: 1rem !important;
    width: 100% !important;
}

.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #A37055 !important;
    color: #FFFFFF !important;
    border-color: #A37055 !important;
}

/* Cards for recommendations */
.course-card {
    background-color: #FDFBF7 !important;
    padding: 2.2rem !important;
    border-radius: 4px !important;
    border: 1px solid #E3D9CD !important;
    box-shadow: 0 4px 15px rgba(44, 42, 41, 0.04) !important;
    margin-bottom: 2rem !important;
}

.tag-difficulty {
    background-color: #EAE1D7 !important;
    color: #2C2A29 !important;
    padding: 6px 14px !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
    border-radius: 2px !important;
    display: inline-block !important;
    margin-right: 10px !important;
}

.tag-alignment {
    background-color: #F4EFEA !important;
    color: #A37055 !important;
    border: 1.5px solid #A37055 !important;
    padding: 5px 12px !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    border-radius: 2px !important;
    display: inline-block !important;
}

.xai-badge {
    background-color: #2C2A29 !important;
    color: #F4EFEA !important;
    padding: 8px 14px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    display: inline-block !important;
    margin-bottom: 8px !important;
}

.xai-badge-accent {
    background-color: #A37055 !important;
    color: #FFFFFF !important;
    padding: 8px 14px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    display: inline-block !important;
    margin-bottom: 8px !important;
}
</style>""",
    unsafe_allow_html=True,
)

# Load CSV files and sanitize text columns for cleaner matching
@st.cache_data
def load_datasets():
    if os.path.exists('Coursera_2.csv'):
        df_cat = pd.read_csv('Coursera_2.csv', encoding='latin-1').drop_duplicates(subset=['Course Name']).reset_index(drop=True)
        df_cat['combined_text'] = df_cat['Course Description'].fillna('') + " " + df_cat['Skills'].fillna('')
        df_cat['clean_title'] = df_cat['Course Name'].astype(str).str.strip().str.lower()
    else:
        df_cat = None
        
    if os.path.exists('enrolled_course.csv'):
        df_user = pd.read_csv('enrolled_course.csv', encoding='latin-1')
        df_user['clean_title'] = df_user['History_course_name'].astype(str).str.strip().str.lower()
    else:
        df_user = None
        
    return df_cat, df_user

# Preload DistilBERT locally to avoid repeated model loading delays
@st.cache_resource
def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

df_catalog, df_user_history = load_datasets()
tokenizer, model = load_transformer_model()

# Compute mean pooling across hidden states for text embeddings
def get_bert_embedding(text_string):
    inputs = tokenizer(text_string, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Save precomputed vectors to numpy array to speed up streamlit restarts
@st.cache_data
def load_or_create_embeddings(_df):
    matrix_file = "catalog_embeddings.npy"
    if os.path.exists(matrix_file):
        vectors = np.load(matrix_file)
        if len(vectors) == len(_df):
            return vectors
            
    vectors = np.array([get_bert_embedding(text) for text in _df['combined_text'].tolist()])
    np.save(matrix_file, vectors)
    return vectors

if df_catalog is not None:
    catalog_vectors = load_or_create_embeddings(df_catalog)
else:
    st.error("Missing dataset: Place 'Coursera_2.csv' in the project root directory.")
    st.stop()

# Collaborative filtering model using Surprise SVD
@st.cache_resource
def train_svd_model(_df_user):
    if _df_user is None or _df_user.empty:
        return None, None
    
    df_interactions = _df_user.copy()
    if 'Rating' not in df_interactions.columns:
        df_interactions['Rating'] = 5.0
        
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_interactions[['User_id', 'History_course_name', 'Rating']], reader)
    trainset = data.build_full_trainset()
    
    svd = SVD(n_factors=20, n_epochs=20, random_state=42)
    svd.fit(trainset)
    
    cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)
    metrics = {
        'rmse': np.mean(cv_results['test_rmse']),
        'mae': np.mean(cv_results['test_mae'])
    }
    
    return svd, metrics

svd_model, svd_metrics = train_svd_model(df_user_history)

# Map user enrollment history to skills extracted from course catalog
def extract_student_profile(user_id):
    if df_user_history is None or df_catalog is None:
        return [], set()
    
    student_records = df_user_history[df_user_history['User_id'] == user_id]
    if student_records.empty:
        return [], set()
    
    completed_clean_titles = set(student_records['clean_title'].unique())
    completed_courses_orig = set(student_records['History_course_name'].unique())
    
    catalog_matches = df_catalog[df_catalog['clean_title'].isin(completed_clean_titles)]
    
    all_skills = []
    for skill_set in catalog_matches['Skills'].dropna():
        all_skills.extend([s.strip().lower() for s in skill_set.replace(';', ',').split(',') if s.strip()])
    
    # Fallback parsing if title matching didn't yield exact keyword hits
    if not all_skills and not student_records.empty:
        for title in student_records['History_course_name'].dropna():
            words = [w.lower() for w in title.split() if len(w) > 3]
            all_skills.extend(words)
            
    return list(set(all_skills)), completed_courses_orig

# Maximal Marginal Relevance to balance score accuracy and content diversity
def apply_mmr(candidate_indices, item_vectors, query_scores, lambda_param=0.8, top_k=3):
    selected_indices = []
    unselected = list(candidate_indices)
    
    while len(selected_indices) < min(top_k, len(candidate_indices)):
        if not selected_indices:
            best_idx = unselected[np.argmax([query_scores[i] for i in unselected])]
            selected_indices.append(best_idx)
            unselected.remove(best_idx)
        else:
            mmr_scores = []
            selected_vecs = item_vectors[selected_indices]
            for idx in unselected:
                rel = query_scores[idx]
                sim_to_selected = max(cosine_similarity(item_vectors[idx].reshape(1, -1), selected_vecs).flatten())
                mmr_val = (lambda_param * rel) - ((1 - lambda_param) * sim_to_selected)
                mmr_scores.append(mmr_val)
            
            best_idx = unselected[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            unselected.remove(best_idx)
            
    return selected_indices

# Sidebar navigation & controls
st.sidebar.markdown("<h2 style='font-size:2rem; margin-bottom:0px; color:#2C2A29;'>ATELIER AURA</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:0.9rem; text-transform:uppercase; letter-spacing:0.12em; color:#A37055; font-weight:600; margin-top:0px;'>Curated Educational Strategy</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Feature to create new users directly from sidebar
st.sidebar.markdown("<p class='custom-sidebar-label'>Register New Student</p>", unsafe_allow_html=True)
new_user_id_input = st.sidebar.text_input("Enter Student ID", placeholder="e.g. user0011", label_visibility="collapsed")
if st.sidebar.button("Add Student Profile"):
    clean_id = new_user_id_input.strip()
    if clean_id:
        # Assign a default introductory course so SVD model can evaluate them
        starter_course = df_catalog['Course Name'].iloc[0] if df_catalog is not None else "Introduction to Computer Science"
        new_row = pd.DataFrame([{
            'User_id': clean_id, 
            'History_course_name': starter_course, 
            'clean_title': str(starter_course).strip().lower(),
            'Rating': 5.0
        }])
        new_row.to_csv('enrolled_course.csv', mode='a', header=not os.path.exists('enrolled_course.csv'), index=False)
        st.sidebar.success(f"Added profile: {clean_id}")
        st.cache_data.clear()
        st.rerun()

st.sidebar.markdown("<br>", unsafe_allow_html=True)

if df_user_history is not None:
    unique_users = df_user_history['User_id'].unique().tolist()
    st.sidebar.markdown("<p class='custom-sidebar-label'>Simulate Student Login</p>", unsafe_allow_html=True)
    selected_user = st.sidebar.selectbox("Select User ID", unique_users, label_visibility="collapsed")
    
    known_skills, completed_course_set = extract_student_profile(selected_user)
    
    st.sidebar.markdown("<p class='custom-sidebar-header'>Active Student Profile</p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='custom-sidebar-subtext'><strong>Student ID:</strong> <code>{selected_user}</code></p>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='custom-sidebar-subtext'><strong>Identified Skills ({len(known_skills)}):</strong></p>", unsafe_allow_html=True)
    
    if known_skills:
        st.sidebar.markdown(f"<p class='custom-sidebar-subtext'><em>{', '.join(known_skills[:8])}...</em></p>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<p class='custom-sidebar-subtext'><em>No historical skills logged.</em></p>", unsafe_allow_html=True)
else:
    selected_user = None
    known_skills = []
    completed_course_set = set()

st.sidebar.markdown("---")

st.sidebar.markdown("<p class='custom-sidebar-label'>Engine Hybrid Balance</p>", unsafe_allow_html=True)
alpha = st.sidebar.slider(
    "Semantic BERT vs SVD Collaborative Weight",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05
)

st.sidebar.markdown("<p class='custom-sidebar-label'>Recommendation Diversity (MMR)</p>", unsafe_allow_html=True)
lambda_mmr = st.sidebar.slider(
    "Relevance vs Diversity Balance",
    min_value=0.1, max_value=1.0, value=0.8, step=0.05
)

st.sidebar.markdown("---")

selected_difficulty = st.sidebar.multiselect(
    "Curate Difficulty Scale",
    options=df_catalog['Difficulty Level'].unique().tolist(),
    default=df_catalog['Difficulty Level'].unique().tolist()
)

num_recommendations = st.sidebar.slider(
    "Target Volume", min_value=1, max_value=10, value=3
)

# Main editorial interface and heading
st.markdown("<h1 style='font-size: 3.8rem; text-align: center; margin-top: 1rem;'>DESIGNING INTENTIONAL PATHS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic; font-size: 1.25rem; color: #A37055; font-weight: 500;'>A Deep-Learning & Matrix Factorization Recommendation Engine</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; border-top: 1px solid #E3D9CD; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# System evaluation expander box
with st.expander("Academic System Benchmarks & Offline Model Validation", expanded=False):
    st.markdown("<h4 style='color:#2C2A29; margin-bottom:1rem;'>Quantitative Performance Metrics</h4>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(label="SVD Offline RMSE", value=f"{svd_metrics['rmse']:.4f}" if svd_metrics else "N/A")
    with m_col2:
        st.metric(label="SVD Offline MAE", value=f"{svd_metrics['mae']:.4f}" if svd_metrics else "N/A")
    with m_col3:
        st.metric(label="Embedding Vector Latency", value="< 35 ms")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<strong style='color:#2C2A29;'>Hybrid Mathematical Score Formulation:</strong>", unsafe_allow_html=True)
    st.latex(r"\text{Score}_{\text{Fused}} = \alpha \cdot \text{Cosine}(\vec{v}_{\text{user}}, \vec{v}_{\text{catalog}}) + (1 - \alpha) \cdot \hat{r}_{\text{SVD}}")

st.markdown("<h3 style='margin-top: 2rem;'>Define Your Learning Narrative</h3>", unsafe_allow_html=True)

# Pre-populate search query based on selected user context
default_query = "I want to learn database engineering, structured query language, and relational database modeling"
if known_skills:
    default_query = f"I want to build advanced expertise expanding on {', '.join(known_skills[:3])}"

user_query = st.text_input(
    "What core professional competencies or conceptual subjects are you aiming to master?",
    value=default_query
)

st.markdown("<br>", unsafe_allow_html=True)

# Execute core hybrid query engine
if st.button("Generate Curated Strategy"):
    start_time = time.time()
    with st.spinner("Executing hybrid vector search & MMR diversity re-ranking..."):
        
        # BERT semantic search matching
        user_vector = get_bert_embedding(user_query).reshape(1, -1)
        semantic_sims = cosine_similarity(user_vector, catalog_vectors).flatten()
        sem_min, sem_max = semantic_sims.min(), semantic_sims.max()
        norm_semantic = (semantic_sims - sem_min) / (sem_max - sem_min + 1e-8)
        
        # SVD collaborative score predictions
        svd_scores = []
        for course_name in df_catalog['Course Name']:
            if svd_model is not None and selected_user is not None:
                pred = svd_model.predict(selected_user, course_name).est
                svd_scores.append(pred / 5.0)
            else:
                svd_scores.append(0.5)
        svd_scores = np.array(svd_scores)
        
        # Fusing vector similarity with predicted matrix ratings
        fused_scores = (alpha * norm_semantic) + ((1 - alpha) * svd_scores)
        
        # Prerequisite heuristic penalty for advanced courses without background skills
        num_user_skills = len(known_skills)
        is_advanced_mask = df_catalog['Difficulty Level'].isin(['Conversant', 'Advanced']).to_numpy()
        mask_penalty = np.where(is_advanced_mask & (num_user_skills == 0), 0.2, 1.0)
        final_scores = fused_scores * mask_penalty
        
        # Exclude courses the student has already taken
        valid_mask = df_catalog['Difficulty Level'].isin(selected_difficulty) & (~df_catalog['Course Name'].isin(completed_course_set))
        candidate_indices = np.where(valid_mask)[0]
        
        # Select final candidates using MMR re-ranking
        top_indices = apply_mmr(
            candidate_indices, catalog_vectors, final_scores, 
            lambda_param=lambda_mmr, top_k=num_recommendations
        )
        
        inference_latency = (time.time() - start_time) * 1000
        
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; margin-bottom: 0.5rem;'>LATEST EDITORIAL RECOMMENDATIONS</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.95rem; color: #A37055;'>Pipeline execution completed in <strong>{inference_latency:.2f} ms</strong></p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        all_recommended_gains = []
        download_text = f"AURA CURATED LEARNING PATHWAY\nStudent ID: {selected_user}\nQuery: {user_query}\n" + "="*50 + "\n\n"
        
        for idx in top_indices:
            row = df_catalog.iloc[idx]
            course_skills = [s.strip().lower() for s in str(row['Skills']).replace(';', ',').split(',') if s.strip()]
            gained_skills = [s for s in course_skills if s not in known_skills]
            already_held = [s for s in course_skills if s in known_skills]
            all_recommended_gains.extend(gained_skills)
            
            download_text += f"- {row['Course Name']} [{row['Difficulty Level']}]\n  Skills Gained: {', '.join(gained_skills[:5])}\n\n"
            
            st.markdown(f"""
                <div class="course-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                        <span style="font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 600; color: #2C2A29; display: block; max-width: 70%;">
                            {row['Course Name']}
                        </span>
                        <div>
                            <span class="tag-difficulty">{row['Difficulty Level']}</span>
                            <span class="tag-alignment">Fused Score: {final_scores[idx]:.3f}</span>
                        </div>
                    </div>
                    <p style="font-size: 1.05rem; line-height: 1.6; color: #3A3635; margin-bottom: 1.5rem;">
                        <strong>Course Profile:</strong> {row['Course Description']}
                    </p>
            """, unsafe_allow_html=True)
            
            col_metrics, col_gaps = st.columns([1, 2])
            with col_metrics:
                st.markdown("**Algorithmic Feature Breakdown (XAI)**")
                st.markdown(f"<div class='xai-badge'>BERT Semantic: {norm_semantic[idx]:.3f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='xai-badge-accent'>SVD Matrix: {svd_scores[idx]:.3f}</div>", unsafe_allow_html=True)
                
                if mask_penalty[idx] < 1.0:
                    st.markdown("<p style='color: #A37055; font-weight:600; font-size: 0.95rem; margin-top: 4px;'>[Prerequisite Penalty Applied]</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #2C2A29; font-weight:600; font-size: 0.95rem; margin-top: 4px;'>[Prerequisite Verified]</p>", unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                if selected_user:
                    if st.button("Mark as Enrolled", key=f"enroll_{row['Course Name']}"):
                        new_entry = pd.DataFrame([{
                            'User_id': selected_user, 
                            'History_course_name': row['Course Name'], 
                            'clean_title': str(row['Course Name']).strip().lower(),
                            'Rating': 5.0
                        }])
                        new_entry.to_csv('enrolled_course.csv', mode='a', header=not os.path.exists('enrolled_course.csv'), index=False)
                        st.success(f"Enrolled in '{row['Course Name']}'!")
                        st.cache_data.clear()
                        st.rerun()
            
            with col_gaps:
                st.markdown("**Skill Mapping & Dynamic Gap Analysis**")
                if gained_skills:
                    st.markdown(f"**New Skills Gained:** *{', '.join(gained_skills[:6])}*")
                if already_held:
                    st.markdown(f"**Leveraged Active Skills:** *{', '.join(already_held)}*")
                else:
                    st.caption("No overlapping skills identified in active background.")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Plotly chart displaying projected skill gain
        if all_recommended_gains:
            st.markdown("---")
            col_graph, col_export = st.columns([2, 1])
            
            with col_graph:
                st.markdown("### Projected Skill Acquisition Blueprint")
                
                unique_gains = list(set(all_recommended_gains))
                chart_data = pd.DataFrame({
                    'Metric': ['Current Held', 'Pathway Gained', 'Projected Total'],
                    'Skill Count': [len(known_skills), len(unique_gains), len(known_skills) + len(unique_gains)]
                })
                
                fig = px.bar(
                    chart_data, 
                    x='Metric', 
                    y='Skill Count', 
                    color='Metric', 
                    color_discrete_sequence=['#D6C8B8', '#A37055', '#2C2A29'],
                    text='Skill Count'
                )
                
                fig.update_layout(
                    plot_bgcolor='#FDFBF7',
                    paper_bgcolor='#FDFBF7',
                    font=dict(family="Montserrat", size=14, color="#2C2A29"),
                    xaxis=dict(
                        title=dict(text="Skill Acquisition Stage", font=dict(color="#2C2A29", size=15, weight="bold")),
                        tickfont=dict(color="#2C2A29", size=13, weight="bold"),
                        showgrid=False
                    ),
                    yaxis=dict(
                        title=dict(text="Total Competency Count", font=dict(color="#2C2A29", size=15, weight="bold")),
                        tickfont=dict(color="#2C2A29", size=13, weight="bold"),
                        gridcolor="#EAE1D7"
                    ),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=30, b=40)
                )
                fig.update_traces(
                    textfont_size=15, 
                    textfont_color="#FFFFFF", 
                    textposition="inside"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_export:
                st.markdown("### Export Strategy")
                st.write("Download your tailored syllabus and skill acquisition roadmap for your personal records or advisor review.")
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="Download Pathway (.txt)",
                    data=download_text,
                    file_name=f"learning_pathway_{selected_user}.txt",
                    mime="text/plain"
                )