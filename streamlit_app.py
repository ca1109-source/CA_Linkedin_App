import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="LinkedIn User Prediction",
    layout="wide"
)
# Title and description
st.title("LinkedIn Usage Prediction App")
st.write("Enter demographic information in the sidebar and click the button to predict whether a person uses Linkedin")

# Load and prepare data
@st.cache_data
def load_data():
    # Read the data
    s = pd.read_csv('social_media_usage (1).csv')
    
    def clean_sm(x):
        return np.where(x == 1, 1, 0)
    
    # Create ss dataframe
    ss = pd.DataFrame()
    ss['sm_li'] = clean_sm(s['web1h'])
    ss['income'] = np.where(s['income'] > 9, np.nan, s['income'])
    ss['education'] = np.where(s['educ2'] > 8, np.nan, s['educ2'])
    ss['parent'] = np.where(s['par'] == 1, 1, 0)
    ss['married'] = np.where(s['marital'] == 1, 1, 0)
    ss['female'] = np.where(s['gender'] == 2, 1, 0)
    ss['age'] = np.where(s['age'] > 98, np.nan, s['age'])
    ss = ss.dropna()
    
    return ss

# Train model
@st.cache_resource
def train_model(ss):
    # Create target and features
    y = ss['sm_li']
    X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train, y_train)
    
    return lr

# Load data and train model
ss = load_data()
model = train_model(ss)

# Sidebar for user inputs
st.sidebar.header("Enter User Information")

st.sidebar.subheader("Demographics")

income = st.sidebar.slider(
    "Income Level",
    min_value=1,
    max_value=9,
    value=5,
    help="1 = Less than $10k, 9 = $150k+"
)

education = st.sidebar.slider(
    "Education Level",
    min_value=1,
    max_value=8,
    value=4,
    help="1 = Less than HS, 8 = Postgraduate degree"
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=97,
    value=42
)

st.sidebar.subheader("Personal Information")

parent = st.sidebar.selectbox(
    "Parent Status",
    options=[0, 1],
    format_func=lambda x: "Parent" if x == 1 else "Not a Parent",
    index=0
)

married = st.sidebar.selectbox(
    "Marital Status",
    options=[0, 1],
    format_func=lambda x: "Married" if x == 1 else "Not Married",
    index=0
)

female = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 1 else "Male",
    index=0
)

# Create prediction button
predict_button = st.sidebar.button("Predict LinkedIn Usage", type="primary")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Profile")
    
    # Display user information
    profile_data = {
        "Attribute": ["Income Level", "Education Level", "Age", "Parent Status", "Marital Status", "Gender"],
        "Value": [
            income,
            education,
            age,
            "Parent" if parent == 1 else "Not a Parent",
            "Married" if married == 1 else "Not Married",
            "Female" if female == 1 else "Male"
        ]
    }
    
    st.table(pd.DataFrame(profile_data))

with col2:
    st.subheader("Results")
    
    if predict_button:
        # Create input dataframe
        person = pd.DataFrame({
            'income': [income],
            'education': [education],
            'parent': [parent],
            'married': [married],
            'female': [female],
            'age': [age]
        })
        
        # Make prediction
        prediction = model.predict(person)[0]
        probability = model.predict_proba(person)[0]
        
        # Display results
        if prediction == 1:
            st.success("✅ This person is predicted to USE LinkedIn")
        else:
            st.error("❌ This person is predicted to NOT use LinkedIn")
        
        st.write("---")
        
        # Display probabilities
        st.write("**Probability Breakdown:**")
        
        # Create probability bars
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                label="Does NOT use LinkedIn",
                value=f"{probability[0]:.1%}"
            )
            st.progress(probability[0])
        
        with col_b:
            st.metric(
                label="DOES use LinkedIn",
                value=f"{probability[1]:.1%}"
            )
            st.progress(probability[1])
            st.write("**By Variable:**")
        
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        
        # 1. Income Analysis
        income_range = range(1, 10)
        income_probs = []
        for inc in income_range:
            test = pd.DataFrame({
                'income': [inc],
                'education': [education],
                'parent': [parent],
                'married': [married],
                'female': [female],
                'age': [age]
            })
            prob = model.predict_proba(test)[0][1]
            income_probs.append(prob)
        
        axes[0].plot(income_range, income_probs, marker='o', linewidth=2, color='blue')
        axes[0].axvline(x=income, color='red', linestyle='--', linewidth=2, label='Predicted Income')
        axes[0].axhline(y=probability[1], color='red', linestyle=':', alpha=0.5)
        axes[0].set_xlabel('Income Level')
        axes[0].set_ylabel('LinkedIn Usage Probability')
        axes[0].set_title('Income Impact on LinkedIn Usage')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Education Analysis
        education_range = range(1, 9)
        education_probs = []
        for edu in education_range:
            test = pd.DataFrame({
                'income': [income],
                'education': [edu],
                'parent': [parent],
                'married': [married],
                'female': [female],
                'age': [age]
            })
            prob = model.predict_proba(test)[0][1]
            education_probs.append(prob)
        
        axes[1].plot(education_range, education_probs, marker='o', linewidth=2, color='green')
        axes[1].axvline(x=education, color='red', linestyle='--', linewidth=2, label='Predicted Education')
        axes[1].axhline(y=probability[1], color='red', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Education Level')
        axes[1].set_ylabel('LinkedIn Usage Probability')
        axes[1].set_title('Education Impact on LinkedIn Usage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Age Analysis
        age_range = range(18, 98)
        age_probs = []
        for age_val in age_range:
            test = pd.DataFrame({
                'income': [income],
                'education': [education],
                'parent': [parent],
                'married': [married],
                'female': [female],
                'age': [age_val]
            })
            prob = model.predict_proba(test)[0][1]
            age_probs.append(prob)
        
        axes[2].plot(age_range, age_probs, marker='.', linewidth=2, color='purple')
        axes[2].axvline(x=age, color='red', linestyle='--', linewidth=2, label='Predicted Age')
        axes[2].axhline(y=probability[1], color='red', linestyle=':', alpha=0.5)
        axes[2].set_xlabel('Age')
        axes[2].set_ylabel('LinkedIn Usage Probability')
        axes[2].set_title('Age Impact on LinkedIn Usage')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("<-  Enter user information and click 'Predict LinkedIn Usage' to see results")

# Add information section at the bottom
st.write("---")
st.subheader("About This Model")

with st.expander("How does this model work?"):
    st.write("""
    This prediction model uses **Logistic Regression** trained on survey data to predict LinkedIn usage.
    
    **Features used:**
    - Income (1-9 scale)
    - Education (1-8 scale)
    - Age
    - Parent status
    - Marital status
    - Gender
    
    **Model Performance:**
    - Accuracy: 65.87%
    - Precision: 51.94%
    - Recall: 73.63%
    - F1 Score: 60.91%
    """)

with st.expander("What do the scales mean?"):
    st.write("""
    **Income Scale (1-9):**
    - 1: Less than $10,000
    - 2: $10,000 to under $20,000
    - 3: $20,000 to under $30,000
    - 4: $30,000 to under $40,000
    - 5: $40,000 to under $50,000
    - 6: $50,000 to under $75,000
    - 7: $75,000 to under $100,000
    - 8: $100,000 to under $150,000
    - 9: $150,000 or more
    
    **Education Scale (1-8):**
    - 1: Less than high school
    - 2: High school incomplete
    - 3: High school graduate
    - 4: Some college, no degree
    - 5: Two-year associate degree
    - 6: Four-year bachelor's degree
    - 7: Some postgraduate school
    - 8: Postgraduate or professional degree
    """)

# Footer
st.write("---")
st.caption("Created by Clifford Akins | Georgetown MSBA")