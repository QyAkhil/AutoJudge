import streamlit as st
import pickle
from preprocessing import preprocess_for_model
# Caching resources so that it does not need to be loaded again and again
@st.cache_resource
def load_objects():
    with open("web_ui/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("web_ui/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("web_ui/mlb_tags.pkl", "rb") as f:
        mlb = pickle.load(f)
    with open("web_ui/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("best_models/clf_model.pkl", "rb") as f:
        model_clf = pickle.load(f)
    with open("best_models/reg_model.pkl", "rb") as f:
        model_reg = pickle.load(f)
    return tfidf, scaler, mlb, model_clf,model_reg,le 
# Load trained models and preprocessing objects
tfidf, scaler, mlb, model_clf,model_reg,le = load_objects()

all_tags=['math', 'number theory', 'constructive algorithms',
       'data structures', 'greedy', 'sortings', 'brute force',
       'dfs and similar', 'dp', 'graphs', 'trees', 'binary search',
       'geometry', 'implementation', 'ternary search', 'two pointers',
       'combinatorics', 'dsu', 'hashing', 'games', 'probabilities',
       'bitmasks', 'shortest paths', 'divide and conquer', 'interactive',
       'strings', 'chinese remainder theorem', 'fft', 'flows',
       'string suffix structures', 'matrices', 'schedules', '*special',
       'graph matchings', '2-sat', 'meet-in-the-middle',
       'expression parsing']
st.set_page_config(
    page_title="CP Difficulty Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CP Problem Difficulty Predictor",text_alignment='center')
st.markdown(
    "<p style='text-align:center; font-size:18px; color:gray;'>"
    "ML-based difficulty estimation trained on 10,000+ Codeforces problems"
    "</p>",
    unsafe_allow_html=True
)

st.subheader("📌 Problem Features")
cols1,cols2=st.columns(2)

with cols1:
    title=st.text_input('Title')
    description=st.text_area('Description',height=110)
    Input_format=st.text_input('Input format')
    Output_format=st.text_input('Output format')
with cols2:
    time_limit=st.number_input('Time limit  in seconds')    
    mem_limit=st.number_input('Memory limit in MB eg: 256,512 etc')
    tags=st.pills("Select tags",options=all_tags,selection_mode='multi')    
if st.button("Predict Difficulty"):
    if not title or not description or not Input_format or not Output_format :
        st.warning("Please fill all text fields.")
    else:
        with st.spinner("Analyzing problem difficulty..."):
            X_final=preprocess_for_model(title, description,Input_format, Output_format,tags, time_limit,mem_limit,tfidf, scaler, mlb)
            class_pred=model_clf.predict(X_final)
            rating_pred=model_reg.predict(X_final)
        st.header("Prediction Result")
        pred_label = le.inverse_transform(class_pred)[0]
        st.success(f"Predicted Difficulty: {pred_label}")
        st.success(f"Predicted Rating: {int(rating_pred[0])}")

with st.expander("See Example  format"):
    st.write("📌**Title** : Bear and Prime 100")
    st.write('''📌**Description**:  This is an interactive problem. In the output section below you will see the information about 
             flushing the output.\n\nBear Limak thinks of some hidden number — an integer from interval [2, 100]. 
             Your task is to say if the hidden number is prime or composite.\n\nInteger x > 1 is called prime if it 
             has exactly two distinct divisors, 1 and x. If integer x > 1 is not prime, it's called composite.\n\n
             You can ask up to 20 queries about divisors of the hidden number.
              In each query you should print an integer from interval [2, 100].......''')
    st.write('''📌**Input format**: After each query you should read one string from the input. It will be "yes" if the printed integer is a divisor of the hidden number, and "no" otherwise.''')
    st.write('''📌**Output format**: Up to 20 times you can ask a query — print an integer from interval [2, 100] in one line." \
    " You have to both print the end-of-line character and flush the output. 
             After flushing you should read a response from the input.........''')
    st.write("📌**Time limit**: 1 sec")
    st.write("📌**Memory limit**: 512 MB")
    st.write("📌**Tags**: combinatorics, dp, math.")
