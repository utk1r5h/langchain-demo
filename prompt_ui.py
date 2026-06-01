from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import streamlit as st

# LLM
model = ChatOllama(
    model="llama3",
    temperature=0
)

st.title("Research Paper Explainer")

# Inputs
paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT",
        "GPT-3: Language Models are Few-Shot Learners"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner",
        "Technical",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "1-2 paragraphs",
        "2-3 paragraphs",
        "3-4 paragraphs"
    ]
)

# Prompt Template
template = PromptTemplate(
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ],
    template="""
You are a research paper expert.

Summarize the research paper titled:

"{paper_input}"

Requirements:

- Explanation Style: {style_input}
- Explanation Length: {length_input}

Include:

1. Main idea of the paper.
2. Key contributions.
3. Important mathematical concepts and equations (if applicable).
4. Simple examples or pseudocode when useful.
5. Relatable analogies to explain difficult concepts.

Important:
- Use your knowledge of the paper.
- Do not say "Insufficient information available".
- If a specific detail is uncertain, simply omit it.
- Keep the explanation accurate and easy to understand.
"""
)

# Chain
chain = template | model

# Button
if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):

        response = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            }
        )

        st.subheader("Summary")
        st.write(response.content)