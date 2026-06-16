import streamlit as st
import torch

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

# Load model
MODEL_PATH = "models"

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_PATH
)

model = (
    DistilBertForSequenceClassification
    .from_pretrained(
        MODEL_PATH
    )
)

model.eval()

# Labels
classes = [
    "Billing inquiry",
    "Cancellation request",
    "Product inquiry",
    "Refund request",
    "Technical issue"
]

# Page title
st.title(
    "Customer Support Ticket Routing"
)

st.write(
    "Enter a support ticket to predict category"
)

# Input
ticket = st.text_area(
    "Ticket Description"
)

# Predict
if st.button(
    "Predict"
):

    if ticket:

        inputs = tokenizer(
            ticket,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():

            output = model(
                **inputs
            )

        pred = (
            torch.argmax(
                output.logits,
                dim=1
            )
            .item()
        )

        st.success(
            f"Predicted Category: {classes[pred]}"
        )

    else:

        st.warning(
            "Please enter ticket text."
        )