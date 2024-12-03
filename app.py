import gradio as gr
from groq import Groq
import openai
import json


GROQ_API_KEY = ""
OPENAI_API_KEY = ""


client_groq = Groq(api_key=GROQ_API_KEY)
openai.api_key = OPENAI_API_KEY

def get_responses_and_compare(context, questions_text, ground_truth_text):
    
    if not context or not questions_text or not ground_truth_text:
        return "Please provide valid inputs for context, questions, and ground truth."

    questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
    ground_truth = [gt.strip() for gt in ground_truth_text.splitlines() if gt.strip()]
    
    if len(ground_truth) != len(questions):
        return "Ground truth must match the number of questions."

    
    models = ["llama-3.2-1b-preview", "gpt-4-turbo"]
    
    response_1 = []
    response_2 = []

    for model_index, model_name in enumerate(models):
        for question in questions:
            if model_name == "llama-3.2-1b-preview":
                
                completion = client_groq.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI designed to extract boolean information from text with extreme precision. \n\n"
                                "Rules:\n"
                                "1. Respond ONLY in JSON format\n"
                                "2. Use exactly this structure: {{\"answer\": \"Yes\"}} or {{\"answer\": \"No\"}}\n"
                                "3. Base your answer solely on the literal text of the paragraph\n"
                                "4. Be strict - only respond Yes if the information is explicitly present\n"
                                "5. If there's any doubt, respond with {{\"answer\": \"No\"}}. This is the context: "
                                + context
                            )
                        },
                        {"role": "user", "content": question}
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                )
                response_content = ""
                for chunk in completion:
                    response_content += chunk.choices[0].delta.content or ""

            elif model_name == "gpt-4-turbo":
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an AI designed to extract boolean information from text with extreme precision. \n\n"
                                "Rules:\n"
                                "1. Respond ONLY in JSON format\n"
                                "2. Use exactly this structure: {{\"answer\": \"Yes\"}} or {{\"answer\": \"No\"}}\n"
                                "3. Base your answer solely on the literal text of the paragraph\n"
                                "4. Be strict - only respond Yes if the information is explicitly present\n"
                                "5. If there's any doubt, respond with {{\"answer\": \"No\"}}. This is the context: "
                                + context
                            )
                        },
                        {"role": "user", "content": question}
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                )
                response_content = response.choices[0].message["content"]

            try:
                answer = json.loads(response_content).get("answer", "")
                if model_index == 0:
                    response_1.append(answer)
                else:
                    response_2.append(answer)
            except json.JSONDecodeError as e:
                print(f"Error parsing response: {e}")

    # Calculate accuracy
    accuracy_1 = sum(1 for r, g in zip(response_1, ground_truth) if r == g) / len(ground_truth) if ground_truth else 0
    accuracy_2 = sum(1 for r, g in zip(response_2, ground_truth) if r == g) / len(ground_truth) if ground_truth else 0
    
    comparison_result = f"""
Model 1 (llama-3.2-1b-preview):
Responses: {response_1}
Accuracy: {accuracy_1:.2%}

Model 2 (gpt-4-turbo):
Responses: {response_2}
Accuracy: {accuracy_2:.2%}

Ground Truth: {ground_truth}
"""
    
    return comparison_result

# Create Gradio interface
iface = gr.Interface(
    fn=get_responses_and_compare,
    inputs=[
        gr.Textbox(label="Context", lines=5, placeholder="Enter the context here..."),
        gr.Textbox(label="Questions (one per line)", lines=4, placeholder="Enter questions, one per line..."),
        gr.Textbox(label="Ground Truth (one Yes/No per line)", lines=4, placeholder="Enter ground truth answers, one per line...")
    ],
    outputs=gr.Textbox(label="Comparison Results"),
    title="Groq AI and OpenAI Boolean Information Extractor",
    description="Compare responses between Groq's Llama model and OpenAI's GPT-4."
)


iface.launch()