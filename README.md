# Hippocratic AI - Agent Deployment Engineer Take Home

Objective: Generate a bedtime story for kids ages 5 - 10 years old and have a judge that contributes to automated 
improvement of the story.

## Instructions (Set Up)
1. pip install -r requirements.txt

2. export OPENAI_API_KEY=“Enter your OpenAI API key here”

3. python3 updated_main.py

## Approach
This project takes inspiration from the Generative Adversarial Network (GAN) paradigm, where a generator produces content and a critic evaluates it. Here, the generator is a large language model (ChatGPT-3.5-turbo) that creates bedtime stories. The judge is a separate LLM that scores each story along the below five dimensions.

The weights for each dimension are assigned as the following:
{
    "age_fit": 0.30,
    "warmth": 0.20,
    "creativity": 0.20,
    "coherence": 0.20,
    "safety": 0.10
}

If the total weighted score falls below 0.80, the system enters a training loop, where the story is regenerated using the original draft, judge feedback, and the user's initial story request. This loop continues until the story meets or exceeds the quality threshold.

<img width="2126" height="738" alt="image" src="https://github.com/user-attachments/assets/edd68e95-71e8-44bc-b07c-3feb0b07f7ce" />

A temperature of 0.9 was used for story generation and improvement. This higher temperature produced more imaginative, expressive stories with stronger character personality and narrative richness.

The bedtime story application is fairly straightforward, and, thus, the judge LLM is designed to be harsh and judgmental because the generator LLM is already good. Future extensions could explore more nuanced or complex narratives that challenge the generator further.

## Evaluation

The evaluation results are shown in the file "Evaluation.csv".

I ran the full generator-judge workflow across 25 user inputs, where the last 5 were deliberately age inappropriate requests. For each request, the CSV records: the initial and final judge scores, a comparison of all five scoring dimensions, and the final story text. A notable example is Row 19, where the original story scored 72%, triggering the training loop. After refinement, the final story achieved a score of 96%. 

## Wishlist

Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

I would make the story generation process more interactive, where the chatbot asks the user questions about how they would like the story to turn out.

Wishlist (for later on):
1. Incorporating a text-to-voice feature so that the user can have the story being read aloud to them. 
2. Giving the characters specific, crafted personalities, by giving a few examples in the prompt.
3. Introducing a human judge to score. 
4. Fine-tuning after gathering enough data.
5. Explore other models, such as Gemini, Claude, and/or Perplexity.
