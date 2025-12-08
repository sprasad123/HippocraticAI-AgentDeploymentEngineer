import os
import openai
from openai import OpenAI
from pydantic import BaseModel, Field
import json
import pandas as pd

"""
Before submitting the assignment, describe here in a few sentences what you would have 
built next if you spent 2 more hours on this project:

I would make the story generation process more interactive, where the chatbot asks the 
user questions about how they would like the story to turn out.

Wishlist (for later on):
1. Incorporating a text to voice feature so that the user can have the story being read 
aloud to them. 
2. Giving the characters specific, crafted personalities, by giving a few examples in
the prompt.
3. Introducing a human judge to score. 
4. Fine-tuning after gathering enough data.
5. Explore other models, such as Gemini, Claude, Perplexity
"""

# Configuration
TEMPERATURE = 0.9
MAX_TOKENS  = 800

# Reward weights for calculating score
REWARD_WEIGHTS = {
    "age_fit": 0.30,
    "warmth": 0.20,
    "creativity": 0.20,
    "coherence": 0.20,
    "safety": 0.10
}

# Judge scoring criteria from Pydantic
class JudgeScores(BaseModel):
    age_fit: float = Field(ge=0, le=1, description="Appropriate for ages 5-10")
    warmth: float = Field(ge=0, le=1, description="Comforting bedtime tone")
    creativity: float = Field(ge=0, le=1, description="Imaginative, engaging")
    coherence: float = Field(ge=0, le=1, description="Clear plot, easy to follow")
    safety: float = Field(ge=0, le=1, description="No scary/violent/sexual content")
    notes: str

client = OpenAI()

# Inferencing GPT-3.5-Turbo model based on prompt
def call_model(prompt: str, max_tokens=3000, temperature=0.1) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# Generate a story based on user input
def generate_story(user_input):
    prompt = f"""
    You are a storyteller to kids of ages 5 - 10 years old. Based on the user input, you 
    are going to create a story for kids. Based on the user input, you will: 1. Pick a mode 
    that is appropriate, such as informing, educating, entertaining, enlightening, or a 
    combination of the above modes. 2. Factor in the developmental background of the 
    user. 3. Think of character(s) and their personalities. 4. Create a story arc that involves 
    a journey for the characters going through some darkness to the light. The darkness can 
    range from being mild to medium intensity. There has to be something relatable to a kid 
    who is between the ages of 5 - 10 years old. 5. Have a lesson, moral, or learning from 
    the story. 6. Generate the bedtime story.

    The story MUST be appropriate for a G-rated audience. There MUST NOT be any violence, 
    slang, insults, romance, sex, nor abuse. If the user input is an inappropriate request, 
    tell the user "Please ask for something G-rated."

    User Input: {user_input}

    The story MUST be between 300 - 500 words. Do NOT exceed 500 words.
    """
    response = call_model(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    return response

# Judge scores the story from 0 to 1 based on specific criteria
def judge_story(story):
    prompt = f"""
    You are judge. You are going to judge on the following criteria: age fit, warmth,
    creativity, coherence, and safety. You can include any important notes. 

    Age fit refers to the story being appropriate for kids ages 5-10 years old. It
    should be G rated and not have any violence, slang, insults, romance, sex, and abuse.
    You must score it on a scale of 0 - 1 in decimal format with up to 4 decimal points.
    0 = Extremely inapropriate for ages 5-10 years old, with R rated material.
    0.5 = Somewhat inappropriate for ages 5-10 years old, with PG-13 rated material.
    1 = Extremely appropriate for ages 5-10 years old, with G-rated material.

    Warmth measures how emotionally comforting, gentle, and nurturing the story feels to 
    a young reader. It captures the extent to which the tone of the writing promotes a 
    sense of safety, kindness, and emotional connection. You must score it on a scale of
    0 - 1 in decimal format with up to 4 decimal points. 0 = Cold, unfriendly, emotionally 
    sharp or distant tone. 0.5 = Neutral warmth; not harsh but not particularly comforting.
    1 = Extremely warm, gentle, bedtime-appropriate. 

    Creativity measures how imaginative, original, and engaging the story is for a young 
    reader. It captures the richness of ideas, the novelty of characters or settings, 
    and the story’s ability to spark curiosity and wonder. A creative story feels magical, 
    surprising, and memorable—something that expands a child’s imagination. 
    You must score it on a scale of 0 - 1 in decimal format with up to 4 decimal points.
    0 = Very plain or generic; little imaginative content. 0.5 = Moderately creative; some 
    interesting ideas but not very rich. 1 = Highly imaginative, original, delightful.

    Coherence measures how logically organized, clear, and easy to follow the story is for 
    a young reader. It evaluates whether the plot, events, and character actions make sense 
    together and unfold in a smooth, understandable way. A coherent story feels connected, 
    structured, and readable for children ages 5–10. You must score it on a scale of 0 - 1 
    in decimal format with up to 4 decimal points. 0 = Hard to follow, disorganized, or 
    inconsistent. 0.5 = Partially clear; some confusing moments. 1 = Very clear and easy to 
    follow; strong narrative flow. 

    Safety measures whether the story’s content is developmentally appropriate and free 
    from harmful, frightening, or adult themes. It evaluates whether the story maintains 
    a gentle, age-appropriate emotional atmosphere for children ages 5–10. A safe story 
    avoids anything that could cause fear, distress, confusion, or be inappropriate for 
    young readers. You must score it on a scale of 0 - 1 in decimal format with up to 4 
    decimal points. 0 = Clearly unsafe for children ages 5–10. 0.5 = Partially safe; 
    contains mildly concerning elements. 1 = Completely safe; gentle, age-appropriate.

    Be very critical and judgmental with your scoring. Do NOT give a score of 1.0 in 
    every criteria. You are a literay connoisseur and commenter. You are not pleased
    too easily.

    Write the scores in the following format: {{
        "age_fit": float,
        "warmth": float,
        "creativity": float,
        "coherence": float,
        "safety": float,
        "notes": string
    }}

    Do not include any explanation or text outside the JSON.
    
    Here is the story to evaluate:
    
    \"\"\"{story}\"\"\"
    """
    response = call_model(prompt, temperature=0.5)
    return response

# Returns weighted sum of scores between 0 and 1
def weighted_reward(scores: JudgeScores) -> float:
    return sum(getattr(scores, k) * w for k, w in REWARD_WEIGHTS.items())

# Improves the story based on previous story draft, feedback from judge, and user input
def training(story_draft, scores, user_input):
    prompt = f"""
    You are a storyteller to kids of ages 5 - 10 years old. Based on the user input, you 
    are going to create a story for kids. Based on the user input, you will: 1. Pick a mode 
    that is appropriate, such as informing, educating, entertaining, enlightening, or a 
    combination of the above modes. 2. Factor in the developmental background of the 
    user. 3. Think of character(s) and their personalities. 4. Create a story arc that involves 
    a journey for the characters going through some darkness to the light. The darkness can 
    range from being mild to medium intensity. There has to be something relatable to a kid 
    who is between the ages of 5 - 10 years old. 5. Have a lesson, moral, or learning from 
    the story. 6. Generate the bedtime story.

    Story: {story_draft}

    Feedback: {scores}

    If the story_draft is very inappropriate, then consider creating a new story
    based on the user's input.

    User input: {user_input}

    The story MUST be between 300 - 500 words. Do NOT exceed 500 words.
    """
    while True:
        if weighted_reward(scores) >= 0.8:
            break
        story_draft = call_model(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        judge_score = judge_story(story_draft)
        scores = JudgeScores(**json.loads(judge_score))
    return story_draft, scores


def main():
    # User input for story
    user_input = input("What kind of story do you want to hear? ")
    # Generate a story based on user input
    story = generate_story(user_input)
    # If user request is inappropriate, prompt user again to enter their story idea
    if "Please ask for something G-rated" in story:
        while True:
            user_input = input("Please ask for something G-rated. What kind of story do you want to hear? ")
            story = generate_story(user_input)
            if "Please ask for something G-rated" not in story:
                break
    # Judge scores the story
    judge_score = judge_story(story)
    # Convert the scores into Pydantic format
    scores = JudgeScores(**json.loads(judge_score))
    # Calculate the total score based on different criteria
    total_score = weighted_reward(scores)
    # If the total score surpasses the threshold then print the store
    if total_score >= 0.8:
        print(story)
    # If the total score is less than the threshold then go through the training loop 
    # until the reward is above the threshold
    else:
        print("Entered training loop")
        story_updated, scores = training(story, scores, user_input)
        print("After training")
        print(story_updated)

INPUT_PATH = "Input.xlsx"
EVAL_PATH = "Evaluation.csv"

def log_evaluation(
    story_request: str,
    original_scores: JudgeScores,
    final_scores: JudgeScores,
    original_total: float,
    final_total: float,
    final_story: str,
    path: str = EVAL_PATH
):
    row = {
        "Story Request": story_request,
        "Age Fit Original": original_scores.age_fit,
        "Warmth Original": original_scores.warmth,
        "Creativity Original": original_scores.creativity,
        "Coherence Original": original_scores.coherence,
        "Safety Original": original_scores.safety,
        "Notes Original": original_scores.notes,
        "Score Original": original_total,
        "Age Fit Final": final_scores.age_fit,
        "Warmth Final": final_scores.warmth,
        "Creativity Final": final_scores.creativity,
        "Coherence Final": final_scores.coherence,
        "Safety Final": final_scores.safety,
        "Notes Final": final_scores.notes,
        "Score Final": final_total,
        "Final Story": final_story
    }

    try:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        # If file doesn't exist yet, create it with this single row
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)


# Runs the full story generation. Logs original and final scores to CSV
def run_story_session(user_input: str):
    # Generate initial story from the request
    story = generate_story(user_input)

    # If the model responds with the G-rated warning, skip this request
    if "Please ask for something G-rated" in story:
        print(f"Skipping inappropriate request: '{user_input}'")

        original_scores = JudgeScores(
            age_fit=0.0,
            warmth=0.0,
            creativity=0.0,
            coherence=0.0,
            safety=0.0,
            notes="Invalid request – inappropriate for G rating."
        )
        final_scores = original_scores
        original_total = 0.0
        final_total = 0.0
        final_story = ""

        # Log a row with zero scores
        log_evaluation(
            story_request=user_input,
            original_scores=original_scores,
            final_scores=final_scores,
            original_total=original_total,
            final_total=final_total,
            final_story=final_story,
        )

        # No story to return in this case
        return None, original_scores, final_scores

    # Judge the original story
    judge_score = judge_story(story)  # JSON string from LLM
    original_scores = JudgeScores(**json.loads(judge_score))
    original_total = weighted_reward(original_scores)

    # Decide whether to train/improve
    if original_total >= 0.8:
        # Story is good enough
        final_story = story
        final_scores = original_scores
        final_total = original_total

        print(f"Story for '{user_input}' meets threshold (score={final_total:.3f}).")
        print(final_story)
    else:
        # Story needs improvement, run training loop
        print(f"Training story for '{user_input}' (original score={original_total:.3f})...")
        final_story, final_scores = training(story, original_scores, user_input)
        final_total = weighted_reward(final_scores)

        print(f"Final score for '{user_input}' after training: {final_total:.3f}")
        print(final_story)

    # Log original vs final scores to CSV (now including final_story)
    log_evaluation(
        story_request=user_input,
        original_scores=original_scores,
        final_scores=final_scores,
        original_total=original_total,
        final_total=final_total,
        final_story=final_story,
    )

    return final_story, original_scores, final_scores


# Reads story requests from Excel and runs story generation on the requests
def process_all_story_requests(input_path: str = INPUT_PATH):
    # Read from the Excel file
    df = pd.read_excel(input_path)

    if "Story Request" not in df.columns:
        raise ValueError("Excel file must contain a 'Story Request' column.")

    story_requests = df["Story Request"].dropna().tolist()

    print(f"Found {len(story_requests)} story requests to process.")

    for request in story_requests:
        print(f"\n=== Processing request: {request} ===")
        run_story_session(request)


if __name__ == "__main__":
    main()
    # process_all_story_requests() # Evaluation pipeline
