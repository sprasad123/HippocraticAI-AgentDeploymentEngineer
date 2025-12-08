# Hippocratic AI - Agent Deployment Engineer Take Home

Objective: Generate a bedtime story for kids ages 5 - 10 years old and have a judge that contributes to automated 
improvement of the story.

## Instructions (Set Up)
1. pip install -r requirements.txt

2. export OPENAI_API_KEY=“Enter your OpenAI API key here”

3. python3 updated_main.py

## Approach
I was inspired by the Generative Adversarial Network (GAN) framework to create a judge that critiques the generator. I have a generator LLM, powered by ChatGPT-3.5-turbo, that creates a bedtime story for kids ages 5-10. The judge LLM critiques the story based on the following criteria: age fit, warmth, creativity, coherence, and safety. I sum these weighted scores and if the final score is below 80%, then there is a training loop where the story is improved upon by providing the story draft, judge's feedback, and the user input. 

The weights are assigned as the following:
{
    "age_fit": 0.30,
    "warmth": 0.20,
    "creativity": 0.20,
    "coherence": 0.20,
    "safety": 0.10
}

The bedtime story application is fairly straightforward, so I designed the judge LLM to be harsh and judgmental because the generator LLM is already good. In the future, we could consider making a more complex or nuanced story that is harder to generate.

## Evaluation

The evaluation results are shown in the file "Evaluation.csv".

I ran the generator-judge flow across 25 user inputs, where the last 5 were not age appropriate requests. We can see the scoring across age fit, warmth, creativity, coherence, and safety for the first version of the story and also for the last version of the story. The last column shows the final story text. We can notice that the flow entered the training loop in row 19 where the original story had a score of 72%, and after being improved upon, it resulted in a final score of 96%. 

I kept the temperature value to 0.9 for generating or improving the story. This resulted in more creative stories with specific aspects to the character's personality. The stories with this temperature typically read more interestingly and could keep users engaged. 

## Wishlist

Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

I would make the story generation process more interactive, where the chatbot asks the user questions about how they would like the story to turn out.

Wishlist (for later on):
1. Incorporating a text to voice feature so that the user can have the story being read aloud to them. 
2. Giving the characters specific, crafted personalities, by giving a few examples in the prompt.
3. Introducing a human judge to score. 
4. Fine-tuning after gathering enough data.
5. Explore other models, such as Gemini, Claude, Perplexity
