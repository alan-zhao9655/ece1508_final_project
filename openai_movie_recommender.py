import os
from dotenv import load_dotenv
import random
import time
import pandas as pd
import numpy as np
from openai import OpenAI
from collections import defaultdict
import json
import argparse

class OpenAIMovieRecommender:
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Initialize the OpenAI movie recommender
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable if None)
            model: OpenAI model to use
        """
        # Use API key from environment variable if not explicitly provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client with API key
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_recommendations(self, user_history, num_recommendations=5):
        """
        Generate movie recommendations using OpenAI
        
        Args:
            user_history: String containing user's movie history
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommendations with explanations
        """
        try:
            # Create a prompt for OpenAI
            prompt = f"{user_history}\n\nBased on this user's preferences, recommend {num_recommendations} movies they might enjoy. For each recommendation, include a brief explanation of why it matches their taste profile. Format each recommendation as a numbered list."
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a movie recommendation expert who understands user preferences and provides personalized suggestions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            # Extract recommendations from response
            recommendation_text = response.choices[0].message.content.strip()
            
            # Parse the recommendations
            recommendations = self._parse_recommendations(recommendation_text)
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return [f"Error generating recommendations: {str(e)}"]
    
    def _parse_recommendations(self, recommendation_text):
        """
        Parse the OpenAI response to extract recommendations
        """
        # Split by numbered list items
        import re
        pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|\Z)'
        matches = re.findall(pattern, recommendation_text, re.DOTALL)
        
        # Clean up the recommendations
        recommendations = [match.strip() for match in matches if match.strip()]
        
        # If parsing fails, return the raw text
        if not recommendations:
            return [recommendation_text]
        
        return recommendations

def create_user_histories(ratings_df, movies_df, tags_df=None, min_ratings=5, max_movies_per_user=10):
    """
    Create user history descriptions from movie ratings and tags
    
    Returns:
        Dictionary mapping user_id to user history text
    """
    # Create a mapping of movie ID to title
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    
    # Process tags if provided
    movie_tags = defaultdict(list)
    if tags_df is not None:
        # Group tags by movie
        for _, row in tags_df.iterrows():
            movie_id = row['movieId']
            tag = row['tag']
            if pd.notna(tag) and str(tag).strip():  # Skip empty tags
                movie_tags[movie_id].append(tag.lower().strip())
        
        # Remove duplicates and limit to top tags per movie
        for movie_id in movie_tags:
            unique_tags = list(set(movie_tags[movie_id]))
            movie_tags[movie_id] = unique_tags[:5] if len(unique_tags) > 5 else unique_tags
    
    # Group ratings by user
    user_groups = ratings_df.groupby('userId')
    
    # Collect all eligible users
    eligible_users = []
    for user_id, group in user_groups:
        if len(group) >= min_ratings:
            eligible_users.append((user_id, group))
    
    user_histories = {}
    
    # Process each eligible user
    for user_id, group in eligible_users:
        # Sort ratings by rating value (descending)
        sorted_ratings = group.sort_values('rating', ascending=False)
        
        # Limit number of movies per user
        if max_movies_per_user > 0:
            sorted_ratings = sorted_ratings.head(max_movies_per_user)
        
        # Build user history
        history_lines = []
        history_lines.append(f"User {user_id} has watched the following movies:")
        
        for _, row in sorted_ratings.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # Skip if movie not in our dictionary
            if movie_id not in movie_id_to_title:
                continue
            
            title = movie_id_to_title[movie_id]
            
            # Build a descriptive line
            line = f"{title}, rated {rating:.1f}/5."
            
            # Add sentiment
            if rating > 4.0:
                line += " The user loves this movie."
            
            # Add tags if available
            if movie_id in movie_tags and len(movie_tags[movie_id]) > 0:
                tags_list = ", ".join(movie_tags[movie_id])
                line += f" This movie was tagged by the user with words: {tags_list}."
            
            # Add to history
            history_lines.append(f" - {line}")
        
        # Combine into a single string
        user_history = "\n".join(history_lines)
        user_histories[user_id] = user_history
    
    return user_histories

def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations using OpenAI")
    parser.add_argument('--data_folder', type=str, default="ml-latest-small", help="Path to MovieLens data folder")
    parser.add_argument('--api_key', type=str, default=None, help="OpenAI API key (defaults to OPENAI_API_KEY env variable if not specified)")
    parser.add_argument('--model', type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument('--max_movies', type=int, default=10, help="Maximum number of movies per user")
    parser.add_argument('--num_users', type=int, default=5, help="Number of users to generate recommendations for")
    parser.add_argument('--output', type=str, default="openai_recommendations.json", help="Output JSON file")
    args = parser.parse_args()

    # example usage:
    # python openai_recommender.py --data_folder ml-latest-small --num_users 5 --output openai_recommendations.json
    
    # Load MovieLens data
    ratings_file = f"{args.data_folder}/ratings.csv"
    movies_file = f"{args.data_folder}/movies.csv"
    tags_file = f"{args.data_folder}/tags.csv"
    
    print(f"Loading data from {args.data_folder}...")
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    
    try:
        tags_df = pd.read_csv(tags_file)
        print(f"Loaded {len(tags_df)} tags")
    except:
        tags_df = None
        print("Tags file not found or couldn't be loaded")
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    
    # Create user histories
    print("Creating user histories...")
    user_histories = create_user_histories(
        ratings_df=ratings_df,
        movies_df=movies_df,
        tags_df=tags_df,
        min_ratings=5,
        max_movies_per_user=args.max_movies
    )
    
    print(f"Created histories for {len(user_histories)} users")
    
    # Select random users for recommendations
    # selected_users = random.sample(list(user_histories.keys()), min(args.num_users, len(user_histories)))
    selected_users = [381, 376, 64, 425, 368]
    print(f"Selected {len(selected_users)} users for OpenAI recommendations: {selected_users}")
    
    # Initialize OpenAI recommender
    print("Initializing OpenAI recommender...")
    recommender = OpenAIMovieRecommender(api_key=args.api_key, model=args.model)
    
    # Generate recommendations
    results = []
    for i, user_id in enumerate(selected_users):
        print(f"Generating recommendations for user {user_id} ({i+1}/{len(selected_users)})")
        
        try:
            # Get recommendations
            recommendations = recommender.generate_recommendations(user_histories[user_id])
            
            # Store results
            results.append({
                "user_id": user_id,
                "history": user_histories[user_id],
                "recommendations": recommendations
            })
            
            # Add delay to avoid rate limits
            time.sleep(1)
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {str(e)}")
    
    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump({"data": results}, f, indent=2)
    
    print(f"Saved recommendations to {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()