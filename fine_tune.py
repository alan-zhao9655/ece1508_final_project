import os
import time
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel

# Set up device and random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MovieRecommendationDataset(Dataset):
    """Dataset for movie recommendation using user history and preferences with example recommendations."""
    
    def __init__(self, ratings_df, movies_df, tags_df=None, split="train", 
                 min_ratings=5, max_sequence_length=1024, save_json=True):
        """
        Initialize the dataset with MovieLens data.
        
        Args:
            ratings_df: DataFrame with user ratings
            movies_df: DataFrame with movie information
            tags_df: DataFrame with movie tags (optional)
            split: 'train' or 'validation'
            min_ratings: Minimum number of ratings per user to include
            max_sequence_length: Maximum sequence length for tokenization
            save_json: Whether to save training data to JSON file
        """
        self.max_sequence_length = max_sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create mappings from movie IDs to titles and genres
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        movie_id_to_genres = dict(zip(movies_df['movieId'], movies_df['genres']))
        
        # Process tags if provided
        movie_tags = defaultdict(list)
        if tags_df is not None:
            # Group tags by movie
            for _, row in tags_df.iterrows():
                movie_id = row['movieId']
                tag = row['tag']
                if isinstance(tag, str) and tag.strip():
                    movie_tags[movie_id].append(tag.lower().strip())
            
            # Remove duplicates and limit to top tags per movie
            for movie_id in movie_tags:
                unique_tags = list(set(movie_tags[movie_id]))
                movie_tags[movie_id] = unique_tags[:5] if len(unique_tags) > 5 else unique_tags
        
        # Build user sequences with example recommendations
        user_sequences = []
        user_ids = []
        json_data = []  # For storing training data in JSON format
        
        # Group ratings by user
        for user_id, group in ratings_df.groupby('userId'):
            # Skip users with too few ratings
            if len(group) < min_ratings:
                continue
            
            # Get all movies this user has rated and their ratings
            seen_movies = {}
            for _, row in group.iterrows():
                seen_movies[int(row['movieId'])] = float(row['rating'])  # Convert to standard Python types
            
            # Get highly rated movies (4.0+) for user history
            high_rated_movies = group[group['rating'] >= 4.0]
            
            # Skip users with too few highly rated movies
            if len(high_rated_movies) < 3:
                continue
                
            # Sort high-rated movies by rating value (descending)
            sorted_ratings = high_rated_movies.sort_values('rating', ascending=False)
            
            # Use up to 8 highly rated movies for history
            history_ratings = sorted_ratings.head(8)
            
            # Build the user history section
            sequence_lines = []
            sequence_lines.append(f"User {int(user_id)} has watched and enjoyed the following movies:")
            
            favorite_genres = defaultdict(int)
            favorite_tags = defaultdict(int)
            history_movies = []
            
            for _, row in history_ratings.iterrows():
                movie_id = int(row['movieId'])  # Convert to standard Python int
                rating = float(row['rating'])   # Convert to standard Python float
                
                # Skip if movie not in our dictionary
                if movie_id not in movie_id_to_title:
                    continue
                
                title = movie_id_to_title[movie_id]
                history_movies.append(movie_id)
                
                # Build a descriptive line with natural language interpretation
                line = f"{title}, rated {rating:.1f}/5."
                
                # Add sentiment with more variation
                if rating >= 5.0:
                    sentiment = np.random.choice([
                        "The user absolutely loves this movie.",
                        "This is one of the user's all-time favorites.",
                        "The user considers this a masterpiece."
                    ])
                    line += f" {sentiment}"
                elif rating >= 4.5:
                    sentiment = np.random.choice([
                        "The user greatly enjoyed this film.",
                        "The user thinks very highly of this movie.",
                        "This movie really impressed the user."
                    ])
                    line += f" {sentiment}"
                elif rating >= 4.0:
                    sentiment = np.random.choice([
                        "The user really liked this movie.",
                        "The user found this film quite enjoyable.",
                        "This movie resonated well with the user."
                    ])
                    line += f" {sentiment}"
                    
                # Track favorite genres for highly rated movies
                if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
                    genres = movie_id_to_genres[movie_id].split('|')
                    for genre in genres:
                        favorite_genres[genre] += 1
                        
                    # Add genre information occasionally
                    if np.random.random() < 0.7:
                        genre_str = " | ".join(genres)
                        line += f" Genres: {genre_str}."
                
                # Add tags if available with more comprehensive interpretation
                if movie_id in movie_tags and movie_tags[movie_id]:
                    # Track favorite tags
                    for tag in movie_tags[movie_id]:
                        favorite_tags[tag] += 1
                    
                    # Create tag descriptions with natural language
                    if len(movie_tags[movie_id]) > 2:
                        tags_list = ", ".join(movie_tags[movie_id])
                        line += f" The user appreciated elements like {tags_list} in this film."
                    else:
                        tags_list = " and ".join(movie_tags[movie_id])
                        line += f" The user valued the {tags_list} aspects of this movie."
                
                # Add to sequence with a bullet point
                sequence_lines.append(f"- {line}")
            
            # Add user preference summary
            top_genres = [genre for genre, count in sorted(favorite_genres.items(), key=lambda x: x[1], reverse=True)[:3]]
            if top_genres:
                genre_pref = ", ".join(top_genres)
                sequence_lines.append(f"\nThe user particularly enjoys {genre_pref} films.")
            
            top_tags = [tag for tag, count in sorted(favorite_tags.items(), key=lambda x: x[1], reverse=True)[:3]]
            if top_tags:
                tag_pref = ", ".join(top_tags)
                sequence_lines.append(f"The user often appreciates movies with {tag_pref}.")
            
            # Add the transition to recommendations
            sequence_lines.append("\nBased on these preferences, I recommend:")
            
            # Find recommendation movies (perfect 5.0 movies not yet seen by user)
            # First check if other users have given 5.0 ratings to movies this user hasn't seen
            potential_perfect_movies = ratings_df[
                (ratings_df['rating'] == 5.0) & 
                (~ratings_df['movieId'].isin(seen_movies.keys()))
            ]['movieId'].unique()
            
            # Convert to standard Python list of ints
            potential_perfect_movies = [int(x) for x in potential_perfect_movies]
            
            # Filter for genre match with user's favorites
            recommendations = []
            for movie_id in potential_perfect_movies:
                # Skip if no title info
                if movie_id not in movie_id_to_title:
                    continue
                
                # Calculate genre match score
                score = 0
                if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
                    for genre in movie_id_to_genres[movie_id].split('|'):
                        score += favorite_genres.get(genre, 0)
                
                # Add tag match score
                if movie_id in movie_tags:
                    for tag in movie_tags[movie_id]:
                        score += favorite_tags.get(tag, 0)
                
                # Only consider movies with some match
                if score > 0:
                    recommendations.append((movie_id, score))
            
            # If not enough perfect 5.0 movies, add high-scoring genre matches
            if len(recommendations) < 5:
                for movie_id, genres_str in movie_id_to_genres.items():
                    movie_id = int(movie_id) if isinstance(movie_id, np.integer) else movie_id
                    
                    # Skip if user has seen this movie or already in recommendations
                    if (movie_id in seen_movies) or (movie_id in [m for m, _ in recommendations]):
                        continue
                    
                    # Skip if no title info
                    if movie_id not in movie_id_to_title:
                        continue
                    
                    # Calculate genre match score
                    score = 0
                    if isinstance(genres_str, str):
                        for genre in genres_str.split('|'):
                            score += favorite_genres.get(genre, 0)
                    
                    # Add tag match score
                    if movie_id in movie_tags:
                        for tag in movie_tags[movie_id]:
                            score += favorite_tags.get(tag, 0)
                    
                    # Only consider movies with good match
                    if score >= 3:
                        recommendations.append((movie_id, score))
            
            # Sort by score and take top 5
            recommendations = [int(movie_id) for movie_id, _ in sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]]
            
            # Skip users with too few recommendations
            if len(recommendations) < 3:
                continue
                
            # Format recommendations
            rec_movies = []
            for i, movie_id in enumerate(recommendations, 1):
                title = movie_id_to_title[movie_id]
                rec_movies.append(movie_id)
                
                # Create explanation based on genres and tags
                explanation_parts = []
                
                # Add genre explanation
                if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
                    genres = movie_id_to_genres[movie_id].split('|')
                    if genres:
                        matching_genres = [g for g in genres if g in top_genres]
                        if matching_genres:
                            genre_text = ", ".join(matching_genres[:2])
                            explanation_parts.append(f"it's a {genre_text} film matching the user's preferred genres")
                
                # Add tag explanation
                if movie_id in movie_tags and movie_tags[movie_id]:
                    matching_tags = [t for t in movie_tags[movie_id] if t in top_tags]
                    if matching_tags:
                        tags_text = ", ".join(matching_tags[:2])
                        explanation_parts.append(f"it features {tags_text} that the user values")
                
                # Use generic explanation if needed
                if not explanation_parts:
                    explanation = "This movie aligns perfectly with the user's taste profile"
                else:
                    explanation = "This movie is recommended because " + " and ".join(explanation_parts)
                
                sequence_lines.append(f"{i}. {title} - {explanation}.")
            
            # Combine everything into one string
            sequence_text = "\n".join(sequence_lines)
            
            # Create JSON entry for this user - ensure all values are standard Python types
            json_entry = {
                "user_id": int(user_id),
                "history": {
                    "text": "\n".join(sequence_lines[:len(history_ratings) + 1]),  # User + history only
                    "movie_ids": [int(x) for x in history_movies]  # Convert all IDs to standard Python int
                },
                "recommendations": {
                    "text": "\n".join(sequence_lines[-(len(recommendations) + 1):]),  # Transition + recommendations
                    "movie_ids": [int(x) for x in rec_movies]  # Convert all IDs to standard Python int
                },
                "full_text": sequence_text
            }
            json_data.append(json_entry)
            
            user_sequences.append(sequence_text)
            user_ids.append(int(user_id))  # Convert to standard Python int
        
        # Save to JSON if requested
        if save_json:
            json_file = f"movielens_training_data_{split}.json"
            with open(json_file, 'w') as f:
                # Use the custom encoder to handle NumPy types
                json.dump(json_data, f, indent=2, cls=NumpyEncoder)
            print(f"Saved {len(json_data)} training examples to {json_file}")
        
        # Split into train and validation
        train_sequences, val_sequences, train_ids, val_ids = train_test_split(
            user_sequences, user_ids, test_size=0.2, random_state=RANDOM_SEED
        )
        
        # Select split
        if split == "train":
            self.raw_sequences = train_sequences
            self.user_ids = train_ids
        else:
            self.raw_sequences = val_sequences
            self.user_ids = val_ids
        
        # Tokenize sequences
        self.tokenized_data = []
        for sequence in self.raw_sequences:
            inputs = self.tokenizer(sequence, truncation=True, max_length=max_sequence_length)
            input_ids = torch.tensor(inputs['input_ids'])
            self.tokenized_data.append(input_ids)
        
        # Determine block size
        self.block_size = min(
            self.max_sequence_length,
            max((len(d) for d in self.tokenized_data), default=0)
        )
        
        print(f"Created {len(self.tokenized_data)} sequences for {split} split")
        print(f"Max sequence length: {self.block_size}")
        if len(self.raw_sequences) > 0:
            print(f"Sample sequence:\n---\n{self.raw_sequences[0][:400]}...\n---")
        
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        """
        Return a pair of tensors (x, y) where:
        - x is the input sequence (all tokens except the last)
        - y is the target sequence (all tokens except the first)
        """
        tokens = self.tokenized_data[idx]
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def collate_fn(batch):
    """
    Collate function to handle variable length sequences in a batch.
    
    Args:
        batch: List of (x, y) tuples from dataset.__getitem__
        
    Returns:
        Padded batch of input and target sequences
    """
    # Separate inputs and targets
    xs, ys = zip(*batch)
    
    # Find max length in this batch
    max_len = max([len(x) for x in xs])
    
    # Pad sequences
    def pad_sequence(seq, max_len):
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)])
    
    # Stack to create batched tensors
    x_padded = torch.stack([pad_sequence(x, max_len) for x in xs])
    y_padded = torch.stack([pad_sequence(y, max_len) for y in ys])
    
    return x_padded.to(device), y_padded.to(device)


def load_movielens(folder="ml-latest-small"):
    """Load MovieLens dataset from csv files."""
    ratings_file = os.path.join(folder, "ratings.csv")
    movies_file = os.path.join(folder, "movies.csv")
    tags_file = os.path.join(folder, "tags.csv")
    
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    
    try:
        tags_df = pd.read_csv(tags_file)
        print(f"Loaded {len(tags_df)} tags")
    except:
        tags_df = None
        print("Tags file not found or couldn't be loaded")
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    return ratings_df, movies_df, tags_df


def load_training_data_from_json(json_file, split="train"):
    """
    Load pre-processed training data from JSON file.
    
    Args:
        json_file: Path to JSON file
        split: 'train' or 'validation' to filter data
        
    Returns:
        List of training examples
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter by split if needed
    if split in json_file:
        return data
    
    # Otherwise, split manually
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED
    )
    
    return train_data if split == "train" else val_data


def train_lora_model(train_loader, val_loader, base_model_name="gpt2", epochs=3, learning_rate=2e-4):
    """
    Train a GPT-2 model with LoRA adapters.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        base_model_name: Name of the pretrained model
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    # Load pretrained model
    print(f"Loading pretrained model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    print(model.print_trainable_parameters())
    
    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = epochs * len(train_loader)
    warmup_steps = min(100, int(0.1 * total_steps))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float("inf")
    output_dir = f"{base_model_name}_movielens_lora"
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item() * x.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                val_loss += loss.item() * x.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"✓ Saved best model checkpoint to {output_dir}")
    
    return model, tokenizer


def generate_recommendation(prompt, model_path, base_model_name="gpt2", max_new_tokens=150, temperature=0.7):
    """
    Generate a movie recommendation using the fine-tuned model.
    
    Args:
        prompt: Input text prompt with user history
        model_path: Path to saved LoRA adapter weights
        base_model_name: Name of the base model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated recommendation text
    """
    # Load base model and adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    model.eval()
    
    # Ensure prompt ends with recommendation request
    if "Based on these preferences, I recommend:" not in prompt:
        prompt += "\n\nBased on these preferences, I recommend:"
    
    # Tokenize input WITH attention mask
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=1024
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate output with proper parameters
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.92,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode the output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part (after the prompt)
    recommendation_text = full_text[len(prompt):].strip()
    
    return recommendation_text


def create_generation_prompt(user_id, ratings_df, movies_df, tags_df):
    """
    Create a prompt for a real user without including recommendations.
    This is used for inference to generate new recommendations.
    
    The format matches exactly the history part of the training data.
    """
    # Check if we have saved JSON data
    json_file = "movielens_training_data_train.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Look for this user in the saved data
        for entry in data:
            if entry["user_id"] == user_id:
                # Return just the history part + recommendation request
                return entry["history"]["text"] + "\n\nBased on these preferences, I recommend:"
    
    # If not found in JSON or file doesn't exist, generate from scratch
    # Mappings
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
    movie_id_to_genres = dict(zip(movies_df['movieId'], movies_df['genres']))
    
    # Process tags
    movie_tags = defaultdict(list)
    if tags_df is not None:
        for _, row in tags_df.iterrows():
            if isinstance(row['tag'], str) and row['tag'].strip():
                movie_tags[row['movieId']].append(row['tag'].lower().strip())
        
        for movie_id in movie_tags:
            movie_tags[movie_id] = list(set(movie_tags[movie_id]))[:5]
    
    # Get user ratings, focusing on highly rated movies
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if len(user_ratings) == 0:
        return f"User {user_id} not found or has no ratings."
    
    # Get highly rated movies (4.0+)
    high_rated = user_ratings[user_ratings['rating'] >= 4.0]
    
    # Sort by rating
    sorted_ratings = high_rated.sort_values('rating', ascending=False)
    
    # Limit to top 8 movies
    top_movies = sorted_ratings.head(8)
    
    # Create prompt with exact same format as training data
    sequence_lines = []
    sequence_lines.append(f"User {int(user_id)} has watched and enjoyed the following movies:")
    
    favorite_genres = defaultdict(int)
    favorite_tags = defaultdict(int)
    
    for _, row in top_movies.iterrows():
        movie_id = int(row['movieId'])
        rating = float(row['rating'])
        
        if movie_id not in movie_id_to_title:
            continue
            
        title = movie_id_to_title[movie_id]
        
        # Build a descriptive line with natural language interpretation - same as training data
        line = f"{title}, rated {rating:.1f}/5."
        
        # Add sentiment with more variation - same as training data
        if rating >= 5.0:
            sentiment = np.random.choice([
                "The user absolutely loves this movie.",
                "This is one of the user's all-time favorites.",
                "The user considers this a masterpiece."
            ])
            line += f" {sentiment}"
        elif rating >= 4.5:
            sentiment = np.random.choice([
                "The user greatly enjoyed this film.",
                "The user thinks very highly of this movie.",
                "This movie really impressed the user."
            ])
            line += f" {sentiment}"
        elif rating >= 4.0:
            sentiment = np.random.choice([
                "The user really liked this movie.",
                "The user found this film quite enjoyable.",
                "This movie resonated well with the user."
            ])
            line += f" {sentiment}"
        
        # Track and add genre information - same as training data
        if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
            genres = movie_id_to_genres[movie_id].split('|')
            for genre in genres:
                favorite_genres[genre] += 1
            
            # Add genre information occasionally
            if np.random.random() < 0.7:
                genre_str = " | ".join(genres)
                line += f" Genres: {genre_str}."
        
        # Add tags with same format as training data
        if movie_id in movie_tags and movie_tags[movie_id]:
            # Track favorite tags
            for tag in movie_tags[movie_id]:
                favorite_tags[tag] += 1
            
            # Create tag descriptions with natural language
            if len(movie_tags[movie_id]) > 2:
                tags_list = ", ".join(movie_tags[movie_id])
                line += f" The user appreciated elements like {tags_list} in this film."
            else:
                tags_list = " and ".join(movie_tags[movie_id])
                line += f" The user valued the {tags_list} aspects of this movie."
            
        sequence_lines.append(f"- {line}")
    
    # Add user preference summary - same as training data
    top_genres = [genre for genre, count in sorted(favorite_genres.items(), key=lambda x: x[1], reverse=True)[:3]]
    if top_genres:
        genre_pref = ", ".join(top_genres)
        sequence_lines.append(f"\nThe user particularly enjoys {genre_pref} films.")
    
    top_tags = [tag for tag, count in sorted(favorite_tags.items(), key=lambda x: x[1], reverse=True)[:3]]
    if top_tags:
        tag_pref = ", ".join(top_tags)
        sequence_lines.append(f"The user often appreciates movies with {tag_pref}.")
    
    # Add recommendation request
    sequence_lines.append("\nBased on these preferences, I recommend:")
    
    return "\n".join(sequence_lines)


def main(skip_training=False):
    # Load data
    print("Loading MovieLens dataset...")
    ratings_df, movies_df, tags_df = load_movielens()
    
    # Check if we have saved training data
    train_json = "movielens_training_data_train.json"
    val_json = "movielens_training_data_validation.json"
    
    if os.path.exists(train_json) and os.path.exists(val_json):
        print(f"Found saved training data in {train_json} and {val_json}")
        
        # You can load the data directly instead of re-processing
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        
        with open(val_json, 'r') as f:
            val_data = json.load(f)
            
        print(f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples")
        
        # Display a sample
        if train_data:
            print("\nSample training example:")
            print("="*80)
            print(train_data[0]["full_text"][:500] + "...\n")
            
        # Now we need to create datasets from this data for training
        # First, tokenize the full text for each example
        print("Creating datasets from saved JSON data...")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        max_sequence_length = 1024
        
        # Process training data
        train_tokenized = []
        for item in train_data:
            inputs = tokenizer(item["full_text"], truncation=True, max_length=max_sequence_length)
            input_ids = torch.tensor(inputs['input_ids'])
            train_tokenized.append(input_ids)
            
        # Process validation data
        val_tokenized = []
        for item in val_data:
            inputs = tokenizer(item["full_text"], truncation=True, max_length=max_sequence_length)
            input_ids = torch.tensor(inputs['input_ids'])
            val_tokenized.append(input_ids)
            
        # Create custom dataset classes to hold this data
        class JSONDataset(Dataset):
            def __init__(self, tokenized_data, max_len=1024):
                self.tokenized_data = tokenized_data
                self.block_size = min(max_len, max(len(d) for d in tokenized_data))
                
            def __len__(self):
                return len(self.tokenized_data)
                
            def __getitem__(self, idx):
                tokens = self.tokenized_data[idx]
                x = tokens[:-1]
                y = tokens[1:]
                return x, y
                
        train_dataset = JSONDataset(train_tokenized)
        val_dataset = JSONDataset(val_tokenized)
        print(f"Created dataset with {len(train_dataset)} training examples and {len(val_dataset)} validation examples")
    else:
        # Create datasets with improved formatting
        print("Creating datasets and saving to JSON...")
        train_dataset = MovieRecommendationDataset(
            ratings_df=ratings_df,
            movies_df=movies_df,
            tags_df=tags_df,
            split="train",
            min_ratings=5,
            max_sequence_length=1024,
            save_json=True
        )
        
        val_dataset = MovieRecommendationDataset(
            ratings_df=ratings_df,
            movies_df=movies_df,
            tags_df=tags_df,
            split="validation",
            min_ratings=5,
            max_sequence_length=1024,
            save_json=True
        )
    
    # Create data loaders for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Determine if we should train
    do_training = not skip_training
    if do_training and not skip_training:
        do_training = input("\nDo you want to train the model now? (y/n): ").strip().lower() == 'y'
    
    if do_training:
        # Train model with more epochs for better learning
        print("Training model...")
        model, tokenizer = train_lora_model(
            train_loader=train_loader,
            val_loader=val_loader,
            base_model_name="gpt2",
            epochs=10,
            learning_rate=2e-4
        )
    else:
        print("Skipping training, using previously trained model if available...")
    
    # Generate sample recommendations for evaluation
    print("\nGenerating sample recommendations...")
    
    model_path = "gpt2_movielens_lora"
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} not found. You may need to train the model first.")
        if not do_training:
            print("Exiting as no model is available and training was skipped.")
            return
    
    # Load JSON data to find users with good history
    user_ids_to_test = []
    
    if os.path.exists(train_json):
        with open(train_json, 'r') as f:
            data = json.load(f)
        
        # Find some users with rich history
        for entry in data[:20]:  # Check first 20 entries
            user_ids_to_test.append(entry["user_id"])
            if len(user_ids_to_test) >= 3:
                break
    
    # If no JSON data, use default test users
    if not user_ids_to_test:
        user_ids_to_test = [425, 381, 64]  # Example user IDs
    
    for user_id in user_ids_to_test:
        print(f"\n=============== Recommendations for User {user_id} ===============")
        
        # Create prompt with same format as training data
        prompt = create_generation_prompt(user_id, ratings_df, movies_df, tags_df)
        print("\n--- Prompt ---\n")
        print(prompt)
        
        # Generate recommendations
        try:
            recommendations = generate_recommendation(
                prompt=prompt,
                model_path=model_path,
                max_new_tokens=200,
                temperature=0.8
            )
            
            print("\n--- Generated Recommendations ---\n")
            print(recommendations)
        except Exception as e:
            print(f"\nError generating recommendations: {e}")
            print("This may be due to missing model files or other issues.")
        
        print("=" * 70)
    
    # Save a sample prompt for later use in fine-tuning
    sample_file = "sample_prompt_format.txt"
    with open(sample_file, "w") as f:
        f.write(prompt)
    print(f"\nSaved sample prompt format to {sample_file} for reference when training")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MovieLens recommendation system')
    parser.add_argument('--generate-only', action='store_true',
                       help='Skip training and only generate recommendations')
    args = parser.parse_args()
    
    if args.generate_only:
        # Skip to generation part
        main(skip_training=True)
    else:
        main()