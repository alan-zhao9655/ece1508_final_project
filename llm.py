import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed
from mingpt.model import GPT
import time
import random
from collections import defaultdict

# Set seed for reproducibility
set_seed(42)

class MovieRecommendationDataset(Dataset):
    def __init__(self, ratings_df, movies_df, tags_df=None, split="train", 
                 min_ratings=5, max_sequence_length=1024, generate_examples=True):
        """
        Dataset for movie recommendation using MinGPT with example recommendations

        Args:
            ratings_df: DataFrame with user ratings
            movies_df: DataFrame with movie information
            tags_df: DataFrame with movie tags (optional)
            split: 'train' or 'validation'
            min_ratings: Minimum number of ratings per user to include
            max_sequence_length: Maximum sequence length to use
            generate_examples: Whether to include example recommendations in the training data
        """
        self.tokenizer = BPETokenizer()
        self.max_sequence_length = max_sequence_length
        
        # Create a mapping of movie ID to title and genres
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        movie_id_to_genres = dict(zip(movies_df['movieId'], movies_df['genres']))
        
        # Process tags if provided
        from collections import defaultdict
        movie_tags = defaultdict(list)
        if tags_df is not None:
            # Group tags by movie
            for _, row in tags_df.iterrows():
                movie_id = row['movieId']
                tag = row['tag']
                if pd.notna(tag) and str(tag).strip():  # Skip empty tags
                    movie_tags[movie_id].append(tag.lower().strip())
            
            # Remove duplicates and optionally limit to top tags per movie
            for movie_id in movie_tags:
                unique_tags = list(set(movie_tags[movie_id]))
                # If desired, limit to top 5 tags
                movie_tags[movie_id] = unique_tags[:5] if len(unique_tags) > 5 else unique_tags
        
        # Group ratings by user
        user_groups = ratings_df.groupby('userId')
        
        user_sequences = []
        user_ids = []
        
        # Create a mapping of movie genres for recommending similar movies
        genre_to_movies = defaultdict(list)
        for movie_id, genres in movie_id_to_genres.items():
            if isinstance(genres, str):
                for genre in genres.split('|'):
                    genre_to_movies[genre].append(movie_id)
        
        for user_id, group in user_groups:
            # Skip users with too few ratings
            if len(group) < min_ratings:
                continue
            
            # Sort ratings by rating value (descending), so highest rated are first
            sorted_ratings = group.sort_values('rating', ascending=False)
            sorted_ratings = sorted_ratings.head(10)
            
            # Build a descriptive, multi-line text block
            sequence_lines = []
            sequence_lines.append(f"User {user_id} has watched the following movies:")
            
            # Track user's favorite genres and the movies they've seen
            user_genres = defaultdict(int)
            user_seen_movies = set()
            
            for _, row in sorted_ratings.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                user_seen_movies.add(movie_id)
                
                # Retrieve title from our dictionary
                if movie_id in movie_id_to_title:
                    title = movie_id_to_title[movie_id]
                    
                    # Start building a sentence
                    line = f"{title}, rated {rating:.1f}/5."
                    
                    # If rating > 4.0, add "The user loves this movie."
                    if rating > 4.0:
                        line += " The user loves this movie."
                        
                        # Track favorite genres for highly rated movies
                        if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
                            for genre in movie_id_to_genres[movie_id].split('|'):
                                user_genres[genre] += 1
                    
                    # If there are any tags for this movie, add a descriptive sentence
                    if movie_id in movie_tags and len(movie_tags[movie_id]) > 0:
                        tags_list = ", ".join(movie_tags[movie_id])
                        line += f" This movie was tagged by the user with words: {tags_list}."
                    
                    # Wrap it in a bullet point for clarity
                    sequence_lines.append(f" - {line}")
            
            # Add a recommendation request prompt
            sequence_lines.append("\nBased on this user's preferences, here are movie recommendations:")
            
            # Only add example recommendations during training
            if generate_examples:
                # Generate example recommendations based on user preferences
                recommended_movies = self._generate_recommendations(
                    user_genres, genre_to_movies, user_seen_movies, 
                    movie_id_to_title, movie_id_to_genres, movie_tags
                )
                
                # Add the recommended movies to the sequence
                for i, (movie_id, reason) in enumerate(recommended_movies, 1):
                    title = movie_id_to_title[movie_id]
                    sequence_lines.append(f"\n{i}. {title} - {reason}")
            
            # Combine everything into one string
            sequence_text = "\n".join(sequence_lines)
            
            user_sequences.append(sequence_text)
            user_ids.append(user_id)

        train_sequences, val_sequences, train_ids, val_ids = train_test_split(
            user_sequences, user_ids, test_size=0.2, random_state=42
        )
        
        # Select split
        if split == "train":
            self.raw_sequences = train_sequences
            self.user_ids = train_ids
        else:
            self.raw_sequences = val_sequences
            self.user_ids = val_ids
        
        # Tokenize
        self.data = []
        for sequence in self.raw_sequences:
            tokenized = self.tokenizer(sequence).view(-1)
            # Truncate if needed
            if self.max_sequence_length >= 0:
                tokenized = tokenized[:self.max_sequence_length]
            self.data.append(tokenized)
        
        # Determine block size
        self.block_size = min(
            self.max_sequence_length,
            max((len(d) for d in self.data), default=0)
        )
        
        print(f"Created {len(self.data)} sequences for {split} split")
        print(f"Max sequence length: {self.block_size}")
        if len(self.data) > 0:
            # Show a small sample of the first tokenized sequence
            sample_decoded = self.tokenizer.decode(self.data[0])
            print(f"Sample sequence:\n---\n{sample_decoded}\n---\n")
    
    def _generate_recommendations(self, user_genres, genre_to_movies, user_seen_movies, 
                                movie_id_to_title, movie_id_to_genres, movie_tags, num_recommendations=5):
        """
        Generate example recommendations based on user preferences
        
        Returns:
            List of (movie_id, reason) tuples
        """
        recommendations = []
        
        # Sort genres by user preference
        favorite_genres = sorted(user_genres.items(), key=lambda x: x[1], reverse=True)
        
        # Try to find unseen movies from favorite genres
        candidate_movies = set()
        for genre, _ in favorite_genres:
            # Add movies from this genre to candidates
            for movie_id in genre_to_movies[genre]:
                if movie_id not in user_seen_movies and movie_id in movie_id_to_title:
                    candidate_movies.add(movie_id)
            
            # If we have enough candidates, stop adding more
            if len(candidate_movies) >= num_recommendations * 3:
                break
        
        # If we don't have enough candidates, add some popular movies
        if len(candidate_movies) < num_recommendations:
            for genre, movies in genre_to_movies.items():
                for movie_id in movies:
                    if movie_id not in user_seen_movies and movie_id in movie_id_to_title:
                        candidate_movies.add(movie_id)
                    if len(candidate_movies) >= num_recommendations * 3:
                        break
                if len(candidate_movies) >= num_recommendations * 3:
                    break
        
        # Convert to list and shuffle to add variety
        candidate_list = list(candidate_movies)
        random.shuffle(candidate_list)
        
        # Select and generate reasons for recommendations
        for movie_id in candidate_list[:num_recommendations]:
            # Generate a plausible reason for this recommendation
            reason = self._generate_recommendation_reason(
                movie_id, user_genres, movie_id_to_genres, movie_tags
            )
            recommendations.append((movie_id, reason))
            
        return recommendations
    
    def _generate_recommendation_reason(self, movie_id, user_genres, movie_id_to_genres, movie_tags):
        """
        Generate a plausible reason for recommending this movie based on user preferences
        """
        reasons = []
        
        # Check if the movie has genres that match user preferences
        if movie_id in movie_id_to_genres and isinstance(movie_id_to_genres[movie_id], str):
            movie_genres = movie_id_to_genres[movie_id].split('|')
            matching_genres = [g for g in movie_genres if g in user_genres]
            
            if matching_genres:
                # Format the genre list for readability
                if len(matching_genres) == 1:
                    genre_text = matching_genres[0]
                elif len(matching_genres) == 2:
                    genre_text = f"{matching_genres[0]} and {matching_genres[1]}"
                else:
                    genre_text = ", ".join(matching_genres[:-1]) + f", and {matching_genres[-1]}"
                
                reasons.append(f"This {genre_text} film aligns with the user's preferred genres")
        
        # Check if the movie has tags that might be interesting
        if movie_id in movie_tags and movie_tags[movie_id]:
            tags = movie_tags[movie_id]
            if len(tags) > 0:
                tag_sample = random.sample(tags, min(2, len(tags)))
                tag_text = " and ".join(tag_sample)
                reasons.append(f"features {tag_text} which may appeal to the user")
        
        # If we have no specific reasons, provide a generic one
        if not reasons:
            reasons.append("would be a good addition to the user's watchlist based on their taste profile")
        
        # Combine the reasons
        if len(reasons) == 1:
            return reasons[0].capitalize()
        else:
            return f"{reasons[0].capitalize()} and {reasons[1]}"
    
    def __len__(self):
        return len(self.data)
    
    def get_vocab_size(self):
        return 50257  # Same as the tokenizer's vocab size
    
    def get_block_size(self):
        return self.block_size
    
    def __getitem__(self, idx):
        """
        Return a pair of tensors (x, y) where:
        - x is the input sequence
        - y is the target sequence (shifted by 1)
        """
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return (x, y)


def lm_collate_fn(batch, device=None):
    """
    Custom collate function for language modeling
    """
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Find max length in this batch
    max_input_len = max([len(x) for x in inputs])
    max_target_len = max([len(y) for y in targets])
    
    # Pad sequences
    padded_inputs = [torch.cat([x, torch.zeros(max_input_len - len(x), dtype=torch.long)]) 
                     for x in inputs]
    padded_targets = [torch.cat([y, torch.zeros(max_target_len - len(y), dtype=torch.long)]) 
                      for y in targets]
    
    # Stack to create batched tensors
    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)
    
    # Move to device if specified
    if device is not None:
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)
    
    return inputs_tensor, targets_tensor

def load_and_prepare_data(data_folder="ml-latest-small"):
    """
    Load and prepare MovieLens data
    """
    ratings_file = f"{data_folder}/ratings.csv"
    movies_file = f"{data_folder}/movies.csv"
    tags_file = f"{data_folder}/tags.csv"
    
    # Read CSVs
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    
    # Read tags if the file exists
    try:
        tags_df = pd.read_csv(tags_file)
        print(f"Loaded {len(tags_df)} tags")
    except:
        tags_df = None
        print("Tags file not found or couldn't be loaded")
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    return ratings_df, movies_df, tags_df

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device):
    """
    Custom training loop for language model
    """
    # Training stats
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Criterion for language modeling
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Forward pass
            logits, _ = model(x)
            
            # Reshape logits to [batch_size*seq_len, vocab_size]
            logits = logits.reshape(-1, logits.size(-1))
            y = y.reshape(-1)
            
            # Compute loss (ignore padding tokens)
            loss = criterion(logits, y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Accumulate loss
            total_train_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Compute average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                # Forward pass
                logits, _ = model(x)
                
                # Reshape logits
                logits = logits.reshape(-1, logits.size(-1))
                y = y.reshape(-1)
                
                # Compute loss
                loss = criterion(logits, y)
                
                # Accumulate loss
                total_val_loss += loss.item()
        
        # Compute average validation loss and perplexity
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        perplexity = np.exp(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} completed in {train_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_movie_recommender.pt")
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
    
    # Plot training and validation curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses

def generate_recommendations(
    model, 
    tokenizer, 
    user_sequence, 
    max_new_tokens=500, 
    temperature=0.8, 
    top_k=50, 
    top_p=0.95, 
    device=None
):
    """
    Generate movie recommendations using the trained model, with a more descriptive prompt.
    """
    
    # Create a prompt with examples to demonstrate the expected format
    prompt = (
        user_sequence
        + "\n\nBased on this user's ratings and preferences, here are 5 personalized movie recommendations:"
        + "\n\n1. The Godfather (1972) - This crime drama would appeal to the user based on their appreciation for compelling storytelling and strong character development seen in their other highly-rated films."
        + "\n\n2. Pulp Fiction (1994) - The user would enjoy this film's unique narrative structure and memorable characters, similar to other films they've rated highly."
        + "\n\n3. Inception (2010) - Given the user's interest in thought-provoking and visually stunning films like those in their watch history, this mind-bending thriller would be a perfect match."
        + "\n\n4. The Shawshank Redemption (1994) - The emotional depth and powerful storytelling in this film align with the user's preference for meaningful cinema."
        + "\n\n5. "
    )
    
    # Tokenize the prompt
    x = tokenizer(prompt).view(-1).unsqueeze(0)
    if device:
        x = x.to(device)
    
    # Generate text from the model
    model.eval()
    with torch.no_grad():
        output = model.generate(
            x, 
            device, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode the generated output
    output_array = output[0].cpu().numpy()
    generated_text = tokenizer.decode(output_array)
    
    # Extract the recommendations
    try:
        # Get the text that comes after our examples
        full_recommendations = generated_text.strip()
        
        # Extract all recommendations using regex
        import re
        pattern = r'\d+\.\s+(.*?)(?=\n\n\d+\.|\Z)'
        matches = re.findall(pattern, full_recommendations, re.DOTALL)
        
        # Clean up and return the recommendations
        recommendations = []
        for match in matches:
            match = match.strip()
            if match and not match.startswith("Given the user's preferences"):
                recommendations.append(match)
        
        # If we found the 5th recommendation (the one the model generated)
        if len(recommendations) >= 5:
            return [recommendations[4]]  # Return just the model-generated one
        else:
            return ["Unable to generate a valid recommendation"]
    except Exception as e:
        return [f"Failed to generate valid recommendations: {str(e)}"]



def evaluate_model(model, tokenizer, test_set, num_samples=5, device=None):
    """
    Evaluate the model by generating recommendations for test users,
    with better handling of recommendation generation and display
    """
    # Sample users from test set
    indices = random.sample(range(len(test_set)), min(num_samples, len(test_set)))
    all_recommendations = []
    
    for idx in indices:
        # Get user sequence
        sequence = test_set.raw_sequences[idx]
        user_id = test_set.user_ids[idx]
        
        print("\n" + "="*80)
        print(f"User ID: {user_id}")
        print(f"History (truncated): {sequence[:200]}...")
        
        # Generate recommendations with retry logic
        try:
            recommendations = generate_recommendations(model, tokenizer, sequence, device=device)
            
            # Check if we got valid recommendations
            if not recommendations or all(r.startswith("Failed") or r.startswith("Unable") for r in recommendations):
                print("\nRetrying with different parameters...")
                # Retry with different temperature
                recommendations = generate_recommendations(
                    model, tokenizer, sequence, 
                    temperature=1.2, max_new_tokens=300, device=device
                )
            
            # Print recommendations
            print("\nRecommendations:")
            if recommendations and not all(r.startswith("Failed") or r.startswith("Unable") for r in recommendations):
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"  {i}. {rec}")
                    
                # Store successful recommendations
                all_recommendations.append({
                    "user_id": user_id,
                    "recommendations": recommendations[:5]
                })
            else:
                print("  Could not generate valid recommendations for this user.")
        except Exception as e:
            print(f"\nError generating recommendations: {str(e)}")
            print("  Could not generate valid recommendations for this user.")
    
    # Print summary
    print("\n" + "="*80)
    print(f"Successfully generated recommendations for {len(all_recommendations)} out of {len(indices)} users.")
    
    # Return the recommendations for potential further analysis
    return all_recommendations

def create_new_user_example(model, tokenizer, movie_data, tag_data=None, device=None):
    """
    Create and evaluate an example for a new user with specific movie preferences and tags
    """
    # Sample movies that have tags
    if tag_data is not None:
        tagged_movies = set(tag_data['movieId'].unique())
        movie_id_to_title = dict(zip(movie_data['movieId'], movie_data['title']))
        
        # Create a mapping of movie ID to its tags
        movie_tags = defaultdict(list)
        for _, row in tag_data.iterrows():
            movie_id = row['movieId']
            tag = row['tag']
            if pd.notna(tag) and str(tag).strip():
                movie_tags[movie_id].append(tag.lower().strip())
        
        # Select a few movies with tags for our example
        selected_movies = []
        selected_ids = []
        
        # Try to select movies with tags that are actually in our dataset
        for movie_id in tagged_movies:
            if movie_id in movie_id_to_title and movie_id in movie_tags:
                selected_ids.append(movie_id)
                
                title = movie_id_to_title[movie_id]
                rating = 5.0  # Let's say the user loved these movies
                tags_text = ", ".join(set(movie_tags[movie_id][:3]))  # Up to 3 tags
                
                movie_entry = f"{title} ({rating}) [tags: {tags_text}]"
                selected_movies.append(movie_entry)
                
                if len(selected_movies) >= 5:  # Limit to 5 movies
                    break
    
    # If we couldn't find tagged movies, use a default example
    if not selected_movies or tag_data is None:
        selected_movies = [
            "The Shawshank Redemption (1994) (5.0)",
            "The Godfather (1972) (5.0)",
            "The Dark Knight (2008) (4.5)",
            "Pulp Fiction (1994) (4.0)",
            "Inception (2010) (4.5)"
        ]
    
    # Create user history
    user_history = " | ".join(selected_movies)
    
    print("\n" + "="*80)
    print("Example for a new user with detailed preferences:")
    print(f"User history: {user_history}")
    
    # Generate recommendations
    recommendations = generate_recommendations(model, tokenizer, user_history, device=device)
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. {rec}")

def main():
    """
    Main function to run the training and evaluation
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    print("Loading MovieLens dataset...")
    ratings_df, movies_df, tags_df = load_and_prepare_data()
    
    # Create datasets
    train_dataset = MovieRecommendationDataset(
        ratings_df, movies_df, tags_df, split="train", min_ratings=10
    )
    val_dataset = MovieRecommendationDataset(
        ratings_df, movies_df, tags_df, split="validation", min_ratings=10
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Smaller batch size for memory efficiency
        shuffle=True, 
        collate_fn=lambda batch: lm_collate_fn(batch, device)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=lambda batch: lm_collate_fn(batch, device)
    )
    
    # Initialize model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model_config.n_classification_class = 1  # Dummy value, not used
    
    model = GPT(model_config)
    model = model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        num_epochs=13,
        device=device
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load("best_movie_recommender.pt"))
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    evaluate_model(model, train_dataset.tokenizer, val_dataset, num_samples=5, device=device)
    
    # Create example with tagged movies∂
    create_new_user_example(model, train_dataset.tokenizer, movies_df, tags_df, device=device)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()