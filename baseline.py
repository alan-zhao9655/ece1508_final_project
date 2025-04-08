import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

# ---------------------------------------
# Classes and Functions
# ---------------------------------------

class PopularityRecommender:
    """
    Simple baseline that recommends the most popular items
    """
    def __init__(self):
        self.popularity_scores = None
        
    def fit(self, train_matrix):
        """
        Calculate average rating for each movie
        """
        # Get sum of ratings and count of ratings per movie
        ratings_sum = np.sum(train_matrix, axis=0)
        ratings_count = np.sum(train_matrix > 0, axis=0)
        
        # Calculate average rating, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_ratings = np.divide(ratings_sum, ratings_count)
            
        # Replace NaN with 0
        avg_ratings = np.nan_to_num(avg_ratings)
        
        # Store the popularity scores
        self.popularity_scores = avg_ratings
        
    def recommend(self, user_id, train_matrix, top_n=10):
        """
        Recommend top-N most popular items that the user hasn't rated
        """
        if self.popularity_scores is None:
            raise ValueError("Model must be fit before recommending")
            
        # Get items the user has already rated
        user_ratings = train_matrix[user_id]
        already_rated = user_ratings > 0
        
        # Create a copy of the popularity scores
        scores = self.popularity_scores.copy()
        
        # Set already rated items' scores to -inf to exclude them
        scores[already_rated] = float('-inf')
        
        # Get top-N items
        top_item_indices = np.argsort(scores)[::-1][:top_n]
        
        return top_item_indices
        
    def evaluate(self, test_users, train_matrix, test_matrix, top_n=10):
        """
        Evaluate the model on test users
        """
        precision_list = []
        recall_list = []
        
        for user in test_users:
            # Get items in the test set for this user
            train_items = set(np.where(train_matrix[user] > 0)[0])
            test_items = set(np.where(test_matrix[user] > 0)[0]) - train_items
            
            if len(test_items) == 0:
                continue  # Skip users with no test items
                
            # Get recommended items
            recommended_items = set(self.recommend(user, train_matrix, top_n))
            
            # Calculate precision and recall
            num_relevant = len(recommended_items.intersection(test_items))
            precision = num_relevant / min(top_n, len(recommended_items)) if recommended_items else 0
            recall = num_relevant / len(test_items) if test_items else 0
            
            precision_list.append(precision)
            recall_list.append(recall)
            
        # Calculate average metrics
        avg_precision = np.mean(precision_list) if precision_list else 0
        avg_recall = np.mean(recall_list) if recall_list else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        }


class MatrixFactorizationDataset(Dataset):
    """Dataset for matrix factorization training"""
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return (self.user_ids[idx], self.item_ids[idx], self.ratings[idx])


class MatrixFactorization(nn.Module):
    """
    Basic matrix factorization model for collaborative filtering
    """
    def __init__(self, num_users, num_items, num_factors=20):
        super().__init__()
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids, item_ids):
        """
        Predict ratings for given user-item pairs
        """
        # Get embeddings and biases
        user_embedding = self.user_embedding(user_ids)
        item_embedding = self.item_embedding(item_ids)
        
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Calculate dot product
        dot_product = torch.sum(user_embedding * item_embedding, dim=1)
        
        # Add biases
        prediction = self.global_bias + user_b + item_b + dot_product
        
        return prediction


class FactorizationMachineDataset(Dataset):
    """Dataset for factorization machine training"""
    def __init__(self, user_ids, item_ids, item_genres, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.item_genres = item_genres
        self.ratings = ratings
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return (
            self.user_ids[idx], 
            self.item_ids[idx], 
            self.item_genres[self.item_ids[idx]], 
            self.ratings[idx]
        )


class FactorizationMachine(nn.Module):
    """
    Factorization Machine model that incorporates content features
    """
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=16):
        super().__init__()
        
        # Feature embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        
        # Bias terms
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.normal_(self.genre_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
    def forward(self, user_ids, movie_ids, movie_genres):
        """
        Forward pass of the factorization machine
        
        Parameters:
        user_ids: Tensor of user IDs
        movie_ids: Tensor of movie IDs
        movie_genres: Tensor of genre features for each movie [batch_size, num_genres]
        """
        # First order term (bias)
        bias = self.global_bias + self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()
        
        # User and movie embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Genre embeddings (weighted sum based on genre presence)
        batch_size = user_ids.size(0)
        genre_emb = torch.zeros(batch_size, self.genre_embedding.weight.size(1), device=user_ids.device)
        
        # For each genre that is present (1 in multi-hot), add its embedding
        for i in range(movie_genres.size(1)):
            genre_presence = movie_genres[:, i].float().unsqueeze(1)
            genre_emb += genre_presence * self.genre_embedding.weight[i].unshaped(0)
        
        # Alternative approach for genre embedding calculation
        # This is more efficient but might be harder to understand
        genre_emb = torch.matmul(movie_genres.float(), self.genre_embedding.weight)
        
        # Second order term (factorization machine)
        # Combine all feature embeddings
        all_embeddings = torch.stack([user_emb, movie_emb, genre_emb], dim=1)
        
        # Calculate sum of squares and square of sums (FM formula)
        summed_features = torch.sum(all_embeddings, dim=1)
        summed_squares = torch.sum(summed_features ** 2, dim=1)
        squared_sum = torch.sum(all_embeddings ** 2, dim=(1, 2))
        
        # FM interaction term
        fm_interaction = 0.5 * (summed_squares - squared_sum)
        
        # Final prediction
        prediction = bias + fm_interaction
        
        return prediction


# Fix for factorization machine forward pass - alternate implementation that's more robust
class FixedFactorizationMachine(nn.Module):
    """
    Simplified Factorization Machine model
    """
    def __init__(self, num_users, num_movies, num_genres, embedding_dim=16):
        super().__init__()
        
        # Feature embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Bias terms
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        # Genre projection layer
        self.genre_projection = nn.Linear(num_genres, embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.normal_(self.genre_projection.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
    def forward(self, user_ids, movie_ids, movie_genres):
        """
        Simplified FM forward pass
        """
        # First order term (bias)
        bias = self.global_bias + self.user_bias(user_ids).squeeze() + self.movie_bias(movie_ids).squeeze()
        
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        genre_emb = self.genre_projection(movie_genres.float())
        
        # Simple interaction terms
        um_interaction = torch.sum(user_emb * movie_emb, dim=1)
        ug_interaction = torch.sum(user_emb * genre_emb, dim=1)
        mg_interaction = torch.sum(movie_emb * genre_emb, dim=1)
        
        # Final prediction
        prediction = bias + um_interaction + ug_interaction + mg_interaction
        
        return prediction


def train_mf_model(model, train_loader, epochs=5, lr=0.005, weight_decay=0.01):
    """Train a matrix factorization model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Forward pass
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def train_fm_model(model, train_loader, epochs=5, lr=0.005, weight_decay=0.01):
    """Train a factorization machine model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for user_ids, item_ids, item_genres, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            item_genres = item_genres.to(device)
            ratings = ratings.to(device)
            
            # Forward pass
            outputs = model(user_ids, item_ids, item_genres)
            loss = criterion(outputs, ratings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def recommend_with_mf(model, user_id, train_matrix, num_movies, top_n=10):
    """Generate top-N recommendations using matrix factorization model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create tensor with all movie IDs
    all_movie_ids = torch.arange(num_movies).to(device)
    
    # Repeat user ID for each movie
    user_id_tensor = torch.full((num_movies,), user_id, dtype=torch.long).to(device)
    
    with torch.no_grad():
        # Get predictions for all movies
        predictions = model(user_id_tensor, all_movie_ids).cpu().numpy()
    
    # Get items the user has already rated
    user_ratings = train_matrix[user_id]
    already_rated = user_ratings > 0
    
    # Set already rated items to -inf
    predictions[already_rated] = float('-inf')
    
    # Get top-N items
    top_item_indices = np.argsort(predictions)[::-1][:top_n]
    
    return top_item_indices


def recommend_with_fm(model, user_id, train_matrix, num_movies, movie_genres, top_n=10):
    """Generate top-N recommendations using factorization machine model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create tensor with all movie IDs
    all_movie_ids = torch.arange(num_movies).to(device)
    
    # Repeat user ID for each movie
    user_id_tensor = torch.full((num_movies,), user_id, dtype=torch.long).to(device)
    
    # Get all movie genres
    all_movie_genres = torch.tensor(movie_genres).to(device)
    
    with torch.no_grad():
        # Get predictions for all movies
        predictions = model(user_id_tensor, all_movie_ids, all_movie_genres).cpu().numpy()
    
    # Get items the user has already rated
    user_ratings = train_matrix[user_id]
    already_rated = user_ratings > 0
    
    # Set already rated items to -inf
    predictions[already_rated] = float('-inf')
    
    # Get top-N items
    top_item_indices = np.argsort(predictions)[::-1][:top_n]
    
    return top_item_indices


def evaluate_mf_model(model, test_users, train_matrix, test_matrix, num_movies, top_n=10):
    """Evaluate matrix factorization model"""
    precision_list = []
    recall_list = []
    
    for user in test_users:
        # Get items in the test set for this user
        train_items = set(np.where(train_matrix[user] > 0)[0])
        test_items = set(np.where(test_matrix[user] > 0)[0]) - train_items
        
        if len(test_items) == 0:
            continue  # Skip users with no test items
            
        # Get recommended items
        recommended_items = set(recommend_with_mf(model, user, train_matrix, num_movies, top_n))
        
        # Calculate precision and recall
        num_relevant = len(recommended_items.intersection(test_items))
        precision = num_relevant / min(top_n, len(recommended_items)) if recommended_items else 0
        recall = num_relevant / len(test_items) if test_items else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        
    # Calculate average metrics
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    }


def evaluate_fm_model(model, test_users, train_matrix, test_matrix, num_movies, movie_genres, top_n=10):
    """Evaluate factorization machine model"""
    precision_list = []
    recall_list = []
    
    for user in test_users:
        # Get items in the test set for this user
        train_items = set(np.where(train_matrix[user] > 0)[0])
        test_items = set(np.where(test_matrix[user] > 0)[0]) - train_items
        
        if len(test_items) == 0:
            continue  # Skip users with no test items
            
        # Get recommended items
        recommended_items = set(recommend_with_fm(model, user, train_matrix, num_movies, movie_genres, top_n))
        
        # Calculate precision and recall
        num_relevant = len(recommended_items.intersection(test_items))
        precision = num_relevant / min(top_n, len(recommended_items)) if recommended_items else 0
        recall = num_relevant / len(test_items) if test_items else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        
    # Calculate average metrics
    avg_precision = np.mean(precision_list) if precision_list else 0
    avg_recall = np.mean(recall_list) if recall_list else 0
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    }


def get_movie_title(movie_idx, movie2idx, movies_df):
    """Get movie title from index"""
    original_movie_id = list(movie2idx.keys())[list(movie2idx.values()).index(movie_idx)]
    movie_row = movies_df[movies_df['movieId'] == original_movie_id]
    return movie_row['title'].values[0] if not movie_row.empty else f"Unknown (ID {original_movie_id})"


def load_and_preprocess_data(data_folder="ml-latest-small"):
    """Load and preprocess the MovieLens dataset"""
    ratings_file = f"{data_folder}/ratings.csv"
    movies_file = f"{data_folder}/movies.csv"
    tags_file = f"{data_folder}/tags.csv"

    # Read CSVs
    ratings_df = pd.read_csv(ratings_file)
    movies_df = pd.read_csv(movies_file)
    tags_df = pd.read_csv(tags_file)

    # Print data shapes
    print("Ratings shape:", ratings_df.shape)
    print("Movies shape:", movies_df.shape)

    # Re-index userIds and movieIds for PyTorch embedding usage
    unique_user_ids = ratings_df['userId'].unique()
    unique_movie_ids = ratings_df['movieId'].unique()

    # Create maps: userId -> new_user_index
    user2idx = {old_id: idx for idx, old_id in enumerate(unique_user_ids)}
    movie2idx = {old_id: idx for idx, old_id in enumerate(unique_movie_ids)}

    # Apply these maps so that userId and movieId become 0-based indices
    ratings_df['userId'] = ratings_df['userId'].map(user2idx)
    ratings_df['movieId'] = ratings_df['movieId'].map(movie2idx)

    num_users = len(unique_user_ids)
    num_movies = len(unique_movie_ids)

    print("Number of users:", num_users)
    print("Number of movies:", num_movies)
    
    # Create rating matrix
    rating_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    for row in ratings_df.itertuples():
        rating_matrix[row.userId, row.movieId] = row.rating
    
    print("Rating matrix shape:", rating_matrix.shape)
    
    # Process genres to create a multi-hot encoding
    # Extract all genres from the dataset
    all_genres = set()
    for genres in movies_df['genres'].str.split('|'):
        if isinstance(genres, list):  # Handle NaN values
            all_genres.update(genres)
    
    if '(no genres listed)' in all_genres:
        all_genres.remove('(no genres listed)')
    
    genre_to_idx = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
    num_genres = len(genre_to_idx)
    print(f"Number of unique genres: {num_genres}")
    
    # Create a multi-hot encoding for each movie
    movie_genres = np.zeros((num_movies, num_genres), dtype=np.float32)
    
    for row in movies_df.itertuples():
        if row.movieId in movie2idx:  # Check if movie is in our mapping
            movie_idx = movie2idx[row.movieId]
            genres = row.genres.split('|') if row.genres != '(no genres listed)' else []
            for genre in genres:
                if genre in genre_to_idx:  # Some genres might be missing
                    genre_idx = genre_to_idx[genre]
                    movie_genres[movie_idx, genre_idx] = 1.0
    
    return (ratings_df, movies_df, tags_df, rating_matrix, 
            user2idx, movie2idx, genre_to_idx, num_users, num_movies, num_genres, movie_genres)


# ---------------------------------------
# Main Function
# ---------------------------------------

def main():
    start_time = time.time()
    
    # Load and preprocess data
    (ratings_df, movies_df, tags_df, rating_matrix, 
     user2idx, movie2idx, genre_to_idx, num_users, num_movies, 
     num_genres, movie_genres) = load_and_preprocess_data()
    
    print(f"Data loading and preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    # Split data for training and evaluation
    all_users = np.arange(num_users)
    train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)
    
    # Create train and test matrices
    train_matrix = rating_matrix.copy()
    test_matrix = rating_matrix.copy()
    
    # For test users, hold out 20% of their ratings for evaluation
    for user in test_users:
        rated_items = np.where(rating_matrix[user] > 0)[0]
        if len(rated_items) > 5:  # Make sure user has rated enough items
            n_holdout = max(1, int(len(rated_items) * 0.2))
            holdout_items = np.random.choice(rated_items, size=n_holdout, replace=False)
            train_matrix[user, holdout_items] = 0  # Remove from training
    
    print(f"Data splitting completed in {time.time() - start_time:.2f} seconds")
    
    # Prepare the training data for matrix factorization
    users, items, ratings = [], [], []
    for u in range(num_users):
        for i in range(num_movies):
            if train_matrix[u, i] > 0:
                users.append(u)
                items.append(i)
                ratings.append(train_matrix[u, i])
    
    # Convert to tensors
    users = torch.tensor(users, dtype=torch.long)
    items = torch.tensor(items, dtype=torch.long)
    ratings = torch.tensor(ratings, dtype=torch.float)
    
    # Create dataset and dataloader for MF
    train_dataset = MatrixFactorizationDataset(users, items, ratings)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create dataset and dataloader for FM
    fm_dataset = FactorizationMachineDataset(users, items, torch.tensor(movie_genres), ratings)
    fm_loader = DataLoader(fm_dataset, batch_size=64, shuffle=True)
    
    print("\n========= Training and Evaluating Recommendation Models =========")
    
    # Train and evaluate popularity model
    print("\nTraining Popularity-Based Model...")
    pop_model = PopularityRecommender()
    pop_model.fit(train_matrix)
    pop_metrics = pop_model.evaluate(test_users, train_matrix, test_matrix)
    
    print(f"Popularity Model Metrics: Precision={pop_metrics['precision']:.4f}, Recall={pop_metrics['recall']:.4f}, F1={pop_metrics['f1']:.4f}")
    print(f"Popularity model evaluation completed in {time.time() - start_time:.2f} seconds")
    
    # Train and evaluate matrix factorization model
    print("\nTraining Matrix Factorization Model...")
    mf_model = MatrixFactorization(num_users, num_movies, num_factors=20)
    mf_model = train_mf_model(mf_model, train_loader, epochs=7)
    mf_metrics = evaluate_mf_model(mf_model, test_users, train_matrix, test_matrix, num_movies)
    
    print(f"Matrix Factorization Metrics: Precision={mf_metrics['precision']:.4f}, Recall={mf_metrics['recall']:.4f}, F1={mf_metrics['f1']:.4f}")
    print(f"Matrix Factorization model evaluation completed in {time.time() - start_time:.2f} seconds")
    
    # Train and evaluate factorization machine model
    print("\nTraining Factorization Machine Model...")
    # Use the simplified version to avoid potential errors
    fm_model = FixedFactorizationMachine(num_users, num_movies, num_genres, embedding_dim=32)
    fm_model = train_fm_model(fm_model, fm_loader, epochs=7)
    fm_metrics = evaluate_fm_model(fm_model, test_users, train_matrix, test_matrix, num_movies, movie_genres)
    
    print(f"Factorization Machine Metrics: Precision={fm_metrics['precision']:.4f}, Recall={fm_metrics['recall']:.4f}, F1={fm_metrics['f1']:.4f}")
    print(f"Factorization Machine model evaluation completed in {time.time() - start_time:.2f} seconds")
    
    # Print sample recommendations for a few users
    print("\n========= Sample Recommendations =========")
    
    sample_users = np.random.choice(test_users, size=3, replace=False)
    
    for user in sample_users:
        original_user_id = list(user2idx.keys())[list(user2idx.values()).index(user)]
        print(f"\nUser {original_user_id}")
        
        print("\nPopularity-based recommendations:")
        for i, movie_idx in enumerate(pop_model.recommend(user, train_matrix, top_n=5)):
            print(f"  {i+1}. {get_movie_title(movie_idx, movie2idx, movies_df)}")
        
        print("\nMatrix Factorization recommendations:")
        for i, movie_idx in enumerate(recommend_with_mf(mf_model, user, train_matrix, num_movies, top_n=5)):
            print(f"  {i+1}. {get_movie_title(movie_idx, movie2idx, movies_df)}")
        
        print("\nFactorization Machine recommendations:")
        for i, movie_idx in enumerate(recommend_with_fm(fm_model, user, train_matrix, num_movies, movie_genres, top_n=5)):
            print(f"  {i+1}. {get_movie_title(movie_idx, movie2idx, movies_df)}")
        
        print("\nActually rated movies by this user:")
        for movie_idx in np.where(rating_matrix[user] > 0)[0][:5]:  # Show first 5 rated movies
            print(f"  - {get_movie_title(movie_idx, movie2idx, movies_df)} (Rating: {rating_matrix[user, movie_idx]})")
    
    # Print overall comparison
    print("\n========= Model Comparison Summary =========")
    print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 60)
    print(f"{'Popularity Model':<25} {pop_metrics['precision']:<12.4f} {pop_metrics['recall']:<12.4f} {pop_metrics['f1']:<12.4f}")
    print(f"{'Matrix Factorization':<25} {mf_metrics['precision']:<12.4f} {mf_metrics['recall']:<12.4f} {mf_metrics['f1']:<12.4f}")
    print(f"{'Factorization Machine':<25} {fm_metrics['precision']:<12.4f} {fm_metrics['recall']:<12.4f} {fm_metrics['f1']:<12.4f}")
    print("-" * 60)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("=============================================")

if __name__ == "__main__":
    main()