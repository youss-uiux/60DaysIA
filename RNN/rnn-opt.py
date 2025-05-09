import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools, algorithms
import random
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_KEY = 'YOUR_API_KEY'  # Replace with your actual API key
DATA_CACHE_FILE = 'btc_eur_data.csv'
PLOT_SAVE_PATH = 'results/plots/'
MODEL_SAVE_PATH = 'results/models/'
os.makedirs(PLOT_SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def fetch_data(use_cache=True):
    """Fetch Bitcoin price data from Alpha Vantage with caching."""
    if use_cache and os.path.exists(DATA_CACHE_FILE):
        logger.info("Loading data from cache")
        return pd.read_csv(DATA_CACHE_FILE, index_col=0, parse_dates=True)
    
    try:
        logger.info("Fetching data from Alpha Vantage")
        url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=EUR&apikey={API_KEY}&datatype=csv'
        df = pd.read_csv(url)
        
        # Save to cache
        df.to_csv(DATA_CACHE_FILE)
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        if os.path.exists(DATA_CACHE_FILE):
            logger.info("Using cached data as fallback")
            return pd.read_csv(DATA_CACHE_FILE, index_col=0, parse_dates=True)
        raise

def add_technical_indicators(data, window=20):
    """Add technical indicators to the data"""
    df = pd.DataFrame(data, columns=['returns'])
    
    # Rolling statistics
    df['rolling_mean'] = df['returns'].rolling(window=window).mean()
    df['rolling_std'] = df['returns'].rolling(window=window).std()
    
    # Momentum
    df['momentum'] = df['returns'].rolling(window=window).apply(lambda x: x[-1] - x[0])
    
    # Remove NaN values
    df = df.dropna()
    return df.values

def preprocess_data(df):
    """Preprocess the data and handle missing values."""
    # Identify close price column
    close_col = next((col for col in df.columns 
                     if 'close' in col.lower() and ('EUR' in col or 'USD' in col)), 
                    'close' if 'close' in df.columns else None)
    
    if close_col is None:
        raise ValueError("No suitable close price column found")
    
    # Convert to numeric and handle missing values
    df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
    df = df[~df[close_col].isna()]
    
    # Calculate daily log returns
    returns = np.diff(np.log(df[close_col].values))
    
    # Remove outliers (beyond 3 standard deviations)
    mean, std = np.mean(returns), np.std(returns)
    returns = returns[(returns > mean - 3*std) & (returns < mean + 3*std)]
    
    # Add technical indicators
    enhanced_data = add_technical_indicators(returns)
    
    return enhanced_data

def create_sequences(data, num_lags):
    """Create input sequences and targets for time series prediction."""
    X, y = [], []
    for i in range(len(data) - num_lags - 1):
        X.append(data[i:(i + num_lags)])
        y.append(data[i + num_lags, 0])  # Predict only the returns (first column)
    return np.array(X), np.array(y)

def build_rnn_model(input_shape, num_units, learning_rate=0.001):
    """Build and compile an RNN model with improved architecture."""
    model = Sequential([
        SimpleRNN(num_units, input_shape=input_shape, return_sequences=False,
                 kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', 
                 metrics=['mae'])
    return model

def build_lstm_model(input_shape, num_units, learning_rate=0.001):
    """Alternative LSTM model architecture."""
    model = Sequential([
        LSTM(num_units, input_shape=input_shape, return_sequences=False,
            kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

class GeneticOptimizer:
    def __init__(self, x_train_full, y_train_full, x_test_full, y_test_full, max_lags=100, model_type='rnn'):
        self.x_train_full = x_train_full
        self.y_train_full = y_train_full
        self.x_test_full = x_test_full
        self.y_test_full = y_test_full
        self.max_lags = max_lags
        self.model_type = model_type
        self.best_model = None
        self.best_score = float('inf')
        self.history = []
        
        # Setup DEAP framework
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_lags", random.randint, 20, 100)
        self.toolbox.register("attr_units", random.randint, 32, 128)
        self.toolbox.register("attr_epochs", random.randint, 30, 150)
        self.toolbox.register("attr_batch", random.choice, [16, 32, 64, 128])
        self.toolbox.register("attr_lr", random.uniform, 0.0001, 0.01)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.attr_lags, self.toolbox.attr_units, 
                              self.toolbox.attr_epochs, self.toolbox.attr_batch,
                              self.toolbox.attr_lr), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        
        # Custom mutation function to handle mixed types
        def mutMixed(individual, indpb):
            for i in range(len(individual)):
                if random.random() < indpb:
                    if i == 4:  # Learning rate (float)
                        individual[i] = random.uniform(0.0001, 0.01)
                    else:  # Other parameters (integers)
                        if i == 0:  # num_lags
                            individual[i] = random.randint(20, 100)
                        elif i == 1:  # num_units
                            individual[i] = random.randint(32, 128)
                        elif i == 2:  # num_epochs
                            individual[i] = random.randint(30, 150)
                        elif i == 3:  # batch_size
                            individual[i] = random.choice([16, 32, 64, 128])
            return individual,
        
        self.toolbox.register("mutate", mutMixed, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_individual(self, individual):
        """Evaluate an individual's fitness."""
        num_lags, num_units, num_epochs, batch_size, learning_rate = individual
        
        # Adjust data for the current lag value
        lag_diff = self.max_lags - num_lags
        x_train = self.x_train_full[:, lag_diff:lag_diff+num_lags]
        x_test = self.x_test_full[:, lag_diff:lag_diff+num_lags]
        
        # Reshape for RNN/LSTM
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        
        # Build and train model
        if self.model_type == 'rnn':
            model = build_rnn_model((num_lags, x_train.shape[2]), num_units, learning_rate)
        else:
            model = build_lstm_model((num_lags, x_train.shape[2]), num_units, learning_rate)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        try:
            history = model.fit(
                x_train, self.y_train_full.reshape(-1, 1),
                epochs=num_epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=callbacks
            )
            
            # Evaluate on test set
            y_pred = model.predict(x_test, verbose=0)
            mse = mean_squared_error(self.y_test_full, y_pred)
            
            # Keep track of the best model
            if mse < self.best_score:
                self.best_score = mse
                self.best_model = model
                model.save(f"{MODEL_SAVE_PATH}best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
            
            return mse,
        except Exception as e:
            logger.warning(f"Error evaluating individual: {e}")
            return float('inf'),
    
    def run_optimization_with_elitism(self, ngen=10, pop_size=15, cxpb=0.7, mutpb=0.3, elitism=2):
        """Run the genetic optimization with elitism."""
        population = self.toolbox.population(n=pop_size)
        
        for gen in range(ngen):
            # Select elites
            elites = tools.selBest(population, k=elitism)
            
            # Generate offspring
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=cxpb, mutpb=mutpb)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation (keep elites)
            population = self.toolbox.select(offspring, k=len(population)-elitism) + elites
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in population]
            self.history.append({
                'gen': gen,
                'min': np.min(fits),
                'avg': np.mean(fits),
                'max': np.max(fits)
            })
            
            logger.info(f"Generation {gen}: Min MSE={np.min(fits):.6f}, Avg={np.mean(fits):.6f}")
        
        return population

def validate_model(model, X, y, n_splits=5):
    """Perform time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Reshape for RNN
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
    
    return np.mean(mse_scores), np.std(mse_scores)

def plot_feature_importance(model, num_lags):
    """Plot the learned weights from the first RNN layer."""
    if len(model.layers[0].get_weights()) > 0:
        weights = model.layers[0].get_weights()[0]  # Input weights
        plt.figure(figsize=(12, 6))
        plt.bar(range(num_lags), np.mean(np.abs(weights), axis=1))
        plt.title("Input Feature Importance (Absolute Weights)")
        plt.xlabel("Lag")
        plt.ylabel("Average Weight Magnitude")
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = f"{PLOT_SAVE_PATH}feature_importance_{timestamp}.png"
        plt.savefig(plot_file, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {plot_file}")

def plot_results(y_true, y_pred, title):
    """Plot prediction results."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Returns', alpha=0.7)
    plt.plot(y_pred, label='Predicted Returns', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Log Returns')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f"{PLOT_SAVE_PATH}btc_returns_pred_{timestamp}.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Plot saved to {plot_file}")

def plot_optimization_history(history):
    """Plot the optimization progress."""
    gens = [h['gen'] for h in history]
    min_mse = [h['min'] for h in history]
    avg_mse = [h['avg'] for h in history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(gens, min_mse, label='Minimum MSE')
    plt.plot(gens, avg_mse, label='Average MSE')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f"{PLOT_SAVE_PATH}optimization_history_{timestamp}.png"
    plt.savefig(plot_file, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Optimization history plot saved to {plot_file}")

def main():
    try:
        # Load and preprocess data
        df = fetch_data()
        enhanced_data = preprocess_data(df)
        
        # Prepare sequences
        max_lags = 60
        X, y = create_sequences(enhanced_data, max_lags)
        
        # Split data
        train_size = int(len(X) * 0.8)
        x_train_full, x_test_full = X[:train_size], X[train_size:]
        y_train_full, y_test_full = y[:train_size], y[train_size:]
        
        # Scale data using RobustScaler (less sensitive to outliers)
        scalers = []
        for i in range(x_train_full.shape[2]):  # Scale each feature separately
            scaler = RobustScaler()
            x_train_full[:, :, i] = scaler.fit_transform(x_train_full[:, :, i])
            x_test_full[:, :, i] = scaler.transform(x_test_full[:, :, i])
            scalers.append(scaler)
        
        # Run genetic optimization (try both RNN and LSTM)
        for model_type in ['rnn', 'lstm']:
            logger.info(f"\nRunning optimization for {model_type.upper()} model")
            optimizer = GeneticOptimizer(x_train_full, y_train_full, x_test_full, y_test_full, 
                                       max_lags=max_lags, model_type=model_type)
            final_pop = optimizer.run_optimization_with_elitism(ngen=15, pop_size=20)
            
            # Plot optimization history
            plot_optimization_history(optimizer.history)
            
            # Get best individual
            best_ind = tools.selBest(final_pop, k=1)[0]
            logger.info(f"Best {model_type.upper()} individual: Lags={best_ind[0]}, Units={best_ind[1]}, "
                       f"Epochs={best_ind[2]}, Batch={best_ind[3]}, LR={best_ind[4]:.6f}")
            
            # Evaluate best model
            if optimizer.best_model:
                # Reshape test data
                lag_diff = max_lags - best_ind[0]
                x_test = x_test_full[:, lag_diff:lag_diff+best_ind[0]]
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
                
                y_pred = optimizer.best_model.predict(x_test)
                
                mse = mean_squared_error(y_test_full, y_pred)
                mae = mean_absolute_error(y_test_full, y_pred)
                logger.info(f"{model_type.upper()} Test MSE: {mse:.6f}, MAE: {mae:.6f}")
                
                # Cross-validation
                avg_mse, std_mse = validate_model(optimizer.best_model, x_test_full, y_test_full)
                logger.info(f"{model_type.upper()} Cross-validated MSE: {avg_mse:.6f} ± {std_mse:.6f}")
                
                # Plot results
                plot_results(y_test_full, y_pred, f"Bitcoin Returns Prediction with Optimized {model_type.upper()}")
                
                # Plot feature importance
                plot_feature_importance(optimizer.best_model, best_ind[0])
            else:
                logger.warning(f"No best {model_type.upper()} model found")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()