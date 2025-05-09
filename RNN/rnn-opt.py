import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
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
    
    # Calculate daily returns
    returns = np.diff(np.log(df[close_col].values))
    
    # Remove outliers (beyond 3 standard deviations)
    mean, std = np.mean(returns), np.std(returns)
    returns = returns[(returns > mean - 3*std) & (returns < mean + 3*std)]
    
    return returns

def create_sequences(data, num_lags):
    """Create input sequences and targets for time series prediction."""
    X, y = [], []
    for i in range(len(data) - num_lags - 1):
        X.append(data[i:(i + num_lags)])
        y.append(data[i + num_lags])
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

class GeneticOptimizer:
    def __init__(self, x_train_full, y_train_full, x_test_full, y_test_full, max_lags=100):
        self.x_train_full = x_train_full
        self.y_train_full = y_train_full
        self.x_test_full = x_test_full
        self.y_test_full = y_test_full
        self.max_lags = max_lags
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
        
        # Reshape for RNN
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
        # Build and train model
        model = build_rnn_model((num_lags, 1), num_units, learning_rate)
        
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
    
    def run_optimization(self, ngen=10, pop_size=15, cxpb=0.7, mutpb=0.3):
        """Run the genetic optimization."""
        population = self.toolbox.population(n=pop_size)
        
        for gen in range(ngen):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=cxpb, mutpb=mutpb)
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Select next generation
            population = self.toolbox.select(offspring, k=len(population))
            
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

def main():
    try:
        # Load and preprocess data
        df = fetch_data()
        returns = preprocess_data(df)
        
        # Prepare sequences
        max_lags = 60
        X, y = create_sequences(returns, max_lags)
        
        # Split data
        train_size = int(len(X) * 0.8)
        x_train_full, x_test_full = X[:train_size], X[train_size:]
        y_train_full, y_test_full = y[:train_size], y[train_size:]
        
        # Scale data using RobustScaler (less sensitive to outliers)
        scaler = RobustScaler()
        x_train_full = scaler.fit_transform(x_train_full)
        x_test_full = scaler.transform(x_test_full)
        
        # Run genetic optimization
        optimizer = GeneticOptimizer(x_train_full, y_train_full, x_test_full, y_test_full, max_lags)
        final_pop = optimizer.run_optimization(ngen=8, pop_size=12)
        
        # Get best individual
        best_ind = tools.selBest(final_pop, k=1)[0]
        logger.info(f"Best individual: Lags={best_ind[0]}, Units={best_ind[1]}, "
                   f"Epochs={best_ind[2]}, Batch={best_ind[3]}, LR={best_ind[4]:.6f}")
        
        # Evaluate best model
        if optimizer.best_model:
            y_pred = optimizer.best_model.predict(
                x_test_full.reshape((x_test_full.shape[0], x_test_full.shape[1], 1))
            )
            
            mse = mean_squared_error(y_test_full, y_pred)
            mae = mean_absolute_error(y_test_full, y_pred)
            logger.info(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
            
            # Plot results
            plot_results(y_test_full, y_pred, "Bitcoin Daily Returns Prediction with Optimized RNN")
        else:
            logger.warning("No best model found")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()