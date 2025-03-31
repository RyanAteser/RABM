import os
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
from statsmodels.tsa.stattools import acf
import seaborn as sns
from tqdm import tqdm
import yfinance as yf


class ReflexivityBrownianMotion:
    """
    A comprehensive model of asset price dynamics that incorporates feedback loops
    into standard Brownian motion, implementing the Reflexivity-Adjusted Brownian
    Motion (RABM) framework.

    This model captures:
    1. Direct feedback from past returns to current drift
    2. Regime-switching behavior influenced by price changes
    3. Scale-dependent memory effects through a varying Hurst exponent
    4. Self-exciting jump processes that create clustering of market events
    """

    def __init__(self,
                 S0=100,               # Initial price
                 mu_base=0.05,         # Base drift (annualized)
                 sigma_base=0.15,       # Base volatility (annualized)
                 dt=1/252,             # Time step (1 trading day)
                 alpha=2.0,            # Feedback strength
                 tau=20,               # Memory length (in time steps)
                 gamma=.5,            # Reflexivity parameter (regime sensitivity to price changes)
                 kappa=3.0,            # Regime mean-reversion speed
                 R_bar=0.0,            # Long-term average regime
                 eta=0.5,              # Regime volatility
                 H0=0.6,               # Baseline Hurst exponent
                 delta=0.8,            # Sensitivity of Hurst exponent to autocorrelation
                 lambda0=0.05,          # Baseline jump intensity
                 alpha_J=0.8,          # Jump self-excitation parameter
                 beta=0.2,             # Decay rate of self-excitation
                 jump_mean=0.005,      # Average jump size (negative for crashes)
                 jump_std=0.02,        # Jump size standard deviation
                 rho_WB=0.7            # Correlation between main and regime Brownian motions
                 ):
        # Store parameters
        self.S0 = S0
        self.mu_base = mu_base
        self.sigma_base = sigma_base
        self.dt = dt
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.kappa = kappa
        self.R_bar = R_bar
        self.eta = eta
        self.H0 = H0
        self.delta = delta
        self.lambda0 = lambda0
        self.alpha_J = alpha_J
        self.beta = beta
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.rho_WB = rho_WB

        # Initialize state variables
        self.S = S0                     # Current price
        self.R = R_bar                  # Current regime
        self.H = H0                     # Current Hurst exponent
        self.lambda_t = lambda0         # Current jump intensity
        self.returns_history = []       # History of returns for calculating feedback
        self.price_history = [S0]       # History of prices
        self.regime_history = [R_bar]   # History of regimes
        self.hurst_history = [H0]       # History of Hurst exponents
        self.intensity_history = [lambda0]  # History of jump intensities
        self.jump_times = []            # Times when jumps occurred
        self.time = 0                   # Current time

        # Precompute correlated random variables for efficiency
        self.corr_matrix = np.array([[1.0, self.rho_WB],
                                     [self.rho_WB, 1.0]])

    def mu(self, regime):
        """Calculate drift based on current regime."""
        return self.mu_base * (1 + regime)

    def sigma(self, regime):
        """Calculate volatility based on current regime."""
        return self.sigma_base * (1 + 0.5 * abs(regime))

    def calculate_feedback(self):
        """Calculate the feedback term from past returns."""
        if len(self.returns_history) < self.tau:
            return 0.0

        # Use the most recent tau returns
        recent_returns = self.returns_history[-self.tau:]
        return self.alpha * sum(recent_returns)

    def calculate_autocorrelation(self, lag=1):
        """Calculate autocorrelation of returns at specified lag."""
        if len(self.returns_history) <= lag + 1:
            return 0.0

        recent_returns = np.array(self.returns_history[-30:])  # Use last 30 observations
        if len(recent_returns) <= lag + 1:
            return 0.0

        if np.std(recent_returns) == 0:
            return 0.0

        # Calculate autocorrelation manually to avoid potential issues
        n = len(recent_returns)
        mean = np.mean(recent_returns)
        ac = 0
        variance = np.sum((recent_returns - mean) ** 2) / n

        if variance == 0:
            return 0.0

        for i in range(n - lag):
            ac += (recent_returns[i] - mean) * (recent_returns[i + lag] - mean)

        ac /= (n - lag) * variance
        return max(min(ac, 0.99), -0.99)  # Ensure value is in valid range

    def update_hurst_exponent(self):
        """Update the Hurst exponent based on current autocorrelation."""
        ac = self.calculate_autocorrelation(lag=1)
        self.H = max(0.01, min(0.99, self.H0 + self.delta * ac))
        return self.H

    def update_jump_intensity(self):
        """Update jump intensity based on past events (Hawkes process)."""
        # Decay the existing intensity
        self.lambda_t = self.lambda0

        # Add contribution from past jumps
        for jump_time in self.jump_times:
            time_since_jump = self.time - jump_time
            if time_since_jump > 0:
                self.lambda_t += self.alpha_J * np.exp(-self.beta * time_since_jump)

        return self.lambda_t

    def generate_fractional_brownian_increment(self, H):
        """
        Generate an increment of fractional Brownian motion with Hurst exponent H.

        For simplicity, we approximate this by scaling a standard normal random variable.
        In a more sophisticated implementation, true fBm should be used.
        """
        # This is a simplified approach - a true fBm would require more complex simulation
        Z = np.random.normal(0, 1)

        # Scale to approximate the behavior of fBm
        if H > 0.5:  # Persistent (trending)
            # Increase the probability of continuing in the same direction
            if len(self.returns_history) > 0:
                last_sign = np.sign(self.returns_history[-1])
                # Increase probability of same sign
                p_same = 0.5 + (H - 0.5)
                if np.random.random() < p_same:
                    Z = abs(Z) * last_sign
                else:
                    Z = -abs(Z) * last_sign
        elif H < 0.5:  # Anti-persistent (mean-reverting)
            # Increase the probability of reversing direction
            if len(self.returns_history) > 0:
                last_sign = np.sign(self.returns_history[-1])
                # Increase probability of opposite sign
                p_opposite = 0.5 + (0.5 - H)
                if np.random.random() < p_opposite:
                    Z = -abs(Z) * last_sign
                else:
                    Z = abs(Z) * last_sign

        return Z * np.sqrt(self.dt)

    def generate_correlated_normals(self, correlation_matrix=None):
        """Generate correlated normal random variables."""
        if correlation_matrix is None:
            correlation_matrix = self.corr_matrix

        L = np.linalg.cholesky(correlation_matrix)
        uncorrelated = np.random.normal(0, 1, size=2)
        correlated = np.dot(L, uncorrelated)
        return correlated

    def advanced_autocorrelation_model(self):
        """
        More sophisticated autocorrelation modeling
        """
        weighted_autocorr = []
        # Extended memory mechanism
        if len(self.returns_history) > self.tau:
            # Calculate multi-lag autocorrelation
            recent_returns = np.array(self.returns_history[-self.tau:])

            # Compute weighted autocorrelation
            lag_weights = np.linspace(1, 0, self.tau)
            weighted_autocorr = [
                np.corrcoef(recent_returns[:-i], recent_returns[i:])[0, 1]
                for i in range(1, min(5, self.tau))
            ]

            # Adaptive parameter adjustment
            decay_factor = np.mean(np.abs(weighted_autocorr))
            self.alpha *= (1 + decay_factor)
            self.delta *= (1 - decay_factor)

        return weighted_autocorr

    def generate_fat_tail_increment(self):
        """
        Generate return increments with enhanced fat-tail characteristics
        Using Student's t-distribution for more extreme events
        """
        # Degrees of freedom parameter (lower = fatter tails)
        nu = 4.0  # Adjust based on empirical observation

        # Generate t-distributed increment
        from scipy import stats
        Z = stats.t.rvs(nu)

        # Scale and adjust based on current regime
        tail_factor = 1 + abs(self.R)  # Regime influences tail thickness

        return Z * self.sigma(self.R) * np.sqrt(tail_factor) * self.dt

    def update_skewness_capture(self):
        """
        Dynamically adjust model parameters to better capture skewness

        Returns:
        --------
        float: Calculated skewness or 0 if insufficient data
        """
    # Initialize skewness to 0
        current_skewness = 0.0

        # Calculate recent returns skewness if possible
        if len(self.returns_history) >= self.tau:
            recent_returns = self.returns_history[-self.tau:]
            current_skewness = skew(recent_returns)

            # Adaptive parameter adjustment
            if current_skewness < 0:
                # Increase probability of negative jumps
                self.jump_mean = -abs(self.jump_mean)
                self.jump_std *= 1.1  # Increase jump volatility
            else:
                # Increase probability of positive jumps
                self.jump_mean = abs(self.jump_mean)
                self.jump_std *= 1.1

        return current_skewness

    def step(self):
        """Advance the simulation by one time step."""
        # Calculate current parameters
        mu_t = self.mu(self.R)
        sigma_t = self.sigma(self.R)
        feedback = self.calculate_feedback()

        # Update Hurst exponent based on current autocorrelation
        self.update_hurst_exponent()

        # Update jump intensity
        self.update_jump_intensity()


        # Generate correlated Brownian increments for price and regime
        dW, dB = self.generate_correlated_normals() * np.sqrt(self.dt)

        # Replace standard Brownian with fractional Brownian for price
        dW = self.generate_fractional_brownian_increment(self.H)

        # Determine if a jump occurs
        jump_probability = self.lambda_t * self.dt
        jump_occurs = np.random.random() < jump_probability

        jump_size = 0
        if jump_occurs:
            jump_size = np.random.normal(self.jump_mean, self.jump_std)
            self.jump_times.append(self.time)

        # Calculate return for this step
        current_return = mu_t * self.dt + feedback * self.dt + sigma_t * dW + jump_size

        # Update price
        self.S = self.S * (1 + current_return)

        # Update regime
        dR = self.kappa * (self.R_bar - self.R) * self.dt + self.gamma * current_return + self.eta * dB
        self.R = self.R + dR

        # Store history
        self.returns_history.append(current_return)
        self.price_history.append(self.S)
        self.regime_history.append(self.R)
        self.hurst_history.append(self.H)
        self.intensity_history.append(self.lambda_t)

        # Increment time
        self.time += self.dt


        return current_return
    def predict(self, n_steps):
        """
        Generate predictions for a specified number of steps.

        Parameters:
        -----------
        n_steps : int
            Number of steps to predict

        Returns:
        --------
        dict : Dictionary containing prediction results
        """
        # Simulate the process
        results = self.simulate(n_steps)

        return {
            'price': results['price'].values,
            'returns': results['returns'].values,
            'regime': results['regime'].values
        }
    def simulate(self, n_steps):
        """Simulate the price process for n_steps."""
        # Reset state for a fresh simulation
        self.S = self.S0
        self.R = self.R_bar
        self.H = self.H0
        self.lambda_t = self.lambda0
        self.returns_history = []
        self.price_history = [self.S0]
        self.regime_history = [self.R_bar]
        self.hurst_history = [self.H0]
        self.intensity_history = [self.lambda0]
        self.jump_times = []
        self.time = 0

        # Run simulation
        for _ in tqdm(range(n_steps), desc="Simulating"):
            self.step()

        # Create a DataFrame with the results
        results = pd.DataFrame({
            'price': self.price_history,
            'returns': [0] + self.returns_history,  # Add 0 as first return for alignment
            'regime': self.regime_history,
            'hurst': self.hurst_history,
            'intensity': self.intensity_history
        })

        # Add time index
        results['time'] = np.arange(0, len(self.price_history)) * self.dt
        results.set_index('time', inplace=True)

        return results

    def plot_simulation(self, results=None, n_steps=1000):
        """Plot the results of a simulation."""
        if results is None:
            results = self.simulate(n_steps)

        fig, axs = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

        # Price
        axs[0].plot(results.index, results['price'])
        axs[0].set_title('Price')
        axs[0].set_ylabel('Price')
        axs[0].grid(True)

        # Returns
        axs[1].plot(results.index, results['returns'], 'o-', markersize=2, alpha=0.7)
        axs[1].set_title('Returns')
        axs[1].set_ylabel('Return')
        axs[1].grid(True)

        # Regime
        axs[2].plot(results.index, results['regime'])
        axs[2].set_title('Regime')
        axs[2].set_ylabel('Regime Value')
        axs[2].grid(True)

        # Hurst Exponent
        axs[3].plot(results.index, results['hurst'])
        axs[3].set_title('Hurst Exponent')
        axs[3].set_ylabel('H')
        axs[3].axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        axs[3].grid(True)

        # Jump Intensity
        axs[4].plot(results.index, results['intensity'])
        axs[4].set_title('Jump Intensity')
        axs[4].set_ylabel('λ(t)')
        axs[4].grid(True)

        plt.tight_layout()
        plt.xlabel('Time')
        return fig, axs

    def analyze_statistics(self, results=None, n_steps=1000):
        """Analyze the statistical properties of the simulated returns."""
        if results is None:
            results = self.simulate(n_steps)

        returns = results['returns'].iloc[1:]  # Skip the first (0) return

        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = skew(returns)
        kurt = kurtosis(returns)

        # Autocorrelation
        acf_values = acf(returns, nlags=20)

        # Create plot
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Returns distribution
        axs[0, 0].hist(returns, bins=50, density=True, alpha=0.7)
        x = np.linspace(min(returns), max(returns), 100)
        axs[0, 0].plot(x, norm.pdf(x, mean_return, std_return), 'r-', lw=2)
        axs[0, 0].set_title(f'Returns Distribution (Skew={skewness:.2f}, Kurt={kurt:.2f})')
        axs[0, 0].set_xlabel('Return')
        axs[0, 0].set_ylabel('Density')
        axs[0, 0].grid(True)

        # QQ plot
        from scipy.stats import probplot
        probplot(returns, dist="norm", plot=axs[0, 1])
        axs[0, 1].set_title('Normal QQ Plot of Returns')
        axs[0, 1].grid(True)

        # Autocorrelation
        axs[1, 0].stem(range(len(acf_values)), acf_values)
        axs[1, 0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        # Add confidence bands
        conf_level = 1.96 / np.sqrt(len(returns))
        axs[1, 0].axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
        axs[1, 0].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        axs[1, 0].set_title('Autocorrelation Function')
        axs[1, 0].set_xlabel('Lag')
        axs[1, 0].set_ylabel('ACF')
        axs[1, 0].grid(True)

        # Volatility clustering
        abs_returns = np.abs(returns)
        abs_acf_values = acf(abs_returns, nlags=20)
        axs[1, 1].stem(range(len(abs_acf_values)), abs_acf_values)
        axs[1, 1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
        axs[1, 1].axhline(y=conf_level, color='r', linestyle='--', alpha=0.5)
        axs[1, 1].axhline(y=-conf_level, color='r', linestyle='--', alpha=0.5)
        axs[1, 1].set_title('Autocorrelation of Absolute Returns')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('ACF')
        axs[1, 1].grid(True)

        plt.tight_layout()

        # Print summary statistics
        print("Statistical Summary of Simulated Returns:")
        print(f"Mean: {mean_return:}")
        print(f"Standard Deviation: {std_return:}")
        print(f"Skewness: {skewness:}")
        print(f"Excess Kurtosis: {kurt:}")

        # Calculate significant autocorrelations
        significant_lags = [i for i, v in enumerate(acf_values) if abs(v) > conf_level and i > 0]
        if significant_lags:
            print(f"Significant autocorrelations at lags: {significant_lags}")
        else:
            print("No significant autocorrelations found")

        # Calculate significant volatility clustering
        significant_vol_lags = [i for i, v in enumerate(abs_acf_values) if abs(v) > conf_level and i > 0]
        if significant_vol_lags:
            print(f"Significant volatility clustering at lags: {significant_vol_lags}")
        else:
            print("No significant volatility clustering found")

        return fig, axs

    def compare_with_standard_gbm(self, n_steps=1000):
        """Compare this model with standard GBM."""
        # Simulate our reflexivity model
        results_reflexivity = self.simulate(n_steps)

        # Create a standard GBM for comparison
        def simulate_gbm(S0, mu, sigma, dt, n_steps):
            S = np.zeros(n_steps + 1)
            S[0] = S0

            for t in range(1, n_steps + 1):
                dW = np.random.normal(0, np.sqrt(dt))
                S[t] = S[t-1] * (1 + mu * dt + sigma * dW)

            return S

        # Simulate standard GBM with same baseline parameters
        gbm_prices = simulate_gbm(self.S0, self.mu_base, self.sigma_base, self.dt, n_steps)
        gbm_returns = np.diff(gbm_prices) / gbm_prices[:-1]

        # Create time index for GBM results
        gbm_time = np.arange(0, n_steps + 1) * self.dt

        # Plot comparison
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Price paths
        axs[0, 0].plot(results_reflexivity.index, results_reflexivity['price'], label='Reflexivity Model')
        axs[0, 0].plot(gbm_time, gbm_prices, label='Standard GBM')
        axs[0, 0].set_title('Price Comparison')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('Price')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Returns comparison
        axs[0, 1].plot(results_reflexivity.index[1:], results_reflexivity['returns'].iloc[1:],
                       label='Reflexivity Returns', alpha=0.7)
        axs[0, 1].plot(gbm_time[1:], gbm_returns, label='GBM Returns', alpha=0.7)
        axs[0, 1].set_title('Returns Comparison')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('Return')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Returns distributions
        reflexivity_returns = results_reflexivity['returns'].iloc[1:].values
        axs[1, 0].hist(reflexivity_returns, bins=50, density=True, alpha=0.7, label='Reflexivity Model')
        axs[1, 0].hist(gbm_returns, bins=50, density=True, alpha=0.5, label='Standard GBM')
        axs[1, 0].set_title('Returns Distribution')
        axs[1, 0].set_xlabel('Return')
        axs[1, 0].set_ylabel('Density')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Autocorrelation comparison
        reflexivity_acf = acf(reflexivity_returns, nlags=20)
        gbm_acf = acf(gbm_returns, nlags=20)

        axs[1, 1].stem(range(len(reflexivity_acf)), reflexivity_acf, label='Reflexivity Model', linefmt='b-', markerfmt='bo')
        axs[1, 1].stem(range(len(gbm_acf)), gbm_acf, label='Standard GBM', linefmt='r-', markerfmt='ro')
        axs[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Add confidence bands
        conf_level = 1.96 / np.sqrt(n_steps)
        axs[1, 1].axhline(y=conf_level, color='gray', linestyle='--', alpha=0.5)
        axs[1, 1].axhline(y=-conf_level, color='gray', linestyle='--', alpha=0.5)

        axs[1, 1].set_title('Return Autocorrelation Comparison')
        axs[1, 1].set_xlabel('Lag')
        axs[1, 1].set_ylabel('ACF')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Volatility clustering comparison
        reflexivity_abs_acf = acf(np.abs(reflexivity_returns), nlags=20)
        gbm_abs_acf = acf(np.abs(gbm_returns), nlags=20)

        axs[2, 0].stem(range(len(reflexivity_abs_acf)), reflexivity_abs_acf,
                       label='Reflexivity Model', linefmt='b-', markerfmt='bo')
        axs[2, 0].stem(range(len(gbm_abs_acf)), gbm_abs_acf,
                       label='Standard GBM', linefmt='r-', markerfmt='ro')
        axs[2, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axs[2, 0].axhline(y=conf_level, color='gray', linestyle='--', alpha=0.5)
        axs[2, 0].axhline(y=-conf_level, color='gray', linestyle='--', alpha=0.5)

        axs[2, 0].set_title('Volatility Clustering Comparison')
        axs[2, 0].set_xlabel('Lag')
        axs[2, 0].set_ylabel('ACF of |Returns|')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # QQ plot comparison
        from scipy.stats import probplot

        # Standardize returns for better comparison
        std_reflexivity = (reflexivity_returns - np.mean(reflexivity_returns)) / np.std(reflexivity_returns)
        std_gbm = (gbm_returns - np.mean(gbm_returns)) / np.std(gbm_returns)

        # Plot both on same QQ plot
        qq_reflexivity = probplot(std_reflexivity, dist="norm", fit=False)
        qq_gbm = probplot(std_gbm, dist="norm", fit=False)

        axs[2, 1].plot(qq_reflexivity[0], qq_reflexivity[1], 'bo', alpha=0.7, label='Reflexivity Model')
        axs[2, 1].plot(qq_gbm[0], qq_gbm[1], 'ro', alpha=0.7, label='Standard GBM')
        axs[2, 1].plot([-3, 3], [-3, 3], 'k-', alpha=0.7)  # Plot reference line
        axs[2, 1].set_title('Normal QQ Plot Comparison')
        axs[2, 1].set_xlabel('Theoretical Quantiles')
        axs[2, 1].set_ylabel('Sample Quantiles')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout()

        # Print comparison statistics
        print("Statistical Comparison:")
        print(f"{'Metric':<20} {'Reflexivity Model':<20} {'Standard GBM':<20}")
        print(f"{'-'*60}")
        print(f"{'Mean':<20} {np.mean(reflexivity_returns):<20} {np.mean(gbm_returns):<20}")
        print(f"{'Std Dev':<20} {np.std(reflexivity_returns):<20} {np.std(gbm_returns):<20}")
        print(f"{'Skewness':<20} {skew(reflexivity_returns):<20} {skew(gbm_returns):<20}")
        print(f"{'Excess Kurtosis':<20} {kurtosis(reflexivity_returns):<20} {kurtosis(gbm_returns):<20}")

        # Count significant autocorrelations
        sig_acf_reflexivity = sum(1 for v in reflexivity_acf[1:] if abs(v) > conf_level)
        sig_acf_gbm = sum(1 for v in gbm_acf[1:] if abs(v) > conf_level)

        print(f"{'Sig. Autocorr.':<20} {sig_acf_reflexivity:<20d} {sig_acf_gbm:<20d}")

        # Count significant volatility clustering
        sig_vol_reflexivity = sum(1 for v in reflexivity_abs_acf[1:] if abs(v) > conf_level)
        sig_vol_gbm = sum(1 for v in gbm_abs_acf[1:] if abs(v) > conf_level)

        print(f"{'Sig. Vol. Cluster':<20} {sig_vol_reflexivity:<20d} {sig_vol_gbm:<20d}")

        return fig, axs

    def analyze_multiscale_properties(self, n_steps=1000, scales=[1, 5, 10, 20, 50]):
        """
        Analyze how the statistical properties of returns change across different time scales.

        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        scales : list
            List of time scales (in steps) to analyze

        Returns:
        --------
        fig, axs : matplotlib figure and axes
        results_df : DataFrame with multiscale statistics
        """
        # Simulate at the finest scale
        results = self.simulate(n_steps)
        returns = results['returns'].iloc[1:].values

        # Prepare results storage
        scale_results = {
            'scale': [],
            'mean': [],
            'std': [],
            'skewness': [],
            'kurtosis': [],
            'hurst': [],
            'autocorr_lag1': [],
            'vol_cluster_lag1': []
        }

        # Analyze at each scale
        for scale in scales:
            # Aggregate returns to this scale
            if scale == 1:
                scale_returns = returns
            else:
                # Compute non-overlapping returns at this scale
                n_aggregated = len(returns) // scale
                scale_returns = np.zeros(n_aggregated)

                for i in range(n_aggregated):
                    # Compound the returns within each period
                    period_returns = returns[i*scale:(i+1)*scale]
                    scale_returns[i] = np.prod(1 + period_returns) - 1

            # Calculate statistics
            scale_results['scale'].append(scale)
            scale_results['mean'].append(np.mean(scale_returns) * scale)  # Scale the mean
            scale_results['std'].append(np.std(scale_returns))
            scale_results['skewness'].append(skew(scale_returns))
            scale_results['kurtosis'].append(kurtosis(scale_returns))

            # Estimate Hurst exponent using rescaled range method
            def estimate_hurst(ts):
                """Estimate Hurst exponent using rescaled range method."""
                N = len(ts)
                if N < 20:
                    return np.nan

                max_k = min(20, N // 4)
                RS_values = []
                ns = list(range(10, N // 2, max(1, (N // 2 - 10) // 10)))

                for n in ns:
                    # Calculate the adjusted rescaled range
                    rs_averaged = 0
                    for start in range(0, N - n, n):
                        series = ts[start:start+n]
                        mean = np.mean(series)
                        series = series - mean
                        cum_series = np.cumsum(series)
                        R = max(cum_series) - min(cum_series)
                        S = np.std(series)
                        if S == 0:
                            continue
                        rs_averaged += R/S

                    if rs_averaged > 0:
                        RS_values.append(rs_averaged / (1 + (N // n)))

                if len(RS_values) < 2:
                    return np.nan

                log_rs = np.log(RS_values)
                log_n = np.log(ns[:len(RS_values)])

                # Fit a line to the log-log plot
                slope, _, _, _, _ = np.polyfit(log_n, log_rs, 1, full=True)
                return slope[0]

            scale_results['hurst'].append(estimate_hurst(scale_returns))

            # Calculate autocorrelations
            if len(scale_returns) > 1:
                acf_values = acf(scale_returns, nlags=1, fft=False)
                scale_results['autocorr_lag1'].append(acf_values[1] if len(acf_values) > 1 else np.nan)

                abs_acf_values = acf(np.abs(scale_returns), nlags=1, fft=False)
                scale_results['vol_cluster_lag1'].append(abs_acf_values[1] if len(abs_acf_values) > 1 else np.nan)
            else:
                scale_results['autocorr_lag1'].append(np.nan)
                scale_results['vol_cluster_lag1'].append(np.nan)

        # Create DataFrame with results
        results_df = pd.DataFrame(scale_results)

        # Create plots to visualize scale-dependent properties
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))

        # Plot standard deviation vs. scale
        axs[0, 0].plot(results_df['scale'], results_df['std'], 'o-', linewidth=2)

        # Add reference lines for sqrt(T) and T^H scaling
        x_values = np.array(scales)
        y_base = results_df['std'].iloc[0]

        # Add sqrt(T) scaling reference
        y_sqrt = y_base * np.sqrt(x_values / x_values[0])
        axs[0, 0].plot(x_values, y_sqrt, 'r--', label='√T scaling (H=0.5)')

        # Add T^H scaling reference if we have a Hurst estimate
        if not np.isnan(results_df['hurst'].iloc[0]):
            h = results_df['hurst'].iloc[0]
            y_h = y_base * (x_values / x_values[0])**h
            axs[0, 0].plot(x_values, y_h, 'g--', label=f'T^{h:.2f} scaling (estimated H)')

        axs[0, 0].set_title('Standard Deviation vs. Time Scale')
        axs[0, 0].set_xlabel('Time Scale')
        axs[0, 0].set_ylabel('Standard Deviation')
        axs[0, 0].set_xscale('log')
        axs[0, 0].set_yscale('log')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot Hurst exponent vs. scale
        axs[0, 1].plot(results_df['scale'], results_df['hurst'], 'o-', linewidth=2)
        axs[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random Walk (H=0.5)')
        axs[0, 1].set_title('Hurst Exponent vs. Time Scale')
        axs[0, 1].set_xlabel('Time Scale')
        axs[0, 1].set_ylabel('Hurst Exponent')
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot skewness vs. scale
        axs[1, 0].plot(results_df['scale'], results_df['skewness'], 'o-', linewidth=2)
        axs[1, 0].axhline(y=0, color='r', linestyle='--', label='Normal Distribution (Skew=0)')
        axs[1, 0].set_title('Skewness vs. Time Scale')
        axs[1, 0].set_xlabel('Time Scale')
        axs[1, 0].set_ylabel('Skewness')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot kurtosis vs. scale
        axs[1, 1].plot(results_df['scale'], results_df['kurtosis'], 'o-', linewidth=2)
        axs[1, 1].axhline(y=0, color='r', linestyle='--', label='Normal Distribution (Kurt=0)')
        axs[1, 1].set_title('Excess Kurtosis vs. Time Scale')
        axs[1, 1].set_xlabel('Time Scale')
        axs[1, 1].set_ylabel('Excess Kurtosis')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Plot autocorrelation vs. scale
        axs[2, 0].plot(results_df['scale'], results_df['autocorr_lag1'], 'o-', linewidth=2)
        axs[2, 0].axhline(y=0, color='r', linestyle='--', label='Random Walk (ACF=0)')
        axs[2, 0].set_title('Lag-1 Autocorrelation vs. Time Scale')
        axs[2, 0].set_xlabel('Time Scale')
        axs[2, 0].set_ylabel('Autocorrelation')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Plot volatility clustering vs. scale
        axs[2, 1].plot(results_df['scale'], results_df['vol_cluster_lag1'], 'o-', linewidth=2)
        axs[2, 1].axhline(y=0, color='r', linestyle='--', label='Random Walk (ACF=0)')
        axs[2, 1].set_title('Volatility Clustering vs. Time Scale')
        axs[2, 1].set_xlabel('Time Scale')
        axs[2, 1].set_ylabel('ACF of |Returns|')
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        plt.tight_layout()

        # Print summary
        print("Multiscale Analysis Results:")
        print(results_df.to_string(index=False))

        return fig, axs, results_df

    def calibrate_to_real_data(self, real_data, param_ranges=None, method='bayesian'):
        """
        Calibrate the model parameters to match statistical properties of real data.
        """
        # Ensure only one simulation occurs during calibration
        original_simulate = self.simulate

        # Track the number of simulations
        simulation_count = 0

        def wrapped_simulate(n_steps):
            nonlocal simulation_count
            simulation_count += 1

            # Print warning if multiple simulations occur
            if simulation_count > 1:
                print(f"WARNING: Simulate called {simulation_count} times during calibration")

            return original_simulate(n_steps)

        # Temporarily replace simulate method
        self.simulate = wrapped_simulate

        try:
           
            result = ... 

        finally:
            # Restore original simulate method
            self.simulate = original_simulate

            # Reset simulation count
            simulation_count = 0

        return result

def max_drawdown(returns):
    """
    Calculate the maximum drawdown of a return series.

    Parameters:
    -----------
    returns : array-like
        Series of returns

    Returns:
    --------
    float
        Maximum drawdown percentage
    """
    # Prevent extreme values
    returns = np.clip(returns, -0.5, 0.5)

    # Use a more stable cumulative return calculation
    cumulative_returns = np.cumprod(1 + returns)

    try:
        # Find the maximum drawdown using a stable method
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        # Handle potential numerical issues
        drawdown = np.nan_to_num(drawdown, nan=0, posinf=0, neginf=0)

        return np.min(drawdown)
    except Exception:
        # Fallback to a simpler method if numerical issues persist
        return -0.5  # Conservative worst-case drawdown

def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate the Sharpe ratio of a return series.

    Parameters:
    -----------
    returns : array-like
        Series of returns
    risk_free_rate : float, optional
        Annual risk-free rate
    periods_per_year : int, optional
        Number of periods in a year

    Returns:
    --------
    float
        Sharpe ratio
    """
    # Ensure returns is a NumPy array
    returns = np.asarray(returns)

    # Remove any potential inf or nan values
    returns = returns[np.isfinite(returns)]

    # Ensure there are returns to process
    if len(returns) == 0:
        return 0.0

    # Annualize returns and volatility
    annual_return = np.mean(returns) * periods_per_year
    annual_volatility = np.std(returns) * np.sqrt(periods_per_year)

    # Calculate Sharpe ratio
    return (annual_return - risk_free_rate / periods_per_year) / annual_volatility if annual_volatility > 0 else 0

def step(self):
    """Advance the simulation by one time step."""
    # Calculate current parameters
    mu_t = self.mu(self.R)
    sigma_t = self.sigma(self.R)
    feedback = self.calculate_feedback()

    # Update Hurst exponent based on current autocorrelation
    self.update_hurst_exponent()

    # Update jump intensity
    self.update_jump_intensity()

    # Generate correlated Brownian increments for price and regime
    dW, dB = self.generate_correlated_normals() * np.sqrt(self.dt)

    # Replace standard Brownian with fractional Brownian for price
    dW = self.generate_fractional_brownian_increment(self.H)

    # Determine if a jump occurs
    jump_probability = self.lambda_t * self.dt
    jump_occurs = np.random.random() < jump_probability

    jump_size = 0
    if jump_occurs:
        jump_size = np.random.normal(self.jump_mean, self.jump_std)
        self.jump_times.append(self.time)

    # Calculate return for this step
    current_return = mu_t * self.dt + feedback * self.dt + sigma_t * dW + jump_size

    # Clip return to prevent extreme values
    current_return = np.clip(current_return, -0.5, 0.5)

    # Use log-based price update for numerical stability
    try:
        # Convert price to log space
        log_price = np.log(max(self.S, 1e-10))

        # Update in log space
        log_price += current_return

        # Convert back to price, with a safety check
        self.S = max(np.exp(log_price), 1e-10)
    except Exception:
        # Fallback method if log calculation fails
        self.S *= (1 + np.clip(current_return, -0.5, 0.5))
        self.S = max(self.S, 1e-10)

    # Update regime
    dR = self.kappa * (self.R_bar - self.R) * self.dt + self.gamma * current_return + self.eta * dB
    self.R = self.R + dR

    # Store history
    self.returns_history.append(current_return)
    self.price_history.append(self.S)
    self.regime_history.append(self.R)
    self.hurst_history.append(self.H)
    self.intensity_history.append(self.lambda_t)

    # Increment time
    self.time += self.dt

    return current_return

def validate_against_real_markets(symbols=['SPY', 'QQQ', 'AAPL', 'GOOGL', 'MSFT']):
    """
    Comprehensive validation across multiple assets
    """
    import sys
    import traceback

    results = {}

    print(f"Starting validation for {len(symbols)} symbols: {', '.join(symbols)}")
    print(f"Call stack at validation start:")
    traceback.print_stack(file=sys.stdout)

    def single_symbol_validation(symbol):
        try:
            print(f"\n{'='*50}\nProcessing {symbol}")
            print(f"Current process ID: {os.getpid()}")
            print(f"Current thread: {threading.current_thread().name}")

            # Download historical data
            data = yf.download(symbol, start='2010-01-01', end='2023-12-31')

            # Print detailed data information
            print(f"Data shape: {data.shape}")
            print(f"Data index: {data.index[0]} to {data.index[-1]}")

            # Ensure sufficient data
            if len(data) < 252:
                print(f"Skipping {symbol}: Insufficient data ({len(data)} days)")
                return None

            # Calculate returns
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

            # Skip if not enough returns
            if len(returns) < 100:
                print(f"Skipping {symbol}: Not enough returns")
                return None

            # Create a new model instance
            model = ReflexivityBrownianMotion()

            # Calibrate the model
            print("Starting calibration...")
            calibration_result = model.calibrate_to_real_data(returns)
            print("Calibration complete")

            # Print calibration details
            print(f"Calibration result: {calibration_result}")

            # Simulate ONCE and compare
            print("Starting simulation...")
            sim_results = model.simulate(len(returns))
            simulated_returns = sim_results['returns'].iloc[1:].values
            print("Simulation complete")

            # Verify simulated returns
            print(f"Simulated {len(simulated_returns)} returns")

            return {
                'real_stats': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'skewness': skew(returns),
                    'kurtosis': kurtosis(returns),
                    'autocorr': acf(returns, nlags=10)
                },
                'simulated_stats': {
                    'mean': np.mean(simulated_returns),
                    'std': np.std(simulated_returns),
                    'skewness': skew(simulated_returns),
                    'kurtosis': kurtosis(simulated_returns),
                    'autocorr': acf(simulated_returns, nlags=10)
                }
            }

        except Exception as e:
            print(f"Critical error processing {symbol}: {e}")
            traceback.print_exc()
            return None

    # Process each symbol
    for symbol in symbols:
        symbol_result = single_symbol_validation(symbol)
        if symbol_result is not None:
            results[symbol] = symbol_result
            print(f"Successfully processed {symbol}")

    print(f"Validation complete. Processed {len(results)} out of {len(symbols)} symbols")
    return results

# Modify the run_validation_demo function
def run_validation_demo():
    """
    Run comprehensive model validation
    """
    print("Running comprehensive model validation...")

    # Add import statements for additional debugging
    import os
    import threading

    # Validate against multiple market assets
    validation_results = validate_against_real_markets()

    # If you want to do further analysis, add it here
    for symbol, metrics in validation_results.items():
        print(f"\nValidation Results for {symbol}:")
        # Print key statistics
        for stat_type in ['real_stats', 'simulated_stats']:
            print(f"{stat_type.replace('_', ' ').title()}:")
            for metric, value in metrics[stat_type].items():
                print(f"  {metric}: {value}")

def statistical_comparison(validation_results):
    """
    Detailed statistical comparison and hypothesis testing
    """
    from scipy import stats

    comparison_metrics = {}

    for symbol, data in validation_results.items():
        real = data['real_stats']
        sim = data['simulated_stats']

        # Kolmogorov-Smirnov test
        ks_test = stats.ks_2samp(real['returns'], sim['returns'])

        comparison_metrics[symbol] = {
            'mean_difference': abs(real['mean'] - sim['mean']),
            'std_difference': abs(real['std'] - sim['std']),
            'skewness_difference': abs(real['skewness'] - sim['skewness']),
            'kurtosis_difference': abs(real['kurtosis'] - sim['kurtosis']),
            'ks_statistic': ks_test.statistic,
            'ks_pvalue': ks_test.pvalue
        }

    return comparison_metrics

def predictive_performance_test():
    """
    Out-of-sample predictive performance test
    """
    # Split data into training and testing
    # Calibrate model on training
    # Predict on testing
    # Compare prediction accuracy
    pass

def stress_test_scenarios():
    """
    Test model performance under various market conditions
    - Bull markets
    - Bear markets
    - High volatility periods
    - Low volatility periods
    """
    pass


def run_simulation_demo():
    """
    Run a demonstration of the reflexivity-adjusted Brownian motion model.
    """
    print("Initializing Reflexivity-Adjusted Brownian Motion Model...")

    # Create a model with default parameters
    model = ReflexivityBrownianMotion()

    # Simulate price paths
    n_steps = 1000
    print(f"Simulating {n_steps} steps...")
    results = model.simulate(n_steps)

    # Plot simulation results
    print("Plotting simulation results...")
    model.plot_simulation(results)
    plt.savefig('reflexivity_simulation.png')
    plt.show()

    # Analyze statistical properties
    print("Analyzing statistical properties...")
    model.analyze_statistics(results)
    plt.savefig('reflexivity_statistics.png')
    plt.show()

    # Compare with standard GBM
    print("Comparing with standard GBM...")
    model.compare_with_standard_gbm(n_steps)
    plt.savefig('reflexivity_vs_gbm.png')
    plt.show()

    # Analyze multiscale properties
    print("Analyzing multiscale properties...")
    model.analyze_multiscale_properties(n_steps)
    plt.savefig('reflexivity_multiscale.png')
    plt.show()
    run_validation_demo()
    print("Demonstration complete.")

if __name__ == "__main__":
    run_simulation_demo()
