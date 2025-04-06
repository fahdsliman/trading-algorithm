import pandas as pd
import numpy as np
import yfinance as yf  # For getting stock data
from datetime import datetime, timedelta


class TradingAlgorithm:
    def __init__(self, symbol, start_date, end_date, short_window=20, long_window=50):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window  # Short-term moving average period
        self.long_window = long_window  # Long-term moving average period
        self.data = None
        self.signals = None

    def get_data(self):
        """Fetch historical data from Yahoo Finance"""
        try:
            self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            if self.data.empty:
                raise ValueError("No data retrieved")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        return True

    def calculate_indicators(self):
        """Calculate moving averages and generate trading signals"""
        if self.data is None:
            return False

        # Calculate moving averages
        self.data['Short_MA'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=self.long_window).mean()

        # Initialize signals DataFrame
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Signal'] = 0.0

        # Generate trading signals
        self.signals['Signal'][self.short_window:] = np.where(
            self.data['Short_MA'][self.short_window:] > self.data['Long_MA'][self.short_window:],
            1.0,  # Buy signal
            0.0  # Sell signal
        )

        # Calculate positions
        self.signals['Position'] = self.signals['Signal'].diff()
        return True

    def backtest(self):
        """Backtest the trading strategy"""
        if self.data is None or self.signals is None:
            return None

        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Strategy_Returns'] = self.data['Returns'] * self.signals['Signal'].shift(1)

        # Calculate cumulative returns
        initial_capital = 10000.0
        self.data['Portfolio_Value'] = initial_capital * (1 + self.data['Strategy_Returns']).cumprod()

        # Calculate performance metrics
        total_return = (self.data['Portfolio_Value'][-1] / initial_capital - 1) * 100
        sharpe_ratio = np.sqrt(252) * (self.data['Strategy_Returns'].mean() /
                                       self.data['Strategy_Returns'].std())

        return {
            'Total Return (%)': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': self.calculate_max_drawdown()
        }

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        rolling_max = self.data['Portfolio_Value'].cummax()
        drawdown = (self.data['Portfolio_Value'] - rolling_max) / rolling_max
        return drawdown.min() * 100

    def execute(self):
        """Execute the complete trading algorithm"""
        if not self.get_data():
            return None
        if not self.calculate_indicators():
            return None
        return self.backtest()


# Example usage
if __name__ == "__main__":
    # Define parameters
    symbol = "AAPL"  # Stock symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data

    # Create and run the trading algorithm
    trader = TradingAlgorithm(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        short_window=20,
        long_window=50
    )

    results = trader.execute()

    if results:
        print(f"Trading Results for {symbol}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

        # Optional: Plot results
        import matplotlib.pyplot as plt

        trader.data[['Close', 'Short_MA', 'Long_MA']].plot(figsize=(12, 6))
        plt.title(f'{symbol} Price and Moving Averages')
        plt.show()

        trader.data['Portfolio_Value'].plot(figsize=(12, 6))
        plt.title(f'{symbol} Portfolio Value')
        plt.show()
