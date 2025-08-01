import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
from scipy.stats import pearsonr  # For correlation, nobody is really gonna care about this so obviously it's going in there
import tkinter as tk  # For the GUI because most people don't know how to use command line args
from tkinter import ttk  # For the progress bar
from tkinter import messagebox  # For making the user feel special with pop-ups

# Global variable for stock ticker, we're just gonna default to Microsoft because reasons
STOCK_TICKER = 'MSFT'

# Black-Scholes formula to calculate call option price
def black_scholes_call(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility):
    """
    This function uses the Black-Scholes formula to calculate the price of a call option.
    It inputs the stock price, strike price, time to expiry (in years), the risk-free rate, 
    and the volatility of the stock, and then outputs the call price.
    
    Fun Fact: This formula revolutionized the options market. You're welcome.
    """
    try:
        # Input sanity checks, because we don't trust users or ourselves
        if stock_price <= 0 or strike_price <= 0 or volatility <= 0 or time_to_expiry <= 0:
            return np.nan  # If they can't follow the rules, they get nan'd
        
        # The magic happens here. We calculate the d1 and d2 values for Black-Scholes.
        d1 = (np.log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
                    volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        # Return the call price. This is where we flex on people who don't understand this math.
        return stock_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    except Exception as e:
        print(f"Error in Black-Scholes calculation: {e}")  # When the math gods abandon us
        return np.nan

def calculate_implied_volatility(stock_price, strike_price, time_to_expiry, option_price, risk_free_rate):
    """
    This function calculates the implied volatility by iteratively adjusting the volatility estimate
    until the Black-Scholes formula returns an option price close to the actual market price.
    
    Just be thankful you're not doing this by hand. It's like finding a needle in a haystack but with math.
    """
    vol_estimate = 0.2  # We start with an arbitrary guess, because why not
    tolerance = 1e-5  # Because nobody's perfect, but we'll accept close enough
    max_iterations = 100  # If you haven't found it by now, you should probably give up

    for _ in range(max_iterations):
        price = black_scholes_call(stock_price, strike_price, time_to_expiry, risk_free_rate, vol_estimate)

        # If something goes wrong, we bail out
        if price is None or np.isnan(price):
            return np.nan

        vega = stock_price * norm.pdf(vol_estimate) * np.sqrt(time_to_expiry)  # Vega is how sensitive our option price is to changes in volatility
        price_diff = option_price - price  # We're hunting for this difference to approach zero

        # If we're close enough or vega is zero, we're done
        if abs(price_diff) < tolerance or vega == 0:
            return vol_estimate

        # If our volatility estimate is getting a little too crazy, call it quits
        if vol_estimate > 5:
            return np.nan

        # Otherwise, we update our estimate
        vol_estimate += price_diff / (vega if vega != 0 else 1)

    # If we failed miserably, let's just say nothing and walk away. If life has taught me anything, ignoring your problems is the best approach
    return np.nan

def calculate_weighted_volatility(options_data, stock_price, time_to_expiry, risk_free_rate):
    """
    This function calculates the weighted volatility using a combination of options data. 
    Because more math is always better
    
    The outcome? A Frankenstein's monster of volatility.
    """
    weighted_variance = 0  # Gotta start somewhere

    for option in options_data:
        strike_price = option['strike_price']
        option_price = option['option_price']

        # Ignore garbage input, we have standards (low, but still)
        if stock_price <= 0 or strike_price <= 0 or option_price <= 0:
            continue

        # Calculate the implied volatility for each option, but don't ask too many questions
        implied_vol = calculate_implied_volatility(stock_price, strike_price, time_to_expiry, option_price,
                                                   risk_free_rate)

        # If the volatility makes no sense, pretend it doesn't exist
        if np.isnan(implied_vol) or implied_vol <= 0 or implied_vol > 5:
            continue

        delta_k = 1  # A placeholder for now, another Phase 2 WIP
        weight = (2 / time_to_expiry) * (delta_k / strike_price ** 2) * np.exp(risk_free_rate * time_to_expiry)
        weighted_variance += weight * implied_vol ** 2

    return np.sqrt(weighted_variance) * np.sqrt(30 / 365) if weighted_variance != 0 else np.nan  # Here's your volatility. Hope it's right, otherwise get rekt

# Just using Yahoo Finance to retrieve historical options data for now. I'll switch this to Polygon.io at a later date after I fix the smaller errors
def get_options_data(ticker_symbol, expiration):
    """
    Fetches options data for a given stock ticker and expiration date using Yahoo Finance.
    Because when in doubt, trust a free API for financial decisions.
    """
    ticker = yf.Ticker(ticker_symbol)
    options_data = []  # Prepare for some heavy lifting

    option_chain = ticker.option_chain(expiration)
    calls = option_chain.calls  # We only care about calls, because we're optimists

    # Populate our list with strike prices and last prices for each option
    for index, row in calls.iterrows():
        options_data.append({
            'strike_price': row['strike'],
            'option_price': row['lastPrice'],
        })
    return options_data  # Here's your precious data, do with it what you will

def fetch_price_data(ticker, date):
    """
    Fetches the stock's closing price for a specific date.
    As simple as it sounds, but there's always room for things to go horribly wrong.
    """
    try:
        stock_history = ticker.history(start=date, end=date + timedelta(days=1))  # Fetch the data
        if not stock_history.empty:
            return stock_history['Close'].iloc[0]  # Give them the first close price they see
    except Exception as e:
        print(f"Error fetching price for {date}: {e}")  # If things go south, we'll just throw out an error
    return np.nan  # We're ignoring our problems again if there's an error. Spot the pattern.

# Calculate the VIX for a single stock. We're going to assume calculation over the last 30 days, but the world is your oyster here. The longer the time period you input though, the longer it's going to take
def get_vix_equivalent_over_time(ticker_symbol, start_date, end_date, progress_callback=None, root=None):
    """
    Calculates the VIX equivalent (volatility index) for a stock over a time range.
    Prepare to watch paint dry as we calculate for each day between start_date and end_date.
    """
    dates = []  # List to store the dates
    vix_values = []  # List to store the VIX equivalent values
    delta_days = (end_date - start_date).days  # Total number of days we're working with

    ticker = yf.Ticker(ticker_symbol)  # Stock data from our definitely always right, never wrong free API

    for day in range(delta_days):
        current_date = start_date + timedelta(days=day)  # Move day by day

        # Skip weekends, because even the market needs rest
        if current_date.weekday() >= 5:
            continue

        try:
            expiration_list = ticker.options  # Get the options expiration dates
            if len(expiration_list) == 0:
                print(f"No options data for {current_date}")
                continue

            expiration = expiration_list[0]  # We're going with the nearest expiration
            options_data = get_options_data(ticker_symbol, expiration)

            stock_price = fetch_price_data(ticker, current_date)  # Fetch the stock price for that day
            if np.isnan(stock_price):
                print(f"No price data for {current_date}")
                continue

            time_to_expiry = 30 / 365  # Assume 30 days to expiration. Who needs accuracy?
            risk_free_rate = 0.05  # Here's the 5% risk-free rate assumption. I'm just kinda lazy like that.

            vix_value = calculate_weighted_volatility(options_data, stock_price, time_to_expiry, risk_free_rate)
            dates.append(current_date)
            vix_values.append(vix_value)

            # Update progress in the GUI
            if progress_callback:
                progress_callback(day + 1, delta_days, root)

        except Exception as e:
            print(f"Error on {current_date}: {e}")

    return dates, vix_values  # Return the dates and their corresponding VIX values

# We need to normalize the VIX values for the stock vs VIX for the index, because the values for individual stocks are smol compared to the VIX.
# To do this we'll incorporate a min-max normalization 
def min_max_normalize(values):
    """
    Performs min-max normalization on the given list of values.
    In other words, turning your mess of numbers into a neat 0-to-1 range.
    """
    values = np.array(values)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return values  # If everything is NaN, well, good luck with that.
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    if min_val == max_val:
        return values  # If all the values are the same, we shrug and move on.
    values[mask] = (values[mask] - min_val) / (max_val - min_val)
    return values

def visualize_vix_comparison(dates, vix_stock, vix_spy):
    """
    Plots the comparison between a stock's normalized VIX equivalent and SPY's VIX.
    Because no financial tool is complete without a pretty graph to show off.
    """
    vix_stock_normalized = min_max_normalize(vix_stock)
    vix_spy_normalized = min_max_normalize(vix_spy)

    # Ensure that dates and VIX values have the same length
    dates_common = []
    vix_stock_common = []
    vix_spy_common = []

    for date, vs, vp in zip(dates, vix_stock_normalized, vix_spy_normalized):
        if not np.isnan(vs) and not np.isnan(vp):
            dates_common.append(date)
            vix_stock_common.append(vs)
            vix_spy_common.append(vp)

    if len(dates_common) == 0:
        print("No common data to plot.")
        return

    formatted_dates = [date.strftime("%d") for date in dates_common]

    plt.figure(figsize=(10, 6))
    plt.plot(formatted_dates, vix_stock_common, marker='o', label=f'{STOCK_TICKER} (Normalized)')
    plt.plot(formatted_dates, vix_spy_common, marker='x', label='SPY (Normalized)')
    plt.title(f'Normalized VIX Equivalent for {STOCK_TICKER} vs SPY (Last Month)')
    plt.xlabel('Day')
    plt.ylabel('Normalized VIX Equivalent')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_correlation(vix_stock, vix_spy):
    """
    Calculates the Pearson correlation between the stock's VIX equivalent and the SPY VIX.
    Because you know you're fancy when you start calculating correlations.
    """
    vix_stock = np.array(vix_stock)
    vix_spy = np.array(vix_spy)
    mask = ~np.isnan(vix_stock) & ~np.isnan(vix_spy)  # Only compare values that exist in both
    if mask.sum() > 1:
        correlation, _ = pearsonr(vix_stock[mask], vix_spy[mask])  # Calculate the correlation coefficient. Thanks Pearson.
        return correlation
    return None

# GUI for user input and progress bar
def start_app():
    """
    Creates the GUI for user input and progress display.
    """
    global STOCK_TICKER

    root = tk.Tk()
    root.title("VIX Comparison Tool")

    tk.Label(root, text="Enter Stock Ticker:").pack()  # You can change the ticker
    stock_input = tk.Entry(root)
    stock_input.insert(0, STOCK_TICKER)  # Default to MSFT, because laziness
    stock_input.pack()

    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=10)

    def run():
        global STOCK_TICKER
        STOCK_TICKER = stock_input.get()  # Get whatever stock ticker the user entered
        progress['value'] = 0  # Reset the progress bar
        root.update_idletasks()
        show_results(progress, root)

    tk.Button(root, text="Run", command=run).pack()

    root.mainloop()

def show_results(progress, root):
    """
    Runs the VIX calculation for both the chosen stock and SPY, visualizes the results,
    and shows the correlation between them.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # We only care about the last 30 days

    def update_progress(current, total, root):
        progress['value'] = current / total * 100
        root.update_idletasks()

    # Get VIX equivalent for the stock and SPY
    dates_stock, vix_stock = get_vix_equivalent_over_time(STOCK_TICKER, start_date, end_date, update_progress, root)
    dates_spy, vix_spy = get_vix_equivalent_over_time('SPY', start_date, end_date, update_progress, root)

    # Show a graph comparing the two
    visualize_vix_comparison(dates_stock, vix_stock, vix_spy)

    # Calculate the correlation between the stock's VIX and SPY's VIX
    correlation = calculate_correlation(vix_stock, vix_spy)
    if correlation is not None:
        messagebox.showinfo("Correlation",
                            f"The correlation between {STOCK_TICKER} VIX and SPY VIX is {correlation * 100:.2f}%")
    else:
        messagebox.showinfo("Correlation", "Not enough valid data to calculate correlation.")

if __name__ == "__main__":
    start_app()
