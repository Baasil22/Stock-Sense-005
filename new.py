import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Stock Sense", page_icon="📈", layout="wide")

# Center the image using Streamlit's st.image within markdown
st.markdown(
    """
    <style>
    .center_image {
        display: flex;
        justify-content: center;
        margin-bottom: 20px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="center_image">', unsafe_allow_html=True)
st.image(r"C:\Users\baasi\Downloads\1730217505585tyk4uy8n\original.png", width=150)
st.markdown('</div>', unsafe_allow_html=True)

# Giving background color for whole page
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #F0F0F5;
    }
    .css-1d391kg {
        background-color: #0066CC;
        color: white;
    }
    .main {
        background-color: #F0F0F5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center;'>Stock Sense</h1>", unsafe_allow_html=True)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# list of big companies
stocks_dict = {
    "AAPL": "Apple Inc.",
    "GOOG": "Alphabet Inc. (Google)",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc. (Facebook)",
    "NFLX": "Netflix Inc.",
    "NVDA": "Nvidia Corporation",
    "BABA": "Alibaba Group",
    "GME": "GameStop Corp.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "V": "Visa Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "DIS": "The Walt Disney Company",
    "AMD": "Advanced Micro Devices Inc.",
    "BA": "The Boeing Company",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo Inc."
}

# Create a list of formatted strings for the multiselect
stocks = [f"{symbol} - {name}" for symbol, name in stocks_dict.items()]

selected_stocks = st.multiselect("Select stock symbols for prediction:", stocks, default=["AAPL - Apple Inc."])

#enter custom stock symbols
custom_stocks_input = st.text_input("Or, enter custom stock symbols (comma separated):", "")

# Combine predefined and custom stocks into a final list
if custom_stocks_input:
    custom_stocks = [symbol.strip().upper() for symbol in custom_stocks_input.split(",")]
    # Add the custom stocks to the selected stocks
    final_stocks = [stock.split(" - ")[0] for stock in selected_stocks] + custom_stocks
else:
    # Only use the predefined stocks if no custom input is provided
    final_stocks = [stock.split(" - ")[0] for stock in selected_stocks]

final_stocks = list(set(final_stocks))

# if we didnt enter anything
if final_stocks:
    stock_to_predict = final_stocks[0] 
else:
    st.warning("Please select or enter at least one stock symbol.")
    st.stop()

n_year = st.slider("YEARS OF PREDICTION:", 1, 5)
period = n_year * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(stock_to_predict)
data_load_state.text("Loading data...done!")

# Plot the raw data
st.subheader('Raw data')
st.write(data.tail())


def plot_raw_data():
    st.subheader('Time Series Data')  # Set the subheader for the time series plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Price"))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.subheader('Forecast plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
