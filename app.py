import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DARK TERMINAL THEME ────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  section[data-testid="stSidebar"] {
      background-color: #161b22;
      border-right: 1px solid #30363d;
  }
  /* Metric cards */
  [data-testid="metric-container"] {
      background-color: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px 16px;
  }
  [data-testid="metric-container"] label {
      color: #8b949e !important;
      font-size: 12px !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #e6edf3 !important;
      font-size: 22px !important;
      font-weight: 600 !important;
  }
  /* Inputs */
  .stTextInput input, .stSelectbox select {
      background-color: #21262d !important;
      color: #e6edf3 !important;
      border: 1px solid #30363d !important;
      border-radius: 6px !important;
  }
  /* Buttons */
  .stButton > button {
      background-color: #238636 !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: 6px !important;
      font-weight: 600 !important;
      width: 100%;
      padding: 10px !important;
      font-size: 15px !important;
  }
  .stButton > button:hover {
      background-color: #2ea043 !important;
  }
  /* Section headers */
  h1, h2, h3 { color: #e6edf3 !important; }
  .stMarkdown p { color: #8b949e; }
  /* Dividers */
  hr { border-color: #30363d; }
  /* Dataframe */
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
  /* Slider */
  .stSlider .st-bk { background-color: #238636; }
  /* Success / warning banners */
  .stSuccess { background-color: #1a3a1a !important; border-color: #238636 !important; }
  .stWarning { background-color: #3a2a0a !important; }
  .stError   { background-color: #3a0a0a !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ══════════════════════════════════════════════════════

def calc_ema(series, span):
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calc_rsi(series, period=14):
    """RSI using Wilder's smoothing (standard method)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def resample_weekly(df):
    """Resample daily OHLCV to weekly for weekly RSI."""
    weekly = df['Close'].resample('W').last()
    return weekly


# ══════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    """Download and clean OHLCV data from yfinance."""
    raw = yf.download(ticker, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)
    if raw.empty:
        return None

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════
#  SIGNAL GENERATION  (no lookahead bias)
# ══════════════════════════════════════════════════════

def generate_signals(df):
    """
    Compute all indicators and buy/sell signals.
    All signals are shifted by 1 day so entry is
    on the NEXT day's open — no lookahead bias.
    """
    df = df.copy()

    # ── EMAs ──
    df['ema5']  = calc_ema(df['Close'], 5)
    df['ema13'] = calc_ema(df['Close'], 13)
    df['ema26'] = calc_ema(df['Close'], 26)

    # ── Daily RSI ──
    df['rsi_daily'] = calc_rsi(df['Close'], 14)

    # ── Weekly RSI (resampled then merged back) ──
    weekly_close = df['Close'].resample('W').last()
    weekly_rsi   = calc_rsi(weekly_close, 14)
    # Forward-fill weekly RSI into daily index
    df['rsi_weekly'] = weekly_rsi.reindex(df.index, method='ffill')

    # ── Bullish candle ──
    df['bullish_candle'] = df['Close'] > df['Open']

    # ── Volume rising ──
    df['volume_rising'] = df['Volume'] > df['Volume'].shift(1)

    # ── EMA alignment ──
    df['ema_aligned'] = (
        (df['ema5'] > df['ema13']) &
        (df['ema13'] > df['ema26'])
    )

    # ── EMA crossover sequence ──
    # Step 1: EMA26 crossed above EMA13 (from below)
    df['ema26_cross_ema13'] = (
        (df['ema26'] > df['ema13']) &
        (df['ema26'].shift(1) <= df['ema13'].shift(1))
    )
    # Step 2: After step 1, EMA5 crosses above EMA13
    # Track if ema26 cross happened in last 10 days
    df['seq_step1_recent'] = (
        df['ema26_cross_ema13']
        .rolling(10, min_periods=1).max()
        .astype(bool)
    )
    df['ema5_cross_ema13'] = (
        (df['ema5'] > df['ema13']) &
        (df['ema5'].shift(1) <= df['ema13'].shift(1))
    )
    df['crossover_sequence'] = (
        df['seq_step1_recent'] & df['ema5_cross_ema13']
    )

    # ── RSI confirmation ──
    df['rsi_confirmed'] = (
        (df['rsi_daily'] > 60) &
        (df['rsi_weekly'] > 60)
    )

    # ── Weekly RSI threshold relaxed to 50 ──
    df['rsi_confirmed'] = (
    (df['rsi_daily'] > 60) &
    (df['rsi_weekly'] > 50)   # weekly just needs to be above 50
    )

    # ── MASTER BUY SIGNAL ──
    df['raw_buy'] = (
    df['ema_aligned'] &      # EMA5 > EMA13 > EMA26
    df['volume_rising'] &    # volume rising
    df['bullish_candle'] &   # green candle
    df['rsi_confirmed']      # RSI daily > 60, weekly > 50
    )
    # Enter next day's open — shift signal forward by 1
    df['buy_signal'] = df['raw_buy'].shift(1).fillna(False)

    # ── SELL SIGNALS ──
    # Partial exit (25%): EMA5 crosses below EMA13
    df['sell_25'] = (
        (df['ema5'] < df['ema13']) &
        (df['ema5'].shift(1) >= df['ema13'].shift(1))
    )
    # Full exit (remaining 75%): EMA13 crosses below EMA26
    df['sell_75'] = (
        (df['ema13'] < df['ema26']) &
        (df['ema13'].shift(1) >= df['ema26'].shift(1))
    )
    # Shift sell signals by 1 (act next day)
    df['sell_25'] = df['sell_25'].shift(1).fillna(False)
    df['sell_75'] = df['sell_75'].shift(1).fillna(False)

    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════
#  BACKTESTING ENGINE
# ══════════════════════════════════════════════════════

def run_backtest(df, init_cash=100000, commission=0.001):
    """
    Full backtesting engine with partial exits.
    Tracks equity curve, trade log, and position sizing.
    """
    cash      = float(init_cash)
    shares    = 0.0
    equity    = []
    trades    = []
    in_trade  = False
    entry_px  = 0.0
    entry_day = None

    for i, (idx, row) in enumerate(df.iterrows()):
        price = float(row['Open'])   # entry/exit at open (next day)

        # ── BUY ──
        if row['buy_signal'] and not in_trade and cash > 0:
            shares    = (cash * (1 - commission)) / price
            cash      = 0.0
            in_trade  = True
            entry_px  = price
            entry_day = idx
            trades.append({
                'date'      : idx,
                'type'      : 'BUY',
                'price'     : round(price, 2),
                'shares'    : round(shares, 4),
                'value'     : round(shares * price, 2),
                'pnl'       : 0.0,
                'pnl_pct'   : 0.0
            })

        # ── PARTIAL SELL 25% ──
        elif row['sell_25'] and in_trade and shares > 0:
            exit_shares = shares * 0.25
            proceeds    = exit_shares * price * (1 - commission)
            cash       += proceeds
            shares     -= exit_shares
            pnl         = proceeds - (exit_shares * entry_px)
            trades.append({
                'date'      : idx,
                'type'      : 'SELL 25%',
                'price'     : round(price, 2),
                'shares'    : round(exit_shares, 4),
                'value'     : round(proceeds, 2),
                'pnl'       : round(pnl, 2),
                'pnl_pct'   : round((price / entry_px - 1) * 100, 2)
            })

        # ── FULL SELL remaining 75% ──
        elif row['sell_75'] and in_trade and shares > 0:
            proceeds  = shares * price * (1 - commission)
            cash     += proceeds
            pnl       = proceeds - (shares * entry_px)
            trades.append({
                'date'      : idx,
                'type'      : 'SELL 75%',
                'price'     : round(price, 2),
                'shares'    : round(shares, 4),
                'value'     : round(proceeds, 2),
                'pnl'       : round(pnl, 2),
                'pnl_pct'   : round((price / entry_px - 1) * 100, 2)
            })
            shares   = 0.0
            in_trade = False

        # ── EQUITY ──
        total_equity = cash + shares * float(row['Close'])
        equity.append({'date': idx, 'equity': total_equity})

    # Close any open position at last price
    if in_trade and shares > 0:
        last_price = float(df['Close'].iloc[-1])
        proceeds   = shares * last_price * (1 - commission)
        cash      += proceeds
        trades.append({
            'date'    : df.index[-1],
            'type'    : 'SELL (end)',
            'price'   : round(last_price, 2),
            'shares'  : round(shares, 4),
            'value'   : round(proceeds, 2),
            'pnl'     : round(proceeds - shares * entry_px, 2),
            'pnl_pct' : round((last_price / entry_px - 1) * 100, 2)
        })

    equity_df = pd.DataFrame(equity).set_index('date')
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return equity_df, trades_df, cash


# ══════════════════════════════════════════════════════
#  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════

def calc_metrics(equity_df, init_cash, trades_df):
    final_val   = float(equity_df['equity'].iloc[-1])
    total_ret   = (final_val / init_cash - 1) * 100
    n_days      = (equity_df.index[-1] - equity_df.index[0]).days
    n_years     = n_days / 365.25
    cagr        = ((final_val / init_cash) ** (1 / max(n_years, 0.01)) - 1) * 100

    roll_max    = equity_df['equity'].cummax()
    drawdown    = (equity_df['equity'] - roll_max) / roll_max * 100
    max_dd      = drawdown.min()

    daily_ret   = equity_df['equity'].pct_change().dropna()
    sharpe      = (np.sqrt(252) * daily_ret.mean() /
                   daily_ret.std()) if daily_ret.std() > 0 else 0

    # Win rate from completed sell trades
    sells = trades_df[trades_df['type'].str.startswith('SELL')] if not trades_df.empty else pd.DataFrame()
    win_rate = (
        (sells['pnl'] > 0).sum() / len(sells) * 100
        if len(sells) > 0 else 0
    )
    n_trades = len(trades_df[trades_df['type'] == 'BUY']) if not trades_df.empty else 0

    return {
        'Final Value'   : final_val,
        'Total Return'  : total_ret,
        'CAGR'          : cagr,
        'Max Drawdown'  : max_dd,
        'Sharpe Ratio'  : sharpe,
        'Win Rate'      : win_rate,
        'Total Trades'  : n_trades
    }


# ══════════════════════════════════════════════════════
#  PLOTLY CHART  (trading terminal style)
# ══════════════════════════════════════════════════════

def build_chart(df, equity_df, trades_df, ticker):
    """
    4-panel interactive Plotly chart:
      1. Candlestick + EMAs + signals
      2. Volume
      3. RSI
      4. Equity curve
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.20, 0.20],
        subplot_titles=(
            f'{ticker} — Price & EMAs',
            'Volume',
            'RSI (14)',
            'Equity Curve'
        )
    )

    # ── 1. Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'],  close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        line_width=1
    ), row=1, col=1)

    # EMAs
    ema_styles = [
        ('ema5',  '#f48fb1', 'EMA 5'),
        ('ema13', '#ffcc02', 'EMA 13'),
        ('ema26', '#29b6f6', 'EMA 26'),
    ]
    for col_name, color, label in ema_styles:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            name=label, line=dict(color=color, width=1.3),
            hovertemplate=f'{label}: ₹%{{y:.2f}}<extra></extra>'
        ), row=1, col=1)

    # Buy signals
    if not trades_df.empty:
        buys = trades_df[trades_df['type'] == 'BUY']
        if not buys.empty:
            buy_dates  = pd.to_datetime(buys['date'])
            buy_prices = df.loc[df.index.isin(buy_dates), 'Low'] * 0.97
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices.values,
                mode='markers',
                marker=dict(symbol='triangle-up', size=14,
                            color='#00e676', line=dict(color='white', width=1)),
                name='Buy entry',
                hovertemplate='BUY @ ₹%{y:.2f}<extra></extra>'
            ), row=1, col=1)

        # Sell 25%
        s25 = trades_df[trades_df['type'] == 'SELL 25%']
        if not s25.empty:
            s25_dates  = pd.to_datetime(s25['date'])
            s25_prices = df.loc[df.index.isin(s25_dates), 'High'] * 1.03
            fig.add_trace(go.Scatter(
                x=s25_dates, y=s25_prices.values,
                mode='markers',
                marker=dict(symbol='triangle-down', size=11,
                            color='#ffab40', line=dict(color='white', width=1)),
                name='Sell 25%',
                hovertemplate='SELL 25% @ ₹%{y:.2f}<extra></extra>'
            ), row=1, col=1)

        # Sell 75%
        s75 = trades_df[trades_df['type'].isin(['SELL 75%', 'SELL (end)'])]
        if not s75.empty:
            s75_dates  = pd.to_datetime(s75['date'])
            s75_prices = df.loc[df.index.isin(s75_dates), 'High'] * 1.03
            fig.add_trace(go.Scatter(
                x=s75_dates, y=s75_prices.values,
                mode='markers',
                marker=dict(symbol='triangle-down', size=14,
                            color='#ef5350', line=dict(color='white', width=1)),
                name='Sell 75%',
                hovertemplate='SELL 75% @ ₹%{y:.2f}<extra></extra>'
            ), row=1, col=1)

    # ── 2. Volume ──
    vol_colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(df['Close'], df['Open'])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume', marker_color=vol_colors,
        opacity=0.7,
        hovertemplate='Vol: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    # Volume MA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume'].rolling(20).mean(),
        name='Vol MA20',
        line=dict(color='#ffcc02', width=1.2),
        hovertemplate='Vol MA20: %{y:,.0f}<extra></extra>'
    ), row=2, col=1)

    # ── 3. RSI ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi_daily'],
        name='RSI Daily',
        line=dict(color='#ce93d8', width=1.5),
        hovertemplate='RSI: %{y:.1f}<extra></extra>'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi_weekly'],
        name='RSI Weekly',
        line=dict(color='#80cbc4', width=1.2, dash='dot'),
        hovertemplate='RSI Weekly: %{y:.1f}<extra></extra>'
    ), row=3, col=1)

    # RSI reference lines
    for level, color, label in [
        (60, '#00e676', 'Buy threshold'),
        (70, '#ef5350', 'Overbought'),
        (30, '#29b6f6', 'Oversold')
    ]:
        fig.add_hline(
            y=level, line_dash='dash',
            line_color=color, line_width=1,
            annotation_text=f' {level} {label}',
            annotation_font_color=color,
            annotation_font_size=10,
            row=3, col=1
        )

    # RSI fill above 60
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi_daily'],
        fill='tozeroy',
        fillcolor='rgba(0,230,118,0.07)',
        line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ), row=3, col=1)

    # ── 4. Equity Curve ──
    bh_equity = 100000 * (df['Close'] / df['Close'].iloc[0])

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df['equity'],
        name='Your Strategy',
        line=dict(color='#00e676', width=2),
        fill='tozeroy',
        fillcolor='rgba(0,230,118,0.05)',
        hovertemplate='Strategy: ₹%{y:,.0f}<extra></extra>'
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=bh_equity,
        name='Buy & Hold',
        line=dict(color='#8b949e', width=1.5, dash='dot'),
        hovertemplate='B&H: ₹%{y:,.0f}<extra></extra>'
    ), row=4, col=1)

    # ── Layout ──
    fig.update_layout(
        height=900,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#8b949e', size=11),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(size=10),
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='left',   x=0
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            font_color='#e6edf3'
        )
    )

    # Grid style for all panels
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor='#21262d', gridwidth=0.5,
            zerolinecolor='#30363d',
            showspikes=True, spikecolor='#8b949e',
            spikethickness=1, spikemode='across',
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='#21262d', gridwidth=0.5,
            zerolinecolor='#30363d',
            row=i, col=1
        )

    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # Subplot title colors
    for ann in fig.layout.annotations:
        ann.font.color = '#8b949e'
        ann.font.size  = 11

    return fig


# ══════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div style="border-bottom:1px solid #30363d; padding-bottom:16px; margin-bottom:20px;">
  <h1 style="margin:0; font-size:24px; color:#e6edf3;">
    📈 Triple EMA + Volume + RSI Backtester
  </h1>
  <p style="margin:4px 0 0; color:#8b949e; font-size:13px;">
    EMA (5, 13, 26) · RSI Daily & Weekly · Partial exits · No lookahead bias
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0; border-bottom:1px solid #30363d; margin-bottom:16px;">
      <span style="font-size:20px;">⚙️</span>
      <span style="color:#e6edf3; font-size:15px; font-weight:600; margin-left:8px;">Settings</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#8b949e; font-size:12px; margin-bottom:4px;">STOCKS</p>',
                unsafe_allow_html=True)
    tickers_input = st.text_input(
        "Ticker symbols (comma separated)",
        value="HDFCBANK.NS",
        label_visibility="collapsed",
        placeholder="e.g. RELIANCE.NS, TCS.NS, INFY.NS"
    )

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">DATE RANGE</p>',
                unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("From", value=pd.to_datetime("2020-01-01"),
                                   label_visibility="visible")
    with col_d2:
        end_date = st.date_input("To", value=pd.to_datetime("2025-01-01"),
                                 label_visibility="visible")

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">CAPITAL & COSTS</p>',
                unsafe_allow_html=True)
    init_cash  = st.number_input("Starting capital (₹)", value=100000,
                                  step=10000, min_value=10000)
    commission = st.slider("Commission per trade (%)", 0.0, 0.5, 0.1, 0.05) / 100

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">OPTIONS</p>',
                unsafe_allow_html=True)
    export_csv = st.checkbox("Export trades to CSV", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Backtest")

    st.markdown("""
    <div style="margin-top:24px; padding:12px; background:#161b22;
                border:1px solid #30363d; border-radius:8px;">
      <p style="color:#8b949e; font-size:11px; margin:0; line-height:1.6;">
        <b style="color:#e6edf3;">Buy conditions:</b><br>
        ✦ EMA5 > EMA13 > EMA26<br>
        ✦ EMA26 crosses EMA13, then EMA5 crosses EMA13<br>
        ✦ Volume rising + bullish candle<br>
        ✦ RSI > 60 (daily & weekly)<br><br>
        <b style="color:#e6edf3;">Sell conditions:</b><br>
        ✦ EMA5 cross below EMA13 → exit 25%<br>
        ✦ EMA13 cross below EMA26 → exit 75%
      </p>
    </div>
    """, unsafe_allow_html=True)

# ── Run ──
if run_btn:
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    if not tickers:
        st.error("Please enter at least one ticker symbol.")
        st.stop()

    for ticker in tickers:
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:12px 16px; margin:16px 0 8px;">
          <span style="color:#e6edf3; font-size:16px; font-weight:600;">{ticker}</span>
          <span style="color:#8b949e; font-size:12px; margin-left:8px;">
            {str(start_date)} → {str(end_date)}
          </span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(f"Downloading {ticker} data..."):
            df_raw = fetch_data(ticker, start_date, end_date)

        if df_raw is None or df_raw.empty:
            st.error(f"No data found for {ticker}. Check the ticker symbol.")
            continue

        # Generate signals
        df = generate_signals(df_raw)

        if df.empty:
            st.warning(f"Not enough data to generate signals for {ticker}.")
            continue

        # Run backtest
        equity_df, trades_df, final_cash = run_backtest(
            df, init_cash=init_cash, commission=commission)

        # Metrics
        metrics = calc_metrics(equity_df, init_cash, trades_df)

        # ── Metric cards ──
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        cards = [
            (c1, "Final Value",
             f"₹{metrics['Final Value']:,.0f}", None),
            (c2, "Total Return",
             f"{metrics['Total Return']:+.1f}%",
             metrics['Total Return'] >= 0),
            (c3, "CAGR",
             f"{metrics['CAGR']:+.1f}%",
             metrics['CAGR'] >= 0),
            (c4, "Max Drawdown",
             f"{metrics['Max Drawdown']:.1f}%",
             metrics['Max Drawdown'] > -20),
            (c5, "Sharpe Ratio",
             f"{metrics['Sharpe Ratio']:.2f}",
             metrics['Sharpe Ratio'] >= 1),
            (c6, "Win Rate",
             f"{metrics['Win Rate']:.1f}%",
             metrics['Win Rate'] >= 50),
            (c7, "Total Trades",
             str(metrics['Total Trades']), None),
        ]
        for col, label, val, good in cards:
            delta_color = "normal" if good is None else ("normal" if good else "inverse")
            col.metric(label, val)

        # ── Chart ──
        st.plotly_chart(
            build_chart(df, equity_df, trades_df, ticker),
            use_container_width=True
        )

        # ── Trade log ──
        if not trades_df.empty:
            st.markdown("""
            <p style="color:#8b949e; font-size:12px;
               margin:16px 0 6px; text-transform:uppercase;
               letter-spacing:0.08em;">Trade Log</p>
            """, unsafe_allow_html=True)

            display_df = trades_df.copy()
            display_df['date']    = pd.to_datetime(display_df['date']).dt.strftime('%d %b %Y')
            display_df['price']   = display_df['price'].apply(lambda x: f"₹{x:,.2f}")
            display_df['value']   = display_df['value'].apply(lambda x: f"₹{x:,.0f}")
            display_df['pnl']     = display_df['pnl'].apply(
                lambda x: f"+₹{x:,.0f}" if x >= 0 else f"-₹{abs(x):,.0f}")
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(
                lambda x: f"{x:+.2f}%")
            display_df.columns = [
                'Date', 'Type', 'Price', 'Shares', 'Value', 'P&L', 'P&L %']

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Export CSV
            if export_csv:
                csv = trades_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇  Download trades as CSV",
                    data=csv,
                    file_name=f"{ticker}_trades.csv",
                    mime='text/csv'
                )

        st.markdown("<hr style='border-color:#30363d; margin:24px 0;'>",
                    unsafe_allow_html=True)

else:
    # ── Landing screen ──
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
      <div style="font-size:48px; margin-bottom:16px;">📊</div>
      <h2 style="color:#e6edf3; margin-bottom:8px;">Ready to backtest</h2>
      <p style="color:#8b949e; max-width:460px; margin:0 auto; line-height:1.7;">
        Enter one or more NSE/BSE ticker symbols in the sidebar,
        set your date range and capital, then click
        <b style="color:#238636;">Run Backtest</b>.
      </p>
      <div style="display:flex; justify-content:center; gap:24px; margin-top:40px; flex-wrap:wrap;">
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:140px;">
          <div style="color:#00e676; font-size:20px;">▲</div>
          <div style="color:#e6edf3; font-weight:600; margin:6px 0 4px;">Buy signal</div>
          <div style="color:#8b949e; font-size:12px;">All 5 conditions met</div>
        </div>
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:140px;">
          <div style="color:#ffab40; font-size:20px;">▼</div>
          <div style="color:#e6edf3; font-weight:600; margin:6px 0 4px;">Sell 25%</div>
          <div style="color:#8b949e; font-size:12px;">EMA5 crosses below EMA13</div>
        </div>
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:140px;">
          <div style="color:#ef5350; font-size:20px;">▼</div>
          <div style="color:#e6edf3; font-weight:600; margin:6px 0 4px;">Sell 75%</div>
          <div style="color:#8b949e; font-size:12px;">EMA13 crosses below EMA26</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
