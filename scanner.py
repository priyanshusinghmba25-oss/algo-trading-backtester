import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="NSE Stock Scanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
#  DARK TERMINAL THEME
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
  .stApp { background-color: #0e1117; color: #e0e0e0; }
  section[data-testid="stSidebar"] {
      background-color: #161b22;
      border-right: 1px solid #30363d;
  }
  [data-testid="metric-container"] {
      background-color: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px 16px;
  }
  [data-testid="metric-container"] label {
      color: #8b949e !important; font-size: 12px !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #e6edf3 !important; font-size: 20px !important;
      font-weight: 600 !important;
  }
  .stButton > button {
      background-color: #238636 !important;
      color: #ffffff !important; border: none !important;
      border-radius: 6px !important; font-weight: 600 !important;
      width: 100%; padding: 10px !important; font-size: 15px !important;
  }
  .stButton > button:hover { background-color: #2ea043 !important; }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stMarkdown p { color: #8b949e; }
  hr { border-color: #30363d; }
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
  div[data-testid="stExpander"] {
      background-color: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  STOCK LISTS
# ══════════════════════════════════════════════════════

NIFTY50 = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'HINDUNILVR.NS','SBIN.NS','BHARTIARTL.NS','BAJFINANCE.NS','KOTAKBANK.NS',
    'WIPRO.NS','LT.NS','HCLTECH.NS','ASIANPAINT.NS','AXISBANK.NS',
    'MARUTI.NS','SUNPHARMA.NS','TITAN.NS','ULTRACEMCO.NS','NESTLEIND.NS',
    'TECHM.NS','POWERGRID.NS','NTPC.NS','TATAMOTORS.NS','BAJAJFINSV.NS',
    'ADANIENT.NS','ADANIPORTS.NS','ONGC.NS','JSWSTEEL.NS','TATASTEEL.NS',
    'COALINDIA.NS','DRREDDY.NS','DIVISLAB.NS','CIPLA.NS','EICHERMOT.NS',
    'HEROMOTOCO.NS','BRITANNIA.NS','APOLLOHOSP.NS','BPCL.NS','INDUSINDBK.NS',
    'GRASIM.NS','HINDALCO.NS','TATACONSUM.NS','UPL.NS','SBILIFE.NS',
    'HDFCLIFE.NS','BAJAJ-AUTO.NS','M&M.NS','ITC.NS','LTIM.NS'
]

NIFTY500_EXTRA = [
    'IRCTC.NS','ZOMATO.NS','NYKAA.NS','PAYTM.NS','DMART.NS',
    'PIDILITIND.NS','BERGEPAINT.NS','HAVELLS.NS','VOLTAS.NS','WHIRLPOOL.NS',
    'TRENT.NS','CAMS.NS','CDSL.NS','ANGELONE.NS','MUTHOOTFIN.NS',
    'CHOLAFIN.NS','BAJAJHLDNG.NS','SBICARD.NS','PEL.NS','MANAPPURAM.NS',
    'PAGEIND.NS','AUROPHARMA.NS','TORNTPHARM.NS','ALKEM.NS','LALPATHLAB.NS',
    'METROPOLIS.NS','FORTIS.NS','MAXHEALTH.NS','NHPC.NS','SJVN.NS',
    'IRFC.NS','PFC.NS','RECLTD.NS','HUDCO.NS','CANBK.NS',
    'BANKBARODA.NS','PNB.NS','UNIONBANK.NS','FEDERALBNK.NS','IDFCFIRSTB.NS',
    'PERSISTENT.NS','COFORGE.NS','MPHASIS.NS','LTTS.NS','KPITTECH.NS',
    'TATAELXSI.NS','HAPPSTMNDS.NS','RATEGAIN.NS','KAYNES.NS','AMBER.NS',
    'DIXON.NS','ASTRAZEN.NS','ESCORTS.NS','ASHOKLEY.NS','MOTHERSON.NS',
    'BALKRISIND.NS','MRF.NS','APOLLOTYRE.NS','CEATLTD.NS','JKTYRE.NS',
    'ATUL.NS','DEEPAKNTR.NS','NAVINFLUOR.NS','SRF.NS','BALRAMCHIN.NS',
    'ZYDUSLIFE.NS','GLAXO.NS','PFIZER.NS','ABBOTINDIA.NS','SANOFI.NS',
    'OBEROIRLTY.NS','DLF.NS','GODREJPROP.NS','PRESTIGE.NS','PHOENIXLTD.NS',
    'INDHOTEL.NS','LEMONTREE.NS','CHALET.NS','EIHOTEL.NS','MAHINDCIE.NS'
]

# ══════════════════════════════════════════════════════
#  INDICATOR FUNCTIONS
# ══════════════════════════════════════════════════════

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# ══════════════════════════════════════════════════════
#  FETCH + SCAN SINGLE STOCK
# ══════════════════════════════════════════════════════

def fetch_stock(ticker, days=200):
    """Download last N days of data for a ticker."""
    end   = datetime.today()
    start = end - timedelta(days=days)
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 50:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Open','High','Low','Close','Volume']].copy()
        df.dropna(inplace=True)
        return df
    except:
        return None


def scan_stock(ticker, rsi_daily_min, rsi_weekly_min,
               vol_multiplier, require_bullish):
    df = fetch_stock(ticker, days=365)
    if df is None or len(df) < 60:
        return None

    # ── Indicators ──
    df['ema5']  = calc_ema(df['Close'], 5)
    df['ema13'] = calc_ema(df['Close'], 13)
    df['ema26'] = calc_ema(df['Close'], 26)
    df['rsi']   = calc_rsi(df['Close'], 14)

    # Weekly RSI
    weekly_close = df['Close'].resample('W').last()
    weekly_rsi   = calc_rsi(weekly_close, 14)
    df['rsi_w']  = weekly_rsi.reindex(df.index, method='ffill')

    df['vol_avg'] = df['Volume'].rolling(20).mean()
    df.dropna(inplace=True)

    if len(df) < 10:
        return None

    # ── Check conditions on EVERY bar (not just today) ──
    df['ema_aligned']   = (
        (df['ema5'] > df['ema13']) &
        (df['ema13'] > df['ema26'])
    )
    df['rsi_ok']        = (
        (df['rsi'] > rsi_daily_min) &
        (df['rsi_w'] > rsi_weekly_min)
    )
    df['vol_ok']        = (
        df['Volume'] > df['vol_avg'] * vol_multiplier
    )
    df['bullish_ok']    = (
        df['Close'] > df['Open']
        if require_bullish else True
    )
    df['ema_momentum']  = (
        (df['ema5'] - df['ema13']) >
        (df['ema5'].shift(1) - df['ema13'].shift(1))
    )

    # Master signal — true on every day ALL conditions met
    df['signal'] = (
        df['ema_aligned']  &
        df['rsi_ok']       &
        df['vol_ok']       &
        df['bullish_ok']   &
        df['ema_momentum']
    )

    # ── Find the FIRST day of the current active signal run ──
    # A "run" = consecutive days where signal is True
    # We want the first day of the MOST RECENT run
    if not df['signal'].iloc[-1]:
        # Signal not active today — no current opportunity
        return None

    # Walk backwards to find where current run started
    signal_vals = df['signal'].values
    run_start_idx = len(signal_vals) - 1
    while run_start_idx > 0 and signal_vals[run_start_idx - 1]:
        run_start_idx -= 1

    # First day of this signal run
    first_signal_row  = df.iloc[run_start_idx]
    first_signal_date = df.index[run_start_idx]
    latest            = df.iloc[-1]

    # ── Calculate how late we are ──
    days_since_signal = (df.index[-1] - first_signal_date).days
    entry_price       = round(float(first_signal_row['Open']), 2)
    current_price     = round(float(latest['Close']), 2)
    missed_move       = round(
        (current_price - entry_price) / entry_price * 100, 2)

    # ── Other metrics ──
    price_1d  = round(
        (latest['Close'] - df['Close'].iloc[-2]) /
         df['Close'].iloc[-2] * 100, 2)
    price_1w  = round(
        (latest['Close'] - df['Close'].iloc[-6]) /
         df['Close'].iloc[-6] * 100, 2) if len(df) >= 6 else 0
    price_1m  = round(
        (latest['Close'] - df['Close'].iloc[-22]) /
         df['Close'].iloc[-22] * 100, 2) if len(df) >= 22 else 0

    high_52w  = df['High'].tail(252).max()
    low_52w   = df['Low'].tail(252).min()
    dist_high = round(
        (high_52w - latest['Close']) / high_52w * 100, 2)

    # ── Signal strength score (0-100) ──
    # Higher = stronger signal
    score = 0
    score += min(30, (latest['rsi'] - rsi_daily_min))
    score += min(20, (float(latest['Volume']) /
                      float(latest['vol_avg']) - 1) * 20)
    score += min(20, (latest['ema5'] -
                      latest['ema13']) / latest['ema13'] * 1000)
    score += min(15, max(0, 15 - days_since_signal))
    score += min(15, max(0, (100 - dist_high) / 5))
    score = round(min(100, max(0, score)), 1)

    # Freshness label
    if days_since_signal == 0:
        freshness = "🟢 Fresh today"
    elif days_since_signal <= 3:
        freshness = f"🟡 {days_since_signal}d ago"
    elif days_since_signal <= 7:
        freshness = f"🟠 {days_since_signal}d ago"
    else:
        freshness = f"🔴 {days_since_signal}d ago"

    return {
        'ticker'            : ticker.replace('.NS','').replace('.BO',''),
        'full_ticker'       : ticker,
        'price'             : current_price,
        'entry_price'       : entry_price,
        'entry_date'        : first_signal_date.strftime('%d %b %Y'),
        'days_since_signal' : days_since_signal,
        'missed_move'       : missed_move,
        'freshness'         : freshness,
        'signal_score'      : score,
        'ema5'              : round(float(latest['ema5']), 2),
        'ema13'             : round(float(latest['ema13']), 2),
        'ema26'             : round(float(latest['ema26']), 2),
        'rsi_daily'         : round(float(latest['rsi']), 1),
        'rsi_weekly'        : round(float(latest['rsi_w']), 1),
        'volume'            : int(latest['Volume']),
        'vol_avg'           : int(latest['vol_avg']),
        'vol_ratio'         : round(float(latest['Volume'] /
                                    latest['vol_avg']), 2),
        'chg_1d'            : price_1d,
        'chg_1w'            : price_1w,
        'chg_1m'            : price_1m,
        'high_52w'          : round(float(high_52w), 2),
        'low_52w'           : round(float(low_52w), 2),
        'dist_52h'          : dist_high,
        'df'                : df
    }
# ══════════════════════════════════════════════════════
#  MINI CHART
# ══════════════════════════════════════════════════════

def mini_chart(result):
    """Compact candlestick + EMA chart for one stock."""
    df     = result['df'].tail(60)
    ticker = result['ticker']

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        line_width=1,
        showlegend=False
    ), row=1, col=1)

    # EMAs
    for col_name, color, label in [
        ('ema5','#f48fb1','EMA5'),
        ('ema13','#ffcc02','EMA13'),
        ('ema26','#29b6f6','EMA26')
    ]:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col_name],
            name=label,
            line=dict(color=color, width=1.2),
            showlegend=False
        ), row=1, col=1)

    # Volume
    vol_colors = [
        '#26a69a' if c >= o else '#ef5350'
        for c, o in zip(df['Close'], df['Open'])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=vol_colors,
        opacity=0.7, showlegend=False
    ), row=2, col=1)

    # Mark latest signal point
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[float(df['Low'].iloc[-1]) * 0.97],
        mode='markers',
        marker=dict(symbol='triangle-up', size=12,
                    color='#00e676',
                    line=dict(color='white', width=1)),
        showlegend=False
    ), row=1, col=1)

    fig.update_layout(
        height=280,
        paper_bgcolor='#161b22',
        plot_bgcolor='#161b22',
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#161b22',
                        bordercolor='#30363d',
                        font_color='#e6edf3',
                        font_size=10)
    )
    for i in range(1, 3):
        fig.update_xaxes(
            gridcolor='#21262d', showticklabels=(i==2),
            tickfont=dict(size=9, color='#8b949e'),
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='#21262d',
            tickfont=dict(size=9, color='#8b949e'),
            row=i, col=1
        )
    return fig

# ══════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div style="border-bottom:1px solid #30363d;
            padding-bottom:16px; margin-bottom:20px;">
  <h1 style="margin:0; font-size:24px; color:#e6edf3;">
    🔍 NSE Live Stock Scanner
  </h1>
  <p style="margin:4px 0 0; color:#8b949e; font-size:13px;">
    Scans NSE stocks in real time for your
    Triple EMA + Volume + RSI buy signal
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;
                border-bottom:1px solid #30363d; margin-bottom:16px;">
      <span style="font-size:20px;">⚙️</span>
      <span style="color:#e6edf3; font-size:15px;
                   font-weight:600; margin-left:8px;">Scanner Settings</span>
    </div>
    """, unsafe_allow_html=True)

    # Stock list selector
    st.markdown('<p style="color:#8b949e; font-size:12px; margin-bottom:4px;">UNIVERSE</p>',
                unsafe_allow_html=True)
    universe = st.selectbox(
        "Stock universe",
        ["Nifty 50", "Nifty 500", "Custom list"],
        label_visibility="collapsed"
    )

    custom_tickers = []
    if universe == "Custom list":
        raw_input = st.text_area(
            "Enter tickers (one per line or comma separated)",
            placeholder="RELIANCE.NS\nTCS.NS\nINFY.NS",
            height=120
        )
        if raw_input.strip():
            custom_tickers = [
                t.strip().upper()
                for t in raw_input.replace(',', '\n').splitlines()
                if t.strip()
            ]

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">RSI FILTERS</p>',
                unsafe_allow_html=True)
    rsi_daily_min  = st.slider("Min RSI Daily",
                                40, 80, 60, 1)
    rsi_weekly_min = st.slider("Min RSI Weekly",
                                30, 80, 50, 1)

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">VOLUME FILTER</p>',
                unsafe_allow_html=True)
    vol_multiplier = st.slider(
        "Volume must be N× above 20-day avg",
        0.5, 3.0, 1.0, 0.1
    )

    st.markdown('<p style="color:#8b949e; font-size:12px; margin:12px 0 4px;">OTHER</p>',
                unsafe_allow_html=True)
    require_bullish = st.checkbox(
        "Require bullish candle (close > open)", value=True)
    show_charts     = st.checkbox(
        "Show mini charts for matches", value=True)
    sort_by         = st.selectbox(
        "Sort results by",
        ["RSI Daily", "Volume Ratio",
         "1D Change %", "1W Change %", "Distance from 52W High"]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    scan_btn = st.button("🔍  Run Scanner")

    # Legend
    st.markdown("""
    <div style="margin-top:20px; padding:12px; background:#161b22;
                border:1px solid #30363d; border-radius:8px;">
      <p style="color:#8b949e; font-size:11px; margin:0; line-height:1.8;">
        <b style="color:#e6edf3;">Signal conditions:</b><br>
        ✦ EMA5 > EMA13 > EMA26<br>
        ✦ EMA5 momentum increasing<br>
        ✦ RSI daily > {rsi_d}<br>
        ✦ RSI weekly > {rsi_w}<br>
        ✦ Volume > {vol}× avg<br>
        {bull}
      </p>
    </div>
    """.format(
        rsi_d=rsi_daily_min,
        rsi_w=rsi_weekly_min,
        vol=vol_multiplier,
        bull="✦ Bullish candle required" if require_bullish else ""
    ), unsafe_allow_html=True)

# ── Run Scanner ──
if scan_btn:

    # Build ticker list
    if universe == "Nifty 50":
        ticker_list = NIFTY50
    elif universe == "Nifty 500":
        ticker_list = NIFTY50 + NIFTY500_EXTRA
    else:
        ticker_list = custom_tickers
        if not ticker_list:
            st.error("Please enter at least one ticker in the custom list.")
            st.stop()

    total = len(ticker_list)

    # ── Progress UI ──
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d;
                border-radius:8px; padding:12px 16px; margin-bottom:16px;">
      <span style="color:#8b949e; font-size:13px;">
        Scanning <b style="color:#e6edf3;">{total} stocks</b>
        for your buy signal...
      </span>
    </div>
    """, unsafe_allow_html=True)

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    results       = []

    for i, ticker in enumerate(ticker_list):
        status_text.markdown(
            f'<p style="color:#8b949e; font-size:12px;">'
            f'Checking {ticker} ({i+1}/{total})...</p>',
            unsafe_allow_html=True
        )
        progress_bar.progress((i + 1) / total)

        result = scan_stock(
            ticker,
            rsi_daily_min=rsi_daily_min,
            rsi_weekly_min=rsi_weekly_min,
            vol_multiplier=vol_multiplier,
            require_bullish=require_bullish
        )
        if result:
            results.append(result)

    progress_bar.empty()
    status_text.empty()

    # ── Summary metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Scanned", total)
    col2.metric("Signals Found",  len(results))
    col3.metric("Hit Rate",
                f"{len(results)/total*100:.1f}%" if total > 0 else "0%")
    col4.metric("Scan Time", datetime.now().strftime("%H:%M:%S"))

    if not results:
        st.markdown("""
        <div style="text-align:center; padding:48px;
                    background:#161b22; border:1px solid #30363d;
                    border-radius:8px; margin-top:16px;">
          <div style="font-size:36px; margin-bottom:12px;">📭</div>
          <h3 style="color:#e6edf3; margin-bottom:8px;">
            No signals found right now
          </h3>
          <p style="color:#8b949e; max-width:400px; margin:0 auto;">
            No stocks currently meet all your conditions.
            Try relaxing the RSI or volume filters in the sidebar.
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Sort results ──
    sort_map = {
        "RSI Daily"              : ("rsi_daily",  True),
        "Volume Ratio"           : ("vol_ratio",  True),
        "1D Change %"            : ("chg_1d",     True),
        "1W Change %"            : ("chg_1w",     True),
        "Distance from 52W High" : ("dist_52h",   False)
    }
    sort_key, sort_desc = sort_map[sort_by]
    results.sort(key=lambda x: x[sort_key], reverse=sort_desc)

    st.markdown(f"""
    <div style="margin:16px 0 8px;">
      <span style="color:#00e676; font-size:15px; font-weight:600;">
        ✦ {len(results)} stock{"s" if len(results) != 1 else ""}
        matching your strategy today
      </span>
      <span style="color:#8b949e; font-size:12px; margin-left:12px;">
        sorted by {sort_by}
      </span>
    </div>
    """, unsafe_allow_html=True)

# ── Results table ──
    table_data = []
    for r in results:
        missed_color = ("🔴" if r['missed_move'] > 5
                        else "🟡" if r['missed_move'] > 2
                        else "🟢")
        table_data.append({
            'Stock'            : r['ticker'],
            'Signal'           : r['freshness'],
            'Score'            : f"{r['signal_score']}/100",
            'Entry Date'       : r['entry_date'],
            'Entry Price (₹)'  : f"₹{r['entry_price']:,}",
            'Current Price'    : f"₹{r['price']:,}",
            'Move Since Entry' : f"{missed_color} {r['missed_move']:+.1f}%",
            'RSI Daily'        : r['rsi_daily'],
            'RSI Weekly'       : r['rsi_weekly'],
            'Vol Ratio'        : f"{r['vol_ratio']}×",
            '1D Chg'           : f"{r['chg_1d']:+.2f}%",
            '1W Chg'           : f"{r['chg_1w']:+.2f}%",
            'Dist 52W High'    : f"{r['dist_52h']}%",
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Export CSV
    csv_data = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'df'}
        for r in results
    ])
    st.download_button(
        label="⬇  Download results as CSV",
        data=csv_data.to_csv(index=False).encode('utf-8'),
        file_name=f"scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )

    # ── Mini charts ──
    if show_charts and results:
       st.markdown(f"""
                    <div style="background:#161b22;
                                border:1px solid #30363d;
                                border-top: 2px solid #00e676;
                                border-radius:8px;
                                padding:10px 14px; margin-bottom:6px;">
                      <div style="display:flex;
                                  justify-content:space-between;
                                  align-items:center;">
                        <span style="color:#e6edf3; font-size:15px;
                                     font-weight:700;">{r['ticker']}</span>
                        <span style="color:#8b949e; font-size:11px;">
                          Score: <b style="color:#00e676;">
                          {r['signal_score']}/100</b>
                        </span>
                      </div>
                      <div style="margin-top:5px;">
                        <span style="color:#8b949e; font-size:11px;">
                          {r['freshness']}
                        </span>
                        <span style="color:#8b949e; font-size:11px;
                                     margin-left:10px;">
                          Entry: <b style="color:#ffcc02;">
                          ₹{r['entry_price']:,}</b>
                          on {r['entry_date']}
                        </span>
                      </div>
                      <div style="display:flex; gap:12px;
                                  margin-top:6px; flex-wrap:wrap;">
                        <span style="color:#8b949e; font-size:11px;">
                          Now: ₹{r['price']:,}
                        </span>
                        <span style="color:{'#00e676' if r['missed_move'] >= 0 else '#ef5350'};
                                     font-size:11px; font-weight:600;">
                          {r['missed_move']:+.1f}% since signal
                        </span>
                        <span style="color:#ce93d8; font-size:11px;">
                          RSI {r['rsi_daily']}
                        </span>
                        <span style="color:#ffcc02; font-size:11px;">
                          Vol {r['vol_ratio']}×
                        </span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)


else:
    # ── Landing screen ──
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
      <div style="font-size:52px; margin-bottom:16px;">🔍</div>
      <h2 style="color:#e6edf3; margin-bottom:8px;">
        NSE Live Stock Scanner
      </h2>
      <p style="color:#8b949e; max-width:500px;
                margin:0 auto; line-height:1.7; font-size:14px;">
        Scans Nifty 50, Nifty 500, or your own custom list
        and finds every stock that currently matches your
        Triple EMA + Volume + RSI strategy.
      </p>
      <div style="display:flex; justify-content:center;
                  gap:16px; margin-top:40px; flex-wrap:wrap;">
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:130px;">
          <div style="color:#f48fb1; font-size:18px;
                      font-weight:700;">EMA</div>
          <div style="color:#e6edf3; font-weight:600;
                      margin:6px 0 4px; font-size:13px;">
            5 · 13 · 26
          </div>
          <div style="color:#8b949e; font-size:11px;">
            Bullish alignment
          </div>
        </div>
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:130px;">
          <div style="color:#ce93d8; font-size:18px;
                      font-weight:700;">RSI</div>
          <div style="color:#e6edf3; font-weight:600;
                      margin:6px 0 4px; font-size:13px;">
            Daily & Weekly
          </div>
          <div style="color:#8b949e; font-size:11px;">
            Both above threshold
          </div>
        </div>
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:130px;">
          <div style="color:#ffcc02; font-size:18px;
                      font-weight:700;">VOL</div>
          <div style="color:#e6edf3; font-weight:600;
                      margin:6px 0 4px; font-size:13px;">
            Above average
          </div>
          <div style="color:#8b949e; font-size:11px;">
            Confirms conviction
          </div>
        </div>
        <div style="background:#161b22; border:1px solid #30363d;
                    border-radius:8px; padding:16px 20px; min-width:130px;">
          <div style="color:#00e676; font-size:18px;
                      font-weight:700;">▲</div>
          <div style="color:#e6edf3; font-weight:600;
                      margin:6px 0 4px; font-size:13px;">
            Live Signal
          </div>
          <div style="color:#8b949e; font-size:11px;">
            All conditions met
          </div>
        </div>
      </div>
      <p style="color:#8b949e; margin-top:40px; font-size:13px;">
        Configure your filters in the sidebar and click
        <b style="color:#238636;">Run Scanner</b>
      </p>
    </div>
    """, unsafe_allow_html=True)
