import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SYMBOLS = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "SBI": "SBIN.NS",
    "ITC": "ITC.NS",
    "Axis Bank": "AXISBANK.NS",
    "Custom (khud likhein)": None,
}

# label: (yahoo interval, yahoo period)
TIMEFRAMES = {
    "5 min": ("5m", "5d"),
    "15 min": ("15m", "30d"),
    "30 min": ("30m", "30d"),
    "1 hour": ("60m", "90d"),
    "1 day": ("1d", "1y"),
}


# ---------------------------------------------------------------- data
@st.cache_data(ttl=60, show_spinner=False)
def load_data(symbol, interval, period):
    df = yf.download(
        symbol, interval=interval, period=period,
        auto_adjust=False, progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df


# ---------------------------------------------------------- indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n).values
    hl2 = ((df["High"] + df["Low"]) / 2).values
    close = df["Close"].values
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    direction = np.ones(len(df))
    line = np.full(len(df), np.nan)
    line[0] = flb[0]
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < flb[i] else 1
        else:
            direction[i] = 1 if close[i] > fub[i] else -1
        line[i] = flb[i] if direction[i] == 1 else fub[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.date
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    v = df["Volume"].groupby(day).cumsum()
    return pv / v.replace(0, np.nan)


def add_indicators(df, intraday):
    df = df.copy()
    df["ema9"] = ema(df["Close"], 9)
    df["ema21"] = ema(df["Close"], 21)
    df["rsi"] = rsi(df["Close"])
    macd_line = ema(df["Close"], 12) - ema(df["Close"], 26)
    df["macd"] = macd_line
    df["macd_sig"] = ema(macd_line, 9)
    df["hist"] = df["macd"] - df["macd_sig"]
    df["atr"] = atr(df)
    df["st"], df["dir"] = supertrend(df)
    sma = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std()
    df["bb_up"] = sma + 2 * sd
    df["bb_lo"] = sma - 2 * sd
    df["vwap"] = vwap(df) if intraday else np.nan
    return df


def pivots(daily):
    if len(daily) < 2:
        return None
    p = daily.iloc[-2]
    h, l, c = p["High"], p["Low"], p["Close"]
    pp = (h + l + c) / 3
    return {
        "S2": pp - (h - l), "S1": 2 * pp - h, "Pivot": pp,
        "R1": 2 * pp - l, "R2": pp + (h - l),
    }


# -------------------------------------------------------------- signal
def build_signal(df):
    last = df.iloc[-1]
    votes = {}
    votes["EMA 9/21"] = 1 if last["ema9"] > last["ema21"] else -1
    votes["RSI"] = 1 if last["rsi"] > 55 else (-1 if last["rsi"] < 45 else 0)
    votes["MACD"] = 1 if last["hist"] > 0 else -1
    votes["Supertrend"] = int(last["dir"])
    if pd.notna(last["vwap"]):
        votes["VWAP"] = 1 if last["Close"] > last["vwap"] else -1
    score = sum(votes.values())
    ratio = score / len(votes)
    if ratio >= 0.6:
        bias = "BULLISH 🟢"
    elif ratio <= -0.6:
        bias = "BEARISH 🔴"
    else:
        bias = "NEUTRAL ⚪"
    return bias, votes, ratio


def backtest(df):
    pos = df["dir"].shift(1).fillna(0)
    ret = df["Close"].pct_change().fillna(0)
    strat = pos * ret
    trade_id = (df["dir"].diff().fillna(0) != 0).cumsum()
    trades = strat.groupby(trade_id).apply(lambda x: (1 + x).prod() - 1)
    return {
        "strategy": ((1 + strat).prod() - 1) * 100,
        "buy_hold": ((1 + ret).prod() - 1) * 100,
        "trades": int(len(trades)),
        "win_rate": float((trades > 0).mean() * 100) if len(trades) else 0.0,
    }


# --------------------------------------------------------------- chart
def make_chart(df, intraday):
    d = df.tail(120)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema9"], name="EMA9",
                             line=dict(width=1, color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ema21"], name="EMA21",
                             line=dict(width=1, color="#ff7f0e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["st"], name="Supertrend",
                             line=dict(width=1.5, color="#9467bd", dash="dot")), row=1, col=1)
    if d["vwap"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["vwap"], name="VWAP",
                                 line=dict(width=1, color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["rsi"], name="RSI",
                             line=dict(width=1.2, color="#d62728"), showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
    breaks = [dict(bounds=["sat", "mon"])]
    if intraday:
        breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks, rangeslider_visible=False)
    fig.update_layout(
        height=520, margin=dict(l=5, r=5, t=10, b=5),
        legend=dict(orientation="h", y=1.07, x=0),
        xaxis_rangeslider_visible=False,
    )
    return fig


# ----------------------------------------------------------------- UI
def main():
    st.set_page_config(page_title="Market Signal", page_icon="📈", layout="centered")
    st.title("📈 Market Signal")
    st.caption("Indicator based estimate. Yeh prediction ya financial advice nahi hai.")

    name = st.selectbox("Stock / Index", list(SYMBOLS))
    symbol = SYMBOLS[name]
    if symbol is None:
        symbol = st.text_input("Yahoo symbol (jaise WIPRO.NS)", "WIPRO.NS").strip()
    tf = st.selectbox("Timeframe", list(TIMEFRAMES), index=1)
    interval, period = TIMEFRAMES[tf]
    intraday = interval != "1d"

    with st.expander("Target / Stop-loss settings"):
        sl_mult = st.number_input("Stop-loss (ATR x)", 0.5, 5.0, 1.5, 0.5)
        t1_mult = st.number_input("Target 1 (ATR x)", 0.5, 10.0, 1.5, 0.5)
        t2_mult = st.number_input("Target 2 (ATR x)", 0.5, 10.0, 3.0, 0.5)

    if st.button("🔄 Refresh"):
        st.cache_data.clear()

    with st.spinner("Data la rahe hain..."):
        raw = load_data(symbol, interval, period)

    if raw.empty or len(raw) < 40:
        st.error("Data nahi mila ya kam hai. Symbol ya timeframe badal kar dekhein. "
                 "Market band ho to bhi intraday data kam ho sakta hai.")
        return

    df = add_indicators(raw, intraday)
    last = df.iloc[-1]
    price = float(last["Close"])
    a = float(last["atr"])

    bias, votes, ratio = build_signal(df)

    c1, c2 = st.columns(2)
    c1.metric("Last price", f"{price:,.2f}")
    c2.metric("Bias", bias)
    st.caption(f"Last candle: {df.index[-1]:%d %b %Y %H:%M}  |  Data Yahoo se hai, thoda delay ho sakta hai.")

    st.subheader("Indicators")
    rows = []
    label = {1: "Bullish 🟢", -1: "Bearish 🔴", 0: "Neutral ⚪"}
    for k, v in votes.items():
        rows.append({"Indicator": k, "Signal": label[v]})
    st.table(pd.DataFrame(rows).set_index("Indicator"))
    r = float(last["rsi"])
    note = f"RSI: {r:.1f}"
    if r > 70:
        note += " (overbought, pullback ka risk)"
    elif r < 30:
        note += " (oversold, bounce ho sakta hai)"
    st.write(note + f"  |  ATR: {a:.2f}")

    st.subheader("Target / Stop-loss (ATR se estimate)")
    if ratio >= 0.6:
        st.success(
            f"**Long idea**\n\nEntry: {price:,.2f}\n\n"
            f"Stop-loss: {price - sl_mult * a:,.2f}\n\n"
            f"Target 1: {price + t1_mult * a:,.2f}\n\n"
            f"Target 2: {price + t2_mult * a:,.2f}")
    elif ratio <= -0.6:
        st.error(
            f"**Short idea**\n\nEntry: {price:,.2f}\n\n"
            f"Stop-loss: {price + sl_mult * a:,.2f}\n\n"
            f"Target 1: {price - t1_mult * a:,.2f}\n\n"
            f"Target 2: {price - t2_mult * a:,.2f}")
    else:
        st.info("Indicators ek disha me nahi hain. Abhi koi clear trade idea nahi.")
        st.write(f"Typical range (1 ATR): {price - a:,.2f} se {price + a:,.2f}")

    daily = load_data(symbol, "1d", "1mo") if intraday else raw
    pv = pivots(daily)
    if pv:
        st.subheader("Support / Resistance (pichle din ke H/L/C se)")
        st.table(pd.DataFrame({"Level": {k: round(float(v), 2) for k, v in pv.items()}}))

    st.subheader("Chart")
    st.plotly_chart(make_chart(df, intraday))

    st.subheader("Backtest (Supertrend strategy)")
    bt = backtest(df)
    b1, b2 = st.columns(2)
    b1.metric("Strategy return", f"{bt['strategy']:.2f}%")
    b2.metric("Buy & hold", f"{bt['buy_hold']:.2f}%")
    b3, b4 = st.columns(2)
    b3.metric("Trades", bt["trades"])
    b4.metric("Win rate", f"{bt['win_rate']:.0f}%")
    st.caption("Brokerage, taxes aur slippage count nahi kiye gaye. Purana result aage ki guarantee nahi.")

    st.warning("Sirf seekhne aur analysis ke liye. Trading me paisa doob sakta hai. "
               "Yeh SEBI registered advice nahi hai.")


if __name__ == "__main__":
    main()
