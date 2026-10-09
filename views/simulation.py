"""単勝の期待値で毎日買い続けた場合の資金の増減 (締切時オッズでのシミュレーション)。

データは odds-backtest の Actions が serve/sim_daily.csv に書き出したもの (boatrace/sim.py)。
"""
import altair as alt
import pandas as pd
import streamlit as st

from boatrace import config, sim

st.title("シミュレーション")
st.caption("このアプリの1着確率で、毎日決まった予算を単勝で買い続けたら資金がどう増減したか。お金は賭けていません。")

path = config.SERVE_DIR / "sim_daily.csv"
if not path.exists():
    st.info("まだシミュレーションの結果がありません（GitHub Actions の odds-backtest を単勝で実行すると作られます）。")
    st.stop()

df = pd.read_csv(path)
df["race_date"] = pd.to_datetime(df["race_date"])
df["予算"] = df["budget"].map(lambda b: f"{b:,}円")
order = [f"{b:,}円" for b in sorted(df["budget"].unique())]

st.markdown("""
**買い方**：その日のレースで **期待値（1着確率 × 単勝オッズ）が基準以上の艇** を、
1日の予算を均等に割って買います（100円単位、余りは買わない）。予算が足りない日は期待値の高い順に100円ずつ買います。
""")

colors = alt.Scale(domain=order, range=["#9ecae1", "#4292c6", "#e6550d", "#a50f15"][:len(order)])
for rule, g in df.groupby("rule", sort=True):
    st.subheader(f"期待値 {rule.replace('EV', '')} の艇を買う")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(y="y:Q")
    line = alt.Chart(g).mark_line(strokeWidth=2).encode(
        x=alt.X("race_date:T", title="日付", axis=alt.Axis(format="%Y/%m")),
        y=alt.Y("cum_profit:Q", title="累計の収支（円）", axis=alt.Axis(format=",.0f")),
        color=alt.Color("予算:N", title="1日の予算", scale=colors, sort=order),
        tooltip=[alt.Tooltip("race_date:T", title="日付", format="%Y/%m/%d"), alt.Tooltip("予算:N"),
                 alt.Tooltip("cum_profit:Q", title="累計の収支", format=",.0f"),
                 alt.Tooltip("bets:Q", title="この日の点数"), alt.Tooltip("profit:Q", title="この日の収支", format=",.0f")],
    )
    st.altair_chart(zero + line, width="stretch")
    s = sim.summary(g.drop(columns=["予算"]))
    st.dataframe(
        s.drop(columns=["ルール"]).style.format({"1日の予算": "{:,}円", "購入額": "{:,.0f}円", "払戻": "{:,.0f}円",
                                                "収支": "{:+,.0f}円", "回収率": "{:.1%}", "最大の落ち込み": "{:,.0f}円"}),
        hide_index=True, width="stretch")

days = df["race_date"].nunique()
per_day = df.drop_duplicates("race_date")["races"].mean()
st.markdown(f"""
### 読み方の注意
- 期間は {df['race_date'].min():%Y/%m/%d}〜{df['race_date'].max():%Y/%m/%d}（{days}日）。
  オッズは検証期間から **無作為に選んだレース** のもので、**1日あたり平均 {per_day:.1f} レース** しかありません。
  実際に1日の全レース（約140）から買うと、点数が増えて日ごとの上下はもっと小さくなります。
- 予算を均等に割るので、**4本の線はほぼ同じ形で、大きさが違うだけ** です（違いは100円単位の端数）。
  予算を増やしても増えるか減るかは変わらず、振れ幅の金額が大きくなります。
- **実際より良く見えやすい** 結果です。
  - オッズは **締切時** のもの。実際に買う締切前のオッズとは違い、締切直前に人気が動くので、実際の回収率は下がりやすい。
  - 基準（1.0・1.2）は、同じデータの結果を見て選んでいます。
- 確率は「朝の予想」モデル（2025年9月までで学習）。検証期間のデータは学習に使っていません。
- 実際の締切5分前のオッズでの記録は「EV記録」のページで見られます。
""")
