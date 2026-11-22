import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
from sklearn.neighbors import BallTree

st.set_page_config(page_title="서울 커피 프랜차이즈 입지 전쟁", layout="wide")

COLORS = {
    "스타벅스": [0, 98, 65],
    "메가MGC커피": [255, 200, 0],
    "컴포즈커피": [240, 130, 0],
    "빽다방": [200, 160, 0],
    "투썸플레이스": [200, 30, 50],
    "이디야커피": [20, 40, 120],
    "매머드커피": [110, 70, 40],
    "할리스": [150, 20, 30],
    "더벤티": [120, 60, 160],
    "커피빈": [60, 60, 60],
}
HEX = {k: "#%02x%02x%02x" % tuple(v) for k, v in COLORS.items()}


@st.cache_data
def load():
    return pd.read_csv("data/cafes_brand.csv")


def nearest_dist(a, b, same):
    """a의 각 점포에서 b의 가장 가까운 점포까지 거리(m)."""
    tree = BallTree(np.radians(b[["위도", "경도"]].values), metric="haversine")
    k = 2 if same else 1
    d, _ = tree.query(np.radians(a[["위도", "경도"]].values), k=k)
    return d[:, k - 1] * 6371000


df = load()
brands = list(COLORS)

st.title("☕ 서울 커피 프랜차이즈 입지 전쟁")
st.caption("누가 누구 옆에 붙는가? 브랜드 간 최근접 거리로 입지 전략을 비교합니다.")

# ---------- 입력 위젯 ----------
with st.sidebar:
    st.header("조건 설정")
    base = st.selectbox("기준 브랜드", brands, index=1)
    target = st.selectbox("비교 브랜드 (옆에 있는지 볼 대상)", brands, index=0)
    gus = sorted(df["시군구명"].dropna().unique())
    sel_gu = st.multiselect("시군구 (비우면 서울 전체)", gus, default=[])
    radius = st.slider("반경 (m)", 50, 1000, 200, step=50)

base_df = df[df["브랜드"] == base]
if sel_gu:
    base_df = base_df[base_df["시군구명"].isin(sel_gu)]
target_df = df[df["브랜드"] == target]  # 경계 효과를 막으려 비교 대상은 서울 전체 사용

if base_df.empty:
    st.warning("선택한 조건에 기준 브랜드 점포가 없습니다.")
    st.stop()

base_df = base_df.copy()
base_df["최근접거리"] = nearest_dist(base_df, target_df, same=(base == target))

# ---------- 핵심 숫자 ----------
within = (base_df["최근접거리"] <= radius).mean() * 100
c1, c2, c3 = st.columns(3)
c1.metric(f"{base} 점포 수", f"{len(base_df):,}개")
c2.metric(f"반경 {radius}m 안에 {target}가 있는 비율", f"{within:.1f}%")
c3.metric("최근접 거리 중앙값", f"{base_df['최근접거리'].median():.0f} m")

# ---------- 차트 1: 지도 ----------
st.subheader("① 점포 위치 지도")
show = pd.concat([base_df.assign(브랜드=base), target_df if not sel_gu else
                  target_df[target_df["시군구명"].isin(sel_gu)]]).drop_duplicates(
    subset=["상호명", "지점명", "경도", "위도"])
show = show.copy()
show["color"] = show["브랜드"].map(COLORS)
layer = pdk.Layer("ScatterplotLayer", data=show, get_position="[경도, 위도]",
                  get_fill_color="color", get_radius=60, pickable=True, opacity=0.8)
view = pdk.ViewState(latitude=float(show["위도"].mean()),
                     longitude=float(show["경도"].mean()),
                     zoom=10.5 if not sel_gu else 12)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                         tooltip={"text": "{브랜드}\n{상호명} {지점명}"}))
st.caption("색: " + " · ".join(f"{b}" for b in show["브랜드"].unique()))

# ---------- 차트 2: 거리 분포 ----------
left, right = st.columns(2)
with left:
    st.subheader(f"② {base} → 가장 가까운 {target}까지 거리")
    fig = px.histogram(base_df[base_df["최근접거리"] <= 2000], x="최근접거리",
                       nbins=40, color_discrete_sequence=[HEX[base]],
                       labels={"최근접거리": "거리(m)", "count": "점포 수"})
    fig.add_vline(x=radius, line_dash="dash", annotation_text=f"{radius}m")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 차트 3: 브랜드별 근접 비교 ----------
with right:
    st.subheader(f"③ {base} 점포 반경 {radius}m 안에 다른 브랜드가 있는 비율")
    rows = []
    for b in brands:
        d = nearest_dist(base_df, df[df["브랜드"] == b], same=(b == base))
        rows.append({"브랜드": b, "비율(%)": (d <= radius).mean() * 100})
    fig2 = px.bar(pd.DataFrame(rows), x="브랜드", y="비율(%)", color="브랜드",
                  color_discrete_map=HEX)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ---------- 해석/기준 ----------
with st.expander("데이터 출처와 집계 기준"):
    st.markdown(
        "- 출처: 소상공인시장진흥공단 상가(상권)정보, 공공데이터포털 (기준: 2026년 9월 파일, 서울)\n"
        "- 카페 업종(상권업종 중분류에 '커피/카페' 포함) 중 상호명에 브랜드명이 들어간 점포만 집계\n"
        "- 거리는 점포 좌표 간 직선거리(haversine)이며, 도보 거리와 다를 수 있음\n"
        "- 한계: 한 시점의 스냅샷이고, 상호명 표기가 다른 점포는 누락될 수 있음"
    )
