import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from mplsoccer import Pitch, VerticalPitch
from scipy.spatial import ConvexHull
import warnings
import requests
import io
import sys
import subprocess
import os
import matplotlib.pyplot as plt

# --- CORREÇÃO DE AMBIENTE (Graphviz) ---
try:
    import graphviz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "graphviz"])
    import graphviz

caminho_bin = r'C:\Program Files\Graphviz\bin'
if os.path.exists(caminho_bin):
    if caminho_bin not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + caminho_bin

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="SCOUT ANALYTICS PRO", layout="wide", page_icon="⚽")
warnings.filterwarnings("ignore")

# --- 2. LINKS DE DADOS ---
URL_LOGO = "https://drive.google.com/thumbnail?id=1dhQPX8Lx0RBx7OB--TgtyVdVUQ3UyLmi&sz=w1000"
URL_CAMPANHA = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=1241314919&single=true&output=csv"
URL_CLASSIFICACAO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=1057602586&single=true&output=csv"
URL_CARTOES = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=1354689566&single=true&output=csv"
URL_LIDERES = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=0&single=true&output=csv"
URL_ELENCO = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=340587611&single=true&output=csv"
URL_RADAR_LINHA = "https://docs.google.com/spreadsheets/d/1mWXD93c1IMIrTIbDQkwrdh-LSCBMczKtOEvFKCXUWKs/export?format=csv"
URL_RADAR_GOLEIROS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=1857870630&single=true&output=csv"
URL_MVE = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSOoM4UZ8IxNDqxB9uoIqxaYjEbMjfjz2vxiW3yzuOgAY_DfeGajiPW075soqh0yVIbWUlHBTsqxdGE/pub?gid=1682508291&single=true&output=csv"

# --- CORES E TEMA ---
CORES_GERAIS = {"Background": "#0F172A", "Sidebar": "#1E293B", "Destaque": "#38BDF8", "Texto": "#F8FAFC", "Card": "#334155"}
CORES_EQUIPES = {"A.A. Serrana/FZ": "#90EE90", "ACM/Estacaville": "#228B22", "América FC": "#FF4500", "Aviação F.C.": "#1E90FF", "Caxias F.C.": "#E2E8F0", "E.C. Panagua": "#3B82F6", "G.E. Pirabeiraba": "#EF4444", "Pará FC": "#0EA5E9", "Serbi": "#10B981", "Sercos": "#F59E0B"}

TRADUCAO_METRICAS = {
    'Passe Progressivo': 'Prog. Pass', 'Passe Ultimo Terço': 'Final 1/3 Pass',
    'Passe Chave': 'Key Pass', 'Duelos Aereos': 'Aerial Duels',
    'Ações Defensivas': 'Def. Actions', 'Desarmes': 'Tackles',
    'Conduções': 'Carries', 'Finalizações': 'Shots',
    'Cruzamentos': 'Crosses', 'Dribles': 'Dribbles',
    'Interceptações': 'Intercepts', 'Perdas de Bola': 'Ball Losses'
}

st.markdown(f"""
    <style>
    .stApp {{ background-color: {CORES_GERAIS['Background']} !important; color: {CORES_GERAIS['Texto']} !important; }}
    h1, h2, h3, h4 {{ color: #E2E8F0 !important; font-family: 'Inter', sans-serif; text-transform: uppercase; font-weight: 700; }}
    section[data-testid="stSidebar"] {{ background-color: {CORES_GERAIS['Sidebar']} !important; border-right: 1px solid #334155; }}
    div[data-testid="stMetric"] {{ background-color: {CORES_GERAIS['Card']} !important; border-left: 4px solid {CORES_GERAIS['Destaque']} !important; border-radius: 6px; }}
    div.stButton > button {{ background-color: {CORES_GERAIS['Card']}; color: {CORES_GERAIS['Destaque']}; border: 1px solid {CORES_GERAIS['Destaque']}; }}
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES ---
def converter_link_drive(url):
    if pd.isna(url): return url
    url_str = str(url).strip()
    if "export=download" in url_str or "export?format=csv" in url_str: return url_str
    if "/d/" in url_str:
        try:
            file_id = url_str.split("/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except: return url_str
    return url_str

def corrigir_link_imagem(url):
    if pd.isna(url): return None
    url_str = str(url).strip()
    if "/d/" in url_str:
        try:
            file_id = url_str.split("/d/")[1].split("/")[0]
            return f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
        except: return url_str
    return url_str

@st.cache_data(ttl=60)
def carregar_planilha_csv(url):
    try: 
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def carregar_radares_csv(url_linha, url_goleiros):
    try:
        try: df_l = pd.read_csv(url_linha)
        except: df_l = pd.read_excel(url_linha)
        df_g = pd.read_csv(url_goleiros)
        if not df_l.empty: df_l = df_l.set_index(df_l.columns[0])
        if not df_g.empty: df_g = df_g.set_index(df_g.columns[0])
        return df_l, df_g
    except: return pd.DataFrame(), pd.DataFrame()

def separar_dados_atleta(df, atleta, tipo='linha'):
    if df is None or atleta not in df.index: return None, None, None
    try:
        row = df.loc[atleta].iloc[0] if isinstance(df.loc[atleta], pd.DataFrame) else df.loc[atleta]
        cols = df.columns.tolist()
        idx_jogo = -1
        for i, c in enumerate(cols):
            if str(c).upper().strip() in ['JOGO', 'GAME', 'MATCH', 'PARTIDA']:
                idx_jogo = i; break
        if idx_jogo == -1: idx_jogo = len(cols)//2
        return row.iloc[:idx_jogo//2], row.iloc[idx_jogo//2:idx_jogo], row.iloc[idx_jogo:]
    except: return None, None, None

def traduzir_indices(serie):
    if serie is None: return serie
    novos_indices = [TRADUCAO_METRICAS.get(i.strip(), i.strip()) for i in serie.index]
    serie.index = novos_indices
    return serie

@st.cache_data(ttl=300)
def carregar_scouts_jogos(links, nomes, df_elenco_ref):
    if not links: return pd.DataFrame()
    dfs = []
    dict_nomes = {}
    if not df_elenco_ref.empty:
        col_arquivo = next((c for c in df_elenco_ref.columns if 'Arquivo' in c or 'Ref' in c), df_elenco_ref.columns[0])
        col_real = next((c for c in df_elenco_ref.columns if 'Nome' in c and 'Real' in c), df_elenco_ref.columns[1])
        dict_nomes = dict(zip(df_elenco_ref[col_arquivo], df_elenco_ref[col_real]))

    for url, nome in zip(links, nomes):
        if pd.isna(url): continue
        try:
            final_url = converter_link_drive(url)
            r = requests.get(final_url)
            if r.status_code == 200:
                xls = pd.ExcelFile(io.BytesIO(r.content))
                for sheet in xls.sheet_names:
                    if "Resumo" in sheet: continue
                    df_temp = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=30)
                    h_idx = -1
                    for i, row in df_temp.iterrows():
                        if any(x in str(row.values) for x in ['Time', 'Tempo', 'X', 'Field X', 'FieldX']):
                            h_idx = i; break
                    if h_idx != -1:
                        data = pd.read_excel(xls, sheet_name=sheet, header=h_idx)
                        data['Jogo_Ref'] = nome
                        cols_map = {}
                        for c in data.columns:
                            c_s = str(c).strip()
                            if c_s in ['X', 'Field X', 'FieldX']: cols_map[c] = 'FieldX'
                            if c_s in ['Y', 'Field Y', 'FieldY']: cols_map[c] = 'FieldY'
                            if c_s in ['Tempo (s)', 'Time']: cols_map[c] = 'Tempo'
                            if c_s in ['Player', 'Atleta']: cols_map[c] = 'Jogadores'
                        data.rename(columns=cols_map, inplace=True)
                        dfs.append(data)
        except: continue
    
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if 'Tempo' in df.columns:
        def cvt_min(t):
            try:
                t = str(t).replace(',', '.')
                p = t.split(':')
                if len(p)==2: return float(p[0]) + float(p[1])/60
                if len(p)==3: return float(p[0])*60 + float(p[1]) + float(p[2])/60
                return float(t)/60
            except: return 0.0
        df['Minuto'] = df['Tempo'].apply(cvt_min)
    if 'FieldX' in df.columns:
        df['FieldX'] = pd.to_numeric(df['FieldX'].astype(str).str.replace(',', '.'), errors='coerce')
        df['FieldY'] = pd.to_numeric(df['FieldY'].astype(str).str.replace(',', '.'), errors='coerce')
        max_x = df['FieldX'].max()
        if max_x <= 1.1: df['FieldX'] *= 120; df['FieldY'] *= 80
        elif max_x <= 105: df['FieldX'] = (df['FieldX']/100)*120; df['FieldY'] = (df['FieldY']/100)*80
        df['FieldY'] = 80 - df['FieldY']
    if 'Jogadores' in df.columns:
        df['Passador'] = df['Jogadores'].astype(str).apply(lambda x: x.split('|')[0].strip())
        df['Receptor'] = df['Jogadores'].astype(str).apply(lambda x: x.split('|')[1].strip() if '|' in x else None)
        if dict_nomes:
            df['Passador'] = df['Passador'].map(dict_nomes).fillna(df['Passador'])
            df['Receptor'] = df['Receptor'].map(dict_nomes).fillna(df['Receptor'])
        df['Jogadores'] = df['Passador']
    return df

def plot_radar(cats, vals, title, color='#38BDF8', max_v=100):
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill='toself', fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}', line=dict(color=color, width=2)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_v], color='#94A3B8'), bgcolor='#1E293B'), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'), height=300, margin=dict(t=20, b=20, l=20, r=20))
    return fig

# --- ESTADO ---
if 'tela' not in st.session_state: st.session_state.tela = 'Home'
if 'atleta_sel' not in st.session_state: st.session_state.atleta_sel = None

# LOAD
df_camp = carregar_planilha_csv(URL_CAMPANHA)
df_class = carregar_planilha_csv(URL_CLASSIFICACAO)
df_cart = carregar_planilha_csv(URL_CARTOES)
df_lid = carregar_planilha_csv(URL_LIDERES)
df_ele = carregar_planilha_csv(URL_ELENCO)
df_lin, df_gol = carregar_radares_csv(URL_RADAR_LINHA, URL_RADAR_GOLEIROS)

# --- SIDEBAR ---
st.sidebar.image(corrigir_link_imagem(URL_LOGO), width=160)
st.sidebar.markdown("### DATA INTELLIGENCE")
if st.sidebar.button("HOME"): st.session_state.tela = 'Home'
if st.sidebar.button("OVERVIEW"): st.session_state.tela = 'Equipe'
if st.sidebar.button("SQUAD"): st.session_state.tela = 'Grid'
st.sidebar.divider()

col_link = None
for c in df_camp.columns:
    if "Link" in c and "LongoMatch" in c:
        col_link = c; break

if not df_camp.empty and col_link:
    # --- CORREÇÃO DO ERRO KEYERROR: ADVERSÁRIO vs OPPONENT ---
    col_adv = next((c for c in df_camp.columns if c in ['Adversário', 'Opponent', 'Adversario']), df_camp.columns[2])
    df_camp['Jogo_Label'] = "Game " + df_camp.index.astype(str) + " vs " + df_camp[col_adv].astype(str)
    
    jogo_sel = st.sidebar.selectbox("Select Game", ["Season"] + df_camp['Jogo_Label'].tolist())
    
    if jogo_sel == "Season": 
        df_jogo = carregar_scouts_jogos(df_camp[col_link].tolist(), df_camp['Jogo_Label'].tolist(), df_ele)
    else:
        f = df_camp[df_camp['Jogo_Label'] == jogo_sel]
        df_jogo = carregar_scouts_jogos(f[col_link].tolist(), f['Jogo_Label'].tolist(), df_ele)
        
    st.sidebar.markdown("### Current Minute")
    min_slider = st.sidebar.slider("Minutes", 0, 100, (0, 95))
    if not df_jogo.empty:
        df_jogo_filtrado = df_jogo[(df_jogo['Minuto'] >= min_slider[0]) & (df_jogo['Minuto'] <= min_slider[1])]
    else:
        df_jogo_filtrado = df_jogo
else:
    df_jogo_filtrado = pd.DataFrame()

# --- TELAS ---
if st.session_state.tela == 'Home':
    st.title("WELCOME TO DATA INTELLIGENCE")
    c1, c2 = st.columns(2)
    c1.info("System connected. Use the sidebar to navigate.")

elif st.session_state.tela == 'Equipe':
    st.title("Tactical and Collective Analysis")
    if not df_camp.empty:
        st.subheader("Season Performance")
        col_res = next((c for c in df_camp.columns if c in ['Resultado', 'Result']), df_camp.columns[2])
        col_gols = next((c for c in df_camp.columns if c in ['Gols Pro', 'GF', 'Goals For']), 'GF')
        
        try:
            vitorias = len(df_camp[df_camp[col_res].astype(str).str.contains('Vitória|Win', na=False, case=False)])
            jogos = len(df_camp)
            empates = len(df_camp[df_camp[col_res].astype(str).str.contains('Empate|Draw', na=False, case=False)])
            pontos = (vitorias * 3) + empates
            aproveitamento = (pontos / (jogos * 3)) * 100 if jogos > 0 else 0
            gols_pro = pd.to_numeric(df_camp[col_gols], errors='coerce').sum()
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate", f"{aproveitamento:.1f}%")
            m2.metric("Wins", vitorias)
            m3.metric("Matches", jogos)
            m4.metric("Goals For", int(gols_pro))
        except: st.warning("Metrics data unavailable.")
        st.divider()

    t1, t2, t3 = st.tabs(["League Table", "Statistics", "Pitch"])
    with t1:
        if not df_class.empty:
            cols_r = [c for c in df_class.columns if "Rodada" in c or "Round" in c]
            if cols_r:
                df_long = df_class.melt(id_vars=[df_class.columns[0]], value_vars=cols_r, var_name='Rodada', value_name='Posicao')
                df_long['Rodada_Num'] = df_long['Rodada'].str.extract('(\d+)').astype(int)
                fig = px.line(df_long, x="Rodada_Num", y="Posicao", color=df_class.columns[0], markers=True, color_discrete_map=CORES_EQUIPES)
                fig.update_yaxes(autorange="reversed"); st.plotly_chart(fig, use_container_width=True)
    with t2:
        c1, c2 = st.columns(2)
        c1.dataframe(df_lid, use_container_width=True); c2.dataframe(df_cart, use_container_width=True)
    with t3:
        if not df_jogo_filtrado.empty:
            st.markdown(f"**Analyzing minute: {min_slider[0]}' - {min_slider[1]}'**")
            c_mapa, c_passes = st.columns(2)
            with c_mapa:
                st.subheader("Action Map")
                pitch = Pitch(pitch_type='statsbomb', pitch_color='#1E293B', line_color='#64748B')
                fig, ax = pitch.draw(figsize=(10, 7))
                pitch.scatter(df_jogo_filtrado.FieldX, df_jogo_filtrado.FieldY, ax=ax, c='#38BDF8', s=30, alpha=0.6, edgecolors='white')
                pitch.kdeplot(df_jogo_filtrado.FieldX, df_jogo_filtrado.FieldY, ax=ax, cmap='GnBu', fill=True, alpha=0.3, levels=50)
                st.pyplot(fig)
            with c_passes:
                st.subheader("Pass Network")
                if 'Receptor' in df_jogo_filtrado.columns:
                    conexoes = df_jogo_filtrado.dropna(subset=['Receptor'])
                    if not conexoes.empty:
                        avg_loc = conexoes.groupby('Passador')[['FieldX', 'FieldY']].mean()
                        pass_count = conexoes.groupby(['Passador', 'Receptor']).size().reset_index(name='qtd')
                        pitch_net = Pitch(pitch_type='statsbomb', pitch_color='#0F172A', line_color='#334155')
                        fig_net, ax_net = pitch_net.draw(figsize=(10, 7))
                        pass_net = pass_count.merge(avg_loc, left_on='Passador', right_index=True).merge(avg_loc, left_on='Receptor', right_index=True, suffixes=['_start', '_end'])
                        pitch_net.lines(pass_net.FieldX_start, pass_net.FieldY_start, pass_net.FieldX_end, pass_net.FieldY_end, lw=pass_net.qtd*0.8, color='#38BDF8', alpha=0.5, ax=ax_net)
                        pitch_net.scatter(avg_loc.FieldX, avg_loc.FieldY, s=200, color='#1E293B', edgecolors='#38BDF8', linewidth=2, ax=ax_net)
                        for pl, row in avg_loc.iterrows():
                            pitch_net.annotate(pl, (row.FieldX, row.FieldY-3), ax=ax_net, color='white', ha='center', fontsize=9, weight='bold')
                        st.pyplot(fig_net)
                    else: st.warning("No pass data available.")
                else: st.info("Receptor data missing.")
        else: st.warning("Select 'Season' or a Game to view tactical analysis.")

elif st.session_state.tela == 'Grid':
    st.title("Squad")
    if not df_ele.empty:
        col_nome_real = next((c for c in df_ele.columns if 'Nome' in c and 'Real' in c), df_ele.columns[1])
        cols = st.columns(5)
        for i, at in enumerate(df_ele[df_ele['Status']!='Inativo'].to_dict('records')):
            with cols[i%5]:
                with st.container():
                    st.image(corrigir_link_imagem(at.get('Foto_URL')) or URL_LOGO, width=100)
                    st.write(f"**{at[col_nome_real]}**")
                    if st.button("View", key=f"b_{i}"): st.session_state.atleta_sel = at[col_nome_real]; st.session_state.tela='Player'; st.rerun()

elif st.session_state.tela == 'Player':
    p = st.session_state.atleta_sel
    if st.button("⬅️ Back"): st.session_state.tela='Grid'; st.rerun()
    col_nome_real = next((c for c in df_ele.columns if 'Nome' in c and 'Real' in c), df_ele.columns[1])
    dados_atleta = df_ele[df_ele[col_nome_real] == p].iloc[0] if not df_ele.empty else {}
    st.title(p)
    col_foto, col_info, col_extra = st.columns([1, 2, 2])
    with col_foto:
        if 'Foto_URL' in dados_atleta: st.image(corrigir_link_imagem(dados_atleta['Foto_URL']), width=150)
        else: st.markdown("👤")
    with col_info:
        st.metric("Position", dados_atleta.get('Posicao', '-'))
        st.metric("Preferred Foot", dados_atleta.get('Pe_Dominante', '-'))
        st.metric("Number", dados_atleta.get('Numero', '-'))
    st.divider()
    tipo = 'goleiro' if 'Goleiro' in str(dados_atleta.get('Posicao', '')) else 'linha'
    df_r = df_gol if tipo=='goleiro' else df_lin
    if p in df_r.index:
        da, dv, dm = separar_dados_atleta(df_r, p, tipo)
        da = traduzir_indices(da)
        dv = traduzir_indices(dv)
        c1, c2 = st.columns(2)
        if da is not None: c1.plotly_chart(plot_radar(pd.to_numeric(da, errors='coerce').fillna(0).index, pd.to_numeric(da, errors='coerce').fillna(0).values, "Technical"), use_container_width=True)
        if dv is not None: c2.plotly_chart(plot_radar(pd.to_numeric(dv, errors='coerce').fillna(0).index, pd.to_numeric(dv, errors='coerce').fillna(0).values, "Volume", '#10B981', pd.to_numeric(dv, errors='coerce').fillna(0).max()), use_container_width=True)
    else: st.warning(f"Radar data not found for {p}.")
    if not df_jogo_filtrado.empty:
        st.divider()
        st.subheader(f"Action Map: {p} ({min_slider[0]}'-{min_slider[1]}')")
        df_p = df_jogo_filtrado[df_jogo_filtrado['Jogadores'] == p]
        col_field, col_data = st.columns([2, 1])
        with col_field:
            pitch = Pitch(pitch_type='statsbomb', pitch_color='#0F172A', line_color='#334155')
            fig, ax = pitch.draw(figsize=(10, 6))
            if len(df_p) > 0:
                pitch.kdeplot(df_p.FieldX, df_p.FieldY, ax=ax, cmap='GnBu', fill=True, alpha=0.4, levels=30)
                pitch.scatter(df_p.FieldX, df_p.FieldY, ax=ax, c='#38BDF8', s=40, edgecolors='white')
                if len(df_p) >= 3:
                    try:
                        points = df_p[['FieldX', 'FieldY']].values
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        poly = plt.Polygon(hull_points, facecolor='none', edgecolor='#38BDF8', alpha=0.8, linestyle='--', linewidth=2)
                        ax.add_patch(poly)
                    except: pass
            else: st.info("No actions recorded.")
            st.pyplot(fig)
        with col_data:
            st.markdown("##### Player Connections")
            if 'Receptor' in df_jogo_filtrado.columns:
                recebeu = df_jogo_filtrado[df_jogo_filtrado['Receptor'] == p]['Passador'].value_counts().head(5)
                st.write("📥 **Received most from:**")
                if not recebeu.empty: st.dataframe(recebeu, use_container_width=True)
                tocou = df_jogo_filtrado[df_jogo_filtrado['Passador'] == p]['Receptor'].value_counts().head(5)
                st.write("📤 **Passed most to:**")
                if not tocou.empty: st.dataframe(tocou, use_container_width=True)
    try:
        df_mve = carregar_planilha_csv(URL_MVE)
        mve_p = df_mve[df_mve['Atleta'].str.strip() == str(p).strip()]
        if not mve_p.empty:
            st.divider(); st.subheader("MVE - Vulnerability Map")
            dot = graphviz.Digraph(graph_attr={'rankdir':'LR', 'bgcolor':'transparent'})
            dot.node('root', p, shape='circle', style='filled', fillcolor='#38BDF8', fontcolor='black')
            for _, r in mve_p.iterrows():
                ac, ind, ca = str(r.get('Ação','')), str(r.get('Indicador','')), str(r.get('Caos','1'))
                cc = "#EF4444" if "3" in ca else "#F59E0B" if "2" in ca else "#10B981"
                dot.node(ac, ac, shape='box', style='rounded,filled', fillcolor='#1E293B', fontcolor='white')
                dot.node(ind, ind, shape='note', style='filled', fillcolor='#334155', fontcolor='white')
                dot.edge('root', ac); dot.edge(ac, ind, color=cc, penwidth='2')
            st.graphviz_chart(dot)
    except: pass
