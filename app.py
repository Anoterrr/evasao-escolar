"""Radar de Evasão — app Streamlit.

Uso:  streamlit run app.py
"""
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dados import COLUNAS_ENTRADA, CURSOS, RAIZ, ROTULOS, faixa_risco

st.set_page_config(page_title="Radar de Evasão", page_icon="🎓", layout="wide")

# Cores (paleta de status: sempre acompanhadas do texto da faixa)
COR_FAIXA = {"Baixo": "#0ca30c", "Médio": "#fab219", "Alto": "#d03b3b"}
ICONE_FAIXA = {"Baixo": "🟢", "Médio": "🟡", "Alto": "🔴"}
AZUL = "#2a78d6"

# Grupos de atributos usados na explicação "e se este fator fosse típico?"
GRUPOS = {
    "Desempenho no 1º semestre": [c for c in COLUNAS_ENTRADA if c.startswith("s1_")],
    "Desempenho no 2º semestre": [c for c in COLUNAS_ENTRADA if c.startswith("s2_")],
    "Situação financeira (dívida/mensalidades)": ["devedor", "mensalidades_em_dia"],
    "Bolsa de estudos": ["bolsista"],
    "Idade na matrícula": ["idade_matricula"],
    "Curso e turno": ["curso", "turno_diurno"],
    "Forma de ingresso e opção": ["modo_ingresso", "ordem_opcao"],
    "Notas de entrada": ["nota_admissao", "nota_escolaridade_anterior", "escolaridade_anterior"],
    "Perfil pessoal": ["genero_masculino", "estado_civil", "deslocado",
                       "necessidades_especiais", "internacional", "nacionalidade"],
    "Família (escolaridade/ocupação dos pais)": ["escolaridade_mae", "escolaridade_pai",
                                                 "ocupacao_mae", "ocupacao_pai"],
    "Contexto econômico": ["taxa_desemprego", "taxa_inflacao", "pib"],
}


@st.cache_resource
def carregar_modelo():
    caminho = RAIZ / "modelos" / "modelo.joblib"
    if not caminho.exists():
        st.error("Modelo não encontrado. Rode primeiro: `python treinar.py`")
        st.stop()
    return joblib.load(caminho)


@st.cache_data
def carregar_metricas():
    return json.loads((RAIZ / "modelos" / "metricas.json").read_text(encoding="utf-8"))


ARTEFATO = carregar_modelo()
MODELO, LIMIARES, REF = ARTEFATO["pipeline"], ARTEFATO["limiares"], ARTEFATO["referencia"]


def prever(df: pd.DataFrame) -> pd.Series:
    return pd.Series(MODELO.predict_proba(df[COLUNAS_ENTRADA])[:, 1], index=df.index)


def explicar(aluno: pd.Series) -> pd.DataFrame:
    """Para cada grupo de fatores, recalcula o risco trocando os valores do aluno
    pelos valores típicos (mediana/moda do treino). A diferença mostra quanto
    aquele grupo empurra o risco para cima (+) ou para baixo (−)."""
    base = aluno[COLUNAS_ENTRADA].to_frame().T
    linhas = [base]
    for colunas in GRUPOS.values():
        variante = base.copy()
        for c in colunas:
            variante[c] = REF[c]
        linhas.append(variante)
    probs = MODELO.predict_proba(pd.concat(linhas, ignore_index=True).astype(float))[:, 1]
    efeito = probs[0] - probs[1:]
    return (pd.DataFrame({"fator": list(GRUPOS), "efeito": efeito})
            .sort_values("efeito", key=abs, ascending=False))


def grafico_explicacao(expl: pd.DataFrame) -> go.Figure:
    expl = expl[expl["efeito"].abs() >= 0.005].iloc[::-1]
    cores = [COR_FAIXA["Alto"] if v > 0 else COR_FAIXA["Baixo"] for v in expl["efeito"]]
    fig = go.Figure(go.Bar(
        x=expl["efeito"] * 100, y=expl["fator"], orientation="h", marker_color=cores,
        text=[f"{v:+.1f} p.p." for v in expl["efeito"] * 100], textposition="outside",
        hovertemplate="%{y}: %{x:+.1f} pontos percentuais<extra></extra>"))
    fig.update_layout(height=60 + 34 * len(expl), margin=dict(l=10, r=40, t=10, b=30),
                      xaxis_title="Efeito no risco (pontos percentuais)", bargap=0.35)
    return fig


def mostrar_risco(prob: float) -> None:
    faixa = faixa_risco(prob, LIMIARES)
    c1, c2 = st.columns([1, 2])
    c1.metric("Probabilidade de evasão", f"{prob:.0%}")
    c2.markdown(f"### {ICONE_FAIXA[faixa]} Risco **{faixa}**")
    if faixa == "Alto":
        c2.write("Recomendação: contato imediato da coordenação e encaminhamento "
                 "para apoio pedagógico, psicológico ou financeiro.")
    elif faixa == "Médio":
        c2.write("Recomendação: acompanhar de perto no próximo semestre "
                 "(tutoria, monitoria e conversa com o aluno).")
    else:
        c2.write("Sem sinais relevantes de evasão no momento.")


# ---------------------------------------------------------------------------
st.title("🎓 Radar de Evasão Escolar")
st.caption(f"Previsão do risco de evasão no ensino superior com **{ARTEFATO['nome']}**. "
           f"Faixas: 🔴 Alto ≥ {LIMIARES['alto']:.0%} · 🟡 Médio ≥ {LIMIARES['medio']:.0%} · 🟢 Baixo")

aba_turma, aba_aluno, aba_modelo, aba_sobre = st.tabs(
    ["📋 Painel da turma", "🔍 Avaliar um aluno", "📊 Desempenho do modelo", "ℹ️ Sobre"])

# ----------------------------- PAINEL DA TURMA -----------------------------
with aba_turma:
    arquivo = st.file_uploader(
        "Envie um CSV com os alunos (mesmas colunas de `data/amostra_turma.csv`) "
        "ou use a turma de demonstração.", type="csv")
    if arquivo:
        turma = pd.read_csv(arquivo)
        faltando = [c for c in COLUNAS_ENTRADA if c not in turma.columns]
        if faltando:
            st.error(f"Colunas ausentes no CSV: {', '.join(faltando)}")
            st.stop()
    else:
        turma = pd.read_csv(RAIZ / "data" / "amostra_turma.csv")
        st.info(f"Turma de demonstração: {len(turma)} alunos do conjunto de teste "
                "(o modelo nunca os viu no treino).")
    if "id_aluno" not in turma.columns:
        turma.insert(0, "id_aluno", [f"L{i:04d}" for i in range(len(turma))])

    turma["prob_evasao"] = prever(turma)
    turma["risco"] = turma["prob_evasao"].apply(lambda p: faixa_risco(p, LIMIARES))
    turma["nome_curso"] = turma["curso"].map(CURSOS).fillna("Desconhecido")

    contagem = turma["risco"].value_counts()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Alunos analisados", len(turma))
    k2.metric("🔴 Risco alto", int(contagem.get("Alto", 0)))
    k3.metric("🟡 Risco médio", int(contagem.get("Médio", 0)))
    k4.metric("🟢 Risco baixo", int(contagem.get("Baixo", 0)))

    # Composição de risco por curso (barras empilhadas em %)
    por_curso = (pd.crosstab(turma["nome_curso"], turma["risco"], normalize="index")
                 .reindex(columns=["Alto", "Médio", "Baixo"], fill_value=0)
                 .sort_values(["Alto", "Médio"]))
    n_curso = turma["nome_curso"].value_counts()
    fig = go.Figure()
    for faixa in ["Alto", "Médio", "Baixo"]:
        fig.add_bar(name=f"{ICONE_FAIXA[faixa]} {faixa}", y=por_curso.index,
                    x=por_curso[faixa] * 100, orientation="h",
                    marker=dict(color=COR_FAIXA[faixa], line=dict(width=1, color="rgba(0,0,0,0)")),
                    customdata=n_curso.reindex(por_curso.index),
                    hovertemplate=f"%{{y}}<br>{faixa}: %{{x:.0f}}%% de %{{customdata}} alunos<extra></extra>")
    fig.update_layout(barmode="stack", height=40 + 28 * len(por_curso), bargap=0.25,
                      xaxis=dict(title="% dos alunos do curso", range=[0, 100]),
                      legend=dict(orientation="h", y=1.08, x=0, traceorder="normal"),
                      margin=dict(l=10, r=10, t=40, b=30))
    st.subheader("Risco por curso")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Lista de alunos priorizada")
    filtro = st.multiselect("Mostrar faixas", ["Alto", "Médio", "Baixo"], default=["Alto", "Médio"])
    colunas_tabela = ["id_aluno", "risco", "prob_evasao", "nome_curso", "s1_aprovadas",
                      "s1_matriculadas", "s2_aprovadas", "s2_matriculadas",
                      "s2_nota_media", "mensalidades_em_dia", "devedor"]
    if "situacao_real" in turma.columns:
        colunas_tabela.append("situacao_real")
    lista = turma[turma["risco"].isin(filtro)].sort_values("prob_evasao", ascending=False)
    exibicao = lista[colunas_tabela].copy()
    exibicao["risco"] = exibicao["risco"].map(lambda f: f"{ICONE_FAIXA[f]} {f}")
    exibicao["prob_evasao"] = exibicao["prob_evasao"] * 100
    st.dataframe(
        exibicao, hide_index=True, width="stretch",
        column_config={
            "id_aluno": "Aluno", "risco": "Risco", "nome_curso": "Curso",
            "prob_evasao": st.column_config.ProgressColumn(
                "Prob. evasão", format="%.0f%%", min_value=0, max_value=100),
            "s1_aprovadas": "Aprov. 1º sem", "s1_matriculadas": "Matric. 1º sem",
            "s2_aprovadas": "Aprov. 2º sem", "s2_matriculadas": "Matric. 2º sem",
            "s2_nota_media": st.column_config.NumberColumn("Nota 2º sem", format="%.1f"),
            "mensalidades_em_dia": st.column_config.CheckboxColumn("Mensalid. em dia"),
            "devedor": st.column_config.CheckboxColumn("Devedor"),
            "situacao_real": "Situação real",
        })
    st.download_button("⬇️ Baixar lista (CSV)",
                       lista.drop(columns=["nome_curso"]).to_csv(index=False).encode("utf-8"),
                       "alunos_em_risco.csv", "text/csv")

    if len(lista):
        st.subheader("Por que este aluno está em risco?")
        escolhido = st.selectbox("Aluno", lista["id_aluno"])
        aluno = lista.set_index("id_aluno").loc[escolhido]
        mostrar_risco(float(aluno["prob_evasao"]))
        st.plotly_chart(grafico_explicacao(explicar(aluno)), width="stretch")
        st.caption("Vermelho = fator que aumenta o risco; verde = fator que reduz. "
                   "Cada barra mostra quanto o risco mudaria se aquele grupo de fatores "
                   "tivesse o valor típico dos alunos.")

# ----------------------------- AVALIAR UM ALUNO ----------------------------
with aba_aluno:
    st.write("Preencha os dados do aluno. Campos não exibidos usam o valor mais comum da base.")
    with st.form("form_aluno"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Perfil**")
            curso = st.selectbox("Curso", list(CURSOS), format_func=CURSOS.get,
                                 index=list(CURSOS).index(int(REF["curso"])))
            diurno = st.radio("Turno", [1, 0], format_func={1: "Diurno", 0: "Noturno"}.get,
                              horizontal=True)
            genero = st.radio("Gênero", [0, 1], format_func={0: "Feminino", 1: "Masculino"}.get,
                              horizontal=True)
            idade = st.number_input("Idade na matrícula", 17, 70, int(REF["idade_matricula"]))
            nota_adm = st.slider("Nota de admissão (0–200)", 95.0, 190.0,
                                 float(REF["nota_admissao"]), 0.5)
            deslocado = st.checkbox("Mora fora da cidade de origem")
        with b:
            st.markdown("**Situação financeira**")
            bolsista = st.checkbox("Bolsista")
            em_dia = st.checkbox("Mensalidades em dia", value=True)
            devedor = st.checkbox("Possui dívidas com a instituição")
            st.markdown("**1º semestre**")
            s1_mat = st.number_input("Disciplinas matriculadas", 0, 26, 6, key="s1m")
            s1_apr = st.number_input("Disciplinas aprovadas", 0, 26, 5, key="s1a")
            s1_nota = st.slider("Nota média (0–20)", 0.0, 20.0, 12.5, 0.1, key="s1n")
        with c:
            st.markdown("**2º semestre**")
            s2_mat = st.number_input("Disciplinas matriculadas", 0, 26, 6, key="s2m")
            s2_apr = st.number_input("Disciplinas aprovadas", 0, 26, 5, key="s2a")
            s2_nota = st.slider("Nota média (0–20)", 0.0, 20.0, 12.5, 0.1, key="s2n")
            s2_sem = st.number_input("Disciplinas sem avaliação", 0, 12, 0, key="s2s")
        enviado = st.form_submit_button("Calcular risco", type="primary")

    if enviado:
        if s1_apr > s1_mat or s2_apr > s2_mat:
            st.error("Aprovadas não pode ser maior que matriculadas.")
        else:
            aluno = pd.Series(REF, dtype=float)
            aluno.update(pd.Series({
                "curso": curso, "turno_diurno": diurno, "genero_masculino": genero,
                "idade_matricula": idade, "nota_admissao": nota_adm,
                "deslocado": int(deslocado), "bolsista": int(bolsista),
                "mensalidades_em_dia": int(em_dia), "devedor": int(devedor),
                "s1_matriculadas": s1_mat, "s1_aprovadas": s1_apr, "s1_nota_media": s1_nota,
                "s1_avaliacoes": s1_mat + 2, "s1_sem_avaliacao": 0,
                "s2_matriculadas": s2_mat, "s2_aprovadas": s2_apr, "s2_nota_media": s2_nota,
                "s2_avaliacoes": max(s2_mat - s2_sem, 0) + 2, "s2_sem_avaliacao": s2_sem,
            }, dtype=float))
            prob = float(prever(aluno.to_frame().T).iloc[0])
            mostrar_risco(prob)
            st.plotly_chart(grafico_explicacao(explicar(aluno)), width="stretch")
            st.caption("Vermelho = fator que aumenta o risco; verde = fator que reduz.")

# --------------------------- DESEMPENHO DO MODELO --------------------------
with aba_modelo:
    m = carregar_metricas()
    st.write(f"Treino: **{m['n_treino']}** alunos · Teste: **{m['n_teste']}** alunos · "
             f"Taxa de evasão na base: **{m['taxa_evasao']:.1%}**")

    st.subheader("1. Comparação de modelos (validação cruzada 5-fold, no treino)")
    comp = pd.DataFrame(m["comparacao_cv"]).T
    comp.columns = ["ROC-AUC", "Acurácia", "Precisão", "Recall", "F1"]
    st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#cde2fb"),
                 width="stretch")
    st.caption(f"Escolhido pelo maior ROC-AUC: **{m['modelo_escolhido']}**. "
               "Métricas da tabela usam limiar 0,5.")

    st.subheader("2. Resultado no conjunto de teste")
    t1, t2 = st.columns(2)
    for col, chave, titulo in [(t1, "alto", "Limiar 'Alto' (máx. F1)"),
                               (t2, "medio", "Limiar 'Médio' (máx. F2, prioriza recall)")]:
        r = m["teste"][chave]
        with col:
            st.markdown(f"**{titulo}: {m['limiares'][chave]:.0%}**")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Acurácia", f"{r['acuracia']:.1%}")
            x2.metric("Precisão", f"{r['precisao']:.1%}")
            x3.metric("Recall", f"{r['recall']:.1%}")
            x4.metric("F1", f"{r['f1']:.3f}")
            mc = r["matriz_confusao"]
            rotulos = ["Não evadiu", "Evadiu"]
            fig = go.Figure(go.Heatmap(
                z=mc, x=rotulos, y=rotulos, colorscale=[[0, "#cde2fb"], [1, "#184f95"]],
                showscale=False, text=mc, texttemplate="%{text}", textfont=dict(size=18),
                xgap=2, ygap=2,
                hovertemplate="Real: %{y}<br>Previsto: %{x}<br>%{z} alunos<extra></extra>"))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis_title="Previsto", yaxis=dict(title="Real", autorange="reversed"))
            st.plotly_chart(fig, width="stretch", key=f"mc_{chave}")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"**Curva ROC (AUC = {m['teste']['alto']['roc_auc']:.3f})**")
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                        line=dict(color="#8a8984", dash="dash", width=1), hoverinfo="skip")
        fig.add_scatter(x=m["roc"]["fpr"], y=m["roc"]["tpr"], mode="lines",
                        name=m["modelo_escolhido"], line=dict(color=AZUL, width=2),
                        hovertemplate="Falsos positivos: %{x:.1%}<br>Verdadeiros positivos: %{y:.1%}<extra></extra>")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                          xaxis_title="Taxa de falsos positivos",
                          yaxis_title="Taxa de verdadeiros positivos (recall)",
                          legend=dict(x=0.55, y=0.08))
        st.plotly_chart(fig, width="stretch")
    with g2:
        st.markdown("**Atributos mais importantes (queda no AUC ao embaralhar)**")
        imp = pd.DataFrame(m["importancias"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=imp["importancia"], y=imp["atributo"].map(ROTULOS), orientation="h",
            marker_color=AZUL, error_x=dict(type="data", array=imp["desvio"], color="#8a8984"),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), bargap=0.35,
                          xaxis_title="Queda no ROC-AUC")
        st.plotly_chart(fig, width="stretch")

# --------------------------------- SOBRE -----------------------------------
with aba_sobre:
    st.markdown("""
**Problema.** A evasão no ensino superior gera custo para a instituição e para o aluno.
Identificar cedo quem está em risco permite que a coordenação aja antes da desistência.

**Dados.** *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository
(ID 697): 4.424 alunos de graduação de uma instituição de ensino superior em Portugal, com 36
atributos (demográficos, socioeconômicos e de desempenho acadêmico no 1º e no 2º semestre).
Autores: Martins, Tolledo, Machado, Baptista e Realinho (2021). DOI: 10.24432/C5MC89.

**Formulação.** Classificação binária: *Evadiu* (Dropout) × *Não evadiu* (Graduate ou Enrolled).

**Pipeline.**
1. Tradução das colunas e engenharia de atributos (taxa de aprovação por semestre, variação
   da nota, total de reprovações, semestre zerado, situação financeira ruim).
2. Comparação de Regressão Logística, Random Forest e XGBoost com validação cruzada estratificada.
3. Escolha de dois limiares na validação cruzada: F1 (risco alto) e F2 (risco médio).
4. Avaliação final em 20% dos dados separados no início (teste).
5. Explicação individual: quanto o risco muda se cada grupo de fatores fosse típico.

**Limitações.** Os dados são de uma única instituição portuguesa; o modelo usa informações até o
fim do 1º ano, então serve para alerta a partir desse momento. O modelo indica correlação,
não causa, e deve apoiar (não substituir) a decisão da coordenação.
""")
