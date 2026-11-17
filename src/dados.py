"""Carregamento do dataset, tradução das colunas e engenharia de atributos.

Dataset: "Predict Students' Dropout and Academic Success" (UCI, ID 697).
Martins, Tolledo, Machado, Baptista & Realinho (2021). DOI: 10.24432/C5MC89
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DATASET = RAIZ / "data" / "dataset_uci_697.csv"

# Nome original (inglês) -> nome usado no projeto (português)
TRADUCAO_COLUNAS = {
    "Marital status": "estado_civil",
    "Application mode": "modo_ingresso",
    "Application order": "ordem_opcao",
    "Course": "curso",
    "Daytime/evening attendance": "turno_diurno",
    "Previous qualification": "escolaridade_anterior",
    "Previous qualification (grade)": "nota_escolaridade_anterior",
    "Nacionality": "nacionalidade",
    "Mother's qualification": "escolaridade_mae",
    "Father's qualification": "escolaridade_pai",
    "Mother's occupation": "ocupacao_mae",
    "Father's occupation": "ocupacao_pai",
    "Admission grade": "nota_admissao",
    "Displaced": "deslocado",
    "Educational special needs": "necessidades_especiais",
    "Debtor": "devedor",
    "Tuition fees up to date": "mensalidades_em_dia",
    "Gender": "genero_masculino",
    "Scholarship holder": "bolsista",
    "Age at enrollment": "idade_matricula",
    "International": "internacional",
    "Curricular units 1st sem (credited)": "s1_creditadas",
    "Curricular units 1st sem (enrolled)": "s1_matriculadas",
    "Curricular units 1st sem (evaluations)": "s1_avaliacoes",
    "Curricular units 1st sem (approved)": "s1_aprovadas",
    "Curricular units 1st sem (grade)": "s1_nota_media",
    "Curricular units 1st sem (without evaluations)": "s1_sem_avaliacao",
    "Curricular units 2nd sem (credited)": "s2_creditadas",
    "Curricular units 2nd sem (enrolled)": "s2_matriculadas",
    "Curricular units 2nd sem (evaluations)": "s2_avaliacoes",
    "Curricular units 2nd sem (approved)": "s2_aprovadas",
    "Curricular units 2nd sem (grade)": "s2_nota_media",
    "Curricular units 2nd sem (without evaluations)": "s2_sem_avaliacao",
    "Unemployment rate": "taxa_desemprego",
    "Inflation rate": "taxa_inflacao",
    "GDP": "pib",
    "Target": "situacao",
}

COLUNAS_ENTRADA = [v for v in TRADUCAO_COLUNAS.values() if v != "situacao"]

# Atributos nominais (códigos sem ordem) -> one-hot nos modelos lineares
COLUNAS_CATEGORICAS = [
    "estado_civil", "modo_ingresso", "curso", "escolaridade_anterior",
    "nacionalidade", "escolaridade_mae", "escolaridade_pai",
    "ocupacao_mae", "ocupacao_pai",
]

# Códigos de curso conforme a documentação da UCI
CURSOS = {
    33: "Tecnologias de Produção de Biocombustíveis",
    171: "Design de Animação e Multimédia",
    8014: "Serviço Social (noturno)",
    9003: "Agronomia",
    9070: "Design de Comunicação",
    9085: "Enfermagem Veterinária",
    9119: "Engenharia Informática",
    9130: "Equinicultura",
    9147: "Gestão",
    9238: "Serviço Social",
    9254: "Turismo",
    9500: "Enfermagem",
    9556: "Higiene Oral",
    9670: "Gestão de Publicidade e Marketing",
    9773: "Jornalismo e Comunicação",
    9853: "Educação Básica",
    9991: "Gestão (noturno)",
}

# Nomes legíveis para exibir no app e nos gráficos
ROTULOS = {
    "estado_civil": "Estado civil",
    "modo_ingresso": "Modo de ingresso",
    "ordem_opcao": "Ordem de opção do curso",
    "curso": "Curso",
    "turno_diurno": "Turno diurno",
    "escolaridade_anterior": "Escolaridade anterior",
    "nota_escolaridade_anterior": "Nota da escolaridade anterior",
    "nacionalidade": "Nacionalidade",
    "escolaridade_mae": "Escolaridade da mãe",
    "escolaridade_pai": "Escolaridade do pai",
    "ocupacao_mae": "Ocupação da mãe",
    "ocupacao_pai": "Ocupação do pai",
    "nota_admissao": "Nota de admissão",
    "deslocado": "Mora fora da cidade de origem",
    "necessidades_especiais": "Necessidades educacionais especiais",
    "devedor": "Devedor",
    "mensalidades_em_dia": "Mensalidades em dia",
    "genero_masculino": "Gênero masculino",
    "bolsista": "Bolsista",
    "idade_matricula": "Idade na matrícula",
    "internacional": "Estudante internacional",
    "s1_creditadas": "1º sem: disciplinas creditadas",
    "s1_matriculadas": "1º sem: disciplinas matriculadas",
    "s1_avaliacoes": "1º sem: avaliações feitas",
    "s1_aprovadas": "1º sem: disciplinas aprovadas",
    "s1_nota_media": "1º sem: nota média",
    "s1_sem_avaliacao": "1º sem: disciplinas sem avaliação",
    "s2_creditadas": "2º sem: disciplinas creditadas",
    "s2_matriculadas": "2º sem: disciplinas matriculadas",
    "s2_avaliacoes": "2º sem: avaliações feitas",
    "s2_aprovadas": "2º sem: disciplinas aprovadas",
    "s2_nota_media": "2º sem: nota média",
    "s2_sem_avaliacao": "2º sem: disciplinas sem avaliação",
    "taxa_desemprego": "Taxa de desemprego",
    "taxa_inflacao": "Taxa de inflação",
    "pib": "PIB",
    # atributos criados na engenharia de atributos
    "s1_taxa_aprovacao": "1º sem: taxa de aprovação",
    "s2_taxa_aprovacao": "2º sem: taxa de aprovação",
    "variacao_nota": "Variação da nota (2º − 1º sem)",
    "total_reprovadas": "Total de disciplinas não aprovadas",
    "total_sem_avaliacao": "Total de disciplinas sem avaliação",
    "zerou_semestre": "Nenhuma aprovação em algum semestre",
    "situacao_financeira_ruim": "Devedor ou mensalidade atrasada",
}


def carregar_dataset(caminho: Path = CAMINHO_DATASET) -> pd.DataFrame:
    """Lê o CSV original da UCI e devolve com colunas em português.

    Cria a coluna `evadiu` (1 = Dropout, 0 = Graduate/Enrolled).
    """
    df = pd.read_csv(caminho, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=TRADUCAO_COLUNAS)
    df["evadiu"] = (df["situacao"] == "Dropout").astype(int)
    return df


def _taxa(aprovadas: pd.Series, matriculadas: pd.Series) -> pd.Series:
    return np.where(matriculadas > 0, aprovadas / matriculadas.replace(0, 1), 0.0)


def criar_atributos(X: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de atributos: indicadores de desempenho e de risco financeiro."""
    X = X[COLUNAS_ENTRADA].copy()
    X["s1_taxa_aprovacao"] = _taxa(X["s1_aprovadas"], X["s1_matriculadas"])
    X["s2_taxa_aprovacao"] = _taxa(X["s2_aprovadas"], X["s2_matriculadas"])
    X["variacao_nota"] = X["s2_nota_media"] - X["s1_nota_media"]
    X["total_reprovadas"] = (
        (X["s1_matriculadas"] - X["s1_aprovadas"]).clip(lower=0)
        + (X["s2_matriculadas"] - X["s2_aprovadas"]).clip(lower=0)
    )
    X["total_sem_avaliacao"] = X["s1_sem_avaliacao"] + X["s2_sem_avaliacao"]
    X["zerou_semestre"] = (
        ((X["s1_matriculadas"] > 0) & (X["s1_aprovadas"] == 0))
        | ((X["s2_matriculadas"] > 0) & (X["s2_aprovadas"] == 0))
    ).astype(int)
    X["situacao_financeira_ruim"] = (
        (X["devedor"] == 1) | (X["mensalidades_em_dia"] == 0)
    ).astype(int)
    return X


def faixa_risco(prob: float, limiares: dict) -> str:
    """Faixa de risco a partir dos dois limiares salvos no treino.

    Alto  : prob >= limiar que maximiza F1 (alerta confiável)
    Médio : prob >= limiar que maximiza F2 (prioriza não deixar ninguém passar)
    Baixo : abaixo disso
    """
    if prob >= limiares["alto"]:
        return "Alto"
    if prob >= limiares["medio"]:
        return "Médio"
    return "Baixo"
