"""Treina e compara modelos de previsão de evasão e salva o melhor.

Uso:  python treinar.py
Gera: modelos/modelo.joblib, modelos/metricas.json, data/amostra_turma.csv
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.dados import COLUNAS_CATEGORICAS, COLUNAS_ENTRADA, RAIZ, carregar_dataset, criar_atributos

SEMENTE = 42
PASTA_MODELOS = RAIZ / "modelos"


def montar_modelos(proporcao_negativos: float) -> dict[str, Pipeline]:
    engenharia = FunctionTransformer(criar_atributos)

    # Modelo linear precisa de one-hot nos códigos e padronização dos números
    preproc_linear = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10),
          COLUNAS_CATEGORICAS)],
        remainder=StandardScaler(),
    )

    return {
        "Regressão Logística": Pipeline([
            ("atributos", engenharia),
            ("preproc", preproc_linear),
            ("modelo", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("atributos", engenharia),
            ("modelo", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                n_jobs=-1, random_state=SEMENTE)),
        ]),
        "XGBoost": Pipeline([
            ("atributos", engenharia),
            ("modelo", XGBClassifier(
                n_estimators=400, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=proporcao_negativos,
                eval_metric="logloss", random_state=SEMENTE, n_jobs=-1)),
        ]),
    }


def metricas_binarias(y, prob, limiar) -> dict:
    pred = (prob >= limiar).astype(int)
    return {
        "roc_auc": roc_auc_score(y, prob),
        "acuracia": accuracy_score(y, pred),
        "precisao": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def melhor_limiar(y, prob, beta: float) -> float:
    """Limiar que maximiza o F-beta. beta>1 dá mais peso ao recall."""
    precisao, recall, limiares = precision_recall_curve(y, prob)
    b2 = beta ** 2
    fbeta = (1 + b2) * precisao * recall / np.clip(b2 * precisao + recall, 1e-9, None)
    return float(limiares[np.argmax(fbeta[:-1])])


def main() -> None:
    df = carregar_dataset()
    X, y = df[COLUNAS_ENTRADA], df["evadiu"]
    print(f"Dataset: {len(df)} alunos | evasão: {y.mean():.1%}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEMENTE)

    proporcao = float((y_treino == 0).sum() / (y_treino == 1).sum())
    modelos = montar_modelos(proporcao)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

    # 1) Comparação por validação cruzada (apenas no conjunto de treino)
    comparacao, probs_oof = {}, {}
    for nome, pipe in modelos.items():
        prob = cross_val_predict(pipe, X_treino, y_treino, cv=cv, method="predict_proba")[:, 1]
        probs_oof[nome] = prob
        comparacao[nome] = metricas_binarias(y_treino, prob, 0.5)
        m = comparacao[nome]
        print(f"[CV] {nome:20s} AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}  "
              f"Recall={m['recall']:.3f}  Precisão={m['precisao']:.3f}")

    melhor = max(comparacao, key=lambda n: comparacao[n]["roc_auc"])

    # Dois limiares, escolhidos na CV (sem olhar o teste):
    #  - "alto"  maximiza F1 -> equilíbrio, alertas confiáveis (poucos falsos alarmes)
    #  - "medio" maximiza F2 -> prioriza recall, quase nenhum evasor passa despercebido
    limiares = {
        "alto": melhor_limiar(y_treino, probs_oof[melhor], beta=1.0),
        "medio": melhor_limiar(y_treino, probs_oof[melhor], beta=2.0),
    }
    print(f"\nMelhor modelo: {melhor} | limiares: alto={limiares['alto']:.3f} "
          f"médio={limiares['medio']:.3f}")

    # 2) Treino final e avaliação no conjunto de teste (nunca visto)
    pipe = modelos[melhor].fit(X_treino, y_treino)
    prob_teste = pipe.predict_proba(X_teste)[:, 1]
    teste = {}
    for faixa, lim in limiares.items():
        teste[faixa] = metricas_binarias(y_teste, prob_teste, lim)
        teste[faixa]["matriz_confusao"] = confusion_matrix(
            y_teste, (prob_teste >= lim).astype(int)).tolist()
        print(f"[TESTE ≥{faixa}] " + "  ".join(
            f"{k}={v:.3f}" for k, v in teste[faixa].items() if k != "matriz_confusao"))
    fpr, tpr, _ = roc_curve(y_teste, prob_teste)

    # 3) Importância por permutação (quanto o AUC cai ao embaralhar cada coluna)
    imp = permutation_importance(pipe, X_teste, y_teste, scoring="roc_auc",
                                 n_repeats=10, random_state=SEMENTE, n_jobs=-1)
    importancias = sorted(
        ({"atributo": c, "importancia": float(m), "desvio": float(s)}
         for c, m, s in zip(COLUNAS_ENTRADA, imp.importances_mean, imp.importances_std)),
        key=lambda d: d["importancia"], reverse=True)

    # 4) Valores de referência (mediana/moda) usados no app para preencher campos
    #    não informados e para a explicação "e se este fator fosse típico?"
    referencia = {}
    for c in COLUNAS_ENTRADA:
        if c in COLUNAS_CATEGORICAS or X_treino[c].nunique() <= 2:
            referencia[c] = X_treino[c].mode().iloc[0].item()
        else:
            referencia[c] = float(X_treino[c].median())

    PASTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "nome": melhor, "limiares": limiares,
                 "referencia": referencia}, PASTA_MODELOS / "modelo.joblib")

    metricas = {
        "modelo_escolhido": melhor,
        "limiares": limiares,
        "n_treino": len(X_treino),
        "n_teste": len(X_teste),
        "taxa_evasao": float(y.mean()),
        "comparacao_cv": comparacao,
        "teste": teste,
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "importancias": importancias,
    }
    (PASTA_MODELOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Turma de demonstração: alunos do conjunto de teste, com a situação real
    amostra = X_teste.copy()
    amostra.insert(0, "id_aluno", [f"A{i:04d}" for i in X_teste.index])
    amostra["situacao_real"] = df.loc[X_teste.index, "situacao"].map(
        {"Dropout": "Evadiu", "Graduate": "Formou-se", "Enrolled": "Matriculado"})
    amostra.to_csv(RAIZ / "data" / "amostra_turma.csv", index=False)

    print(f"\nSalvo em {PASTA_MODELOS}/ e data/amostra_turma.csv")


if __name__ == "__main__":
    main()
