# 🎓 Radar de Evasão Escolar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anoterrr/evasao-escolar/blob/main/colab.ipynb)

Sistema com Inteligência Artificial que estima a **probabilidade de um aluno de graduação evadir**
e alerta a coordenação, priorizando quem precisa de intervenção primeiro.

> Trabalho final — Sistemas de Inteligência Artificial · Tema 3: Previsão de Risco de Evasão Escolar

**Integrantes:** _(preencher os 4 nomes)_

## Como rodar

O projeto usa o [uv](https://docs.astral.sh/uv/) para gerenciar o Python e as dependências.
O `pyproject.toml` e o `uv.lock` fixam as versões exatas, e o `.python-version` fixa o Python 3.12
(o uv baixa esse Python sozinho se ele não estiver instalado).

```bash
# 1. Instalar o uv (uma vez só)
curl -LsSf https://astral.sh/uv/install.sh | sh                                   # Linux/macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

# 2. Clonar e rodar
git clone https://github.com/Anoterrr/evasao-escolar.git
cd evasao-escolar
uv run streamlit run app.py        # cria o ambiente, instala tudo e abre http://localhost:8501
```

O modelo já treinado está em `modelos/`. Para reproduzir o treino:

```bash
uv run python treinar.py           # treina, compara modelos e salva em modelos/
```

### Google Colab

O notebook [`colab.ipynb`](colab.ipynb) percorre o projeto passo a passo: dados, análise exploratória, engenharia
de atributos, treino, resultados e previsões. É só clicar no selo **Open in Colab** no topo e usar
*Ambiente de execução → Executar tudo*. A última célula abre o app Streamlit por um link temporário
(túnel cloudflared), que funciona enquanto a sessão do Colab estiver ativa.

<details>
<summary>Sem uv (pip tradicional)</summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
</details>

## O problema

A evasão no ensino superior desperdiça recursos da instituição e interrompe a formação do aluno.
Quando a coordenação percebe a desistência, normalmente já é tarde. O sistema usa dados que a
instituição já possui (notas, aprovações, situação financeira, perfil) para **antecipar** o risco.

## Dados

**Predict Students' Dropout and Academic Success**: UCI Machine Learning Repository, ID 697.
Martins, M. V.; Tolledo, D.; Machado, J.; Baptista, L. M. T.; Realinho, V. (2021).
DOI: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)

- 4.424 alunos de 17 cursos de graduação de uma instituição de ensino superior em Portugal
- 36 atributos: demográficos, socioeconômicos, macroeconômicos e desempenho no 1º e no 2º semestre
- Sem valores ausentes
- Alvo original com 3 classes (Dropout 32,1%, Graduate 49,9%, Enrolled 17,9%), convertido em
  **binário**: *Evadiu* (Dropout) × *Não evadiu* (Graduate + Enrolled)
- Escalas: notas de semestre de 0 a 20 e nota de admissão de 0 a 200 (padrão português)

## Metodologia

| Etapa | O que foi feito |
|---|---|
| Limpeza | Tradução das colunas para português e remoção de espaços e caracteres invisíveis nos nomes |
| Engenharia de atributos | Taxa de aprovação por semestre, variação da nota entre semestres, total de disciplinas não aprovadas e sem avaliação, "zerou algum semestre", "situação financeira ruim" (`src/dados.py`) |
| Divisão | 80% treino / 20% teste, estratificado, semente 42 |
| Modelos comparados | Regressão Logística (one-hot + padronização), Random Forest, XGBoost, com tratamento de desbalanceamento (`class_weight` / `scale_pos_weight`) |
| Seleção | Validação cruzada estratificada 5-fold **somente no treino**, critério ROC-AUC |
| Limiares | Escolhidos na validação cruzada: **Alto** = limiar que maximiza F1, **Médio** = limiar que maximiza F2 (dá mais peso ao recall) |
| Avaliação | Conjunto de teste, separado no início e nunca usado no treino |
| Interpretabilidade | Importância por permutação (global) e análise "e se este fator fosse típico?" por aluno (local) |

### Por que dois limiares?

Para a coordenação, **não detectar um aluno que vai evadir custa mais caro** do que conversar com
um aluno que não ia evadir. Mas alarmes demais fazem a equipe parar de confiar no sistema. Por isso
o sistema usa três faixas:

- 🔴 **Alto** (≥ limiar F1): alerta confiável, ação imediata
- 🟡 **Médio** (≥ limiar F2): acompanhamento; junto com o Alto, pega mais de 90% dos evasores
- 🟢 **Baixo**: sem sinais relevantes

## Resultados

Validação cruzada (5-fold, treino, limiar 0,5):

| Modelo | ROC-AUC | F1 | Recall | Precisão |
|---|---|---|---|---|
| Regressão Logística | 0,920 | 0,787 | 0,811 | 0,765 |
| Random Forest | 0,919 | 0,793 | 0,795 | 0,791 |
| **XGBoost** | **0,924** | **0,797** | 0,802 | **0,792** |

Conjunto de teste (885 alunos), **XGBoost**, ROC-AUC = **0,941**:

| Limiar | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Alto (0,64, máx. F1) | 88,9% | 86,9% | 77,1% | 0,817 |
| Médio (0,21, máx. F2) | 79,0% | 61,3% | 93,3% | 0,740 |

Os três modelos ficaram próximos. A Regressão Logística, mais simples e interpretável, perde
pouco para o XGBoost. Esse é um bom ponto para discutir na apresentação.

> Os números acima saem do `python treinar.py` e ficam em `modelos/metricas.json`. A aba
> "Desempenho do modelo" do app mostra também a matriz de confusão, a curva ROC e a importância
> dos atributos.

## O app

1. **📋 Painel da turma:** envie um CSV (ou use a turma de demonstração, que é o conjunto de
   teste). Mostra contagem por faixa, risco por curso, lista priorizada para download e a
   explicação de por que cada aluno está em risco.
2. **🔍 Avaliar um aluno:** formulário com os principais dados, que devolve a probabilidade, a faixa,
   uma recomendação de ação e os fatores que mais pesaram.
3. **📊 Desempenho do modelo:** comparação de modelos, métricas de teste, matrizes de confusão,
   curva ROC e importância dos atributos.
4. **ℹ️ Sobre:** problema, dados, pipeline e limitações.

## Estrutura

```
evasao-escolar/
├── app.py                  # interface Streamlit
├── colab.ipynb             # notebook passo a passo (Google Colab)
├── treinar.py              # treino, comparação, avaliação e exportação
├── src/dados.py            # leitura, tradução de colunas, engenharia de atributos, faixas
├── data/
│   ├── dataset_uci_697.csv # base original (UCI)
│   └── amostra_turma.csv   # conjunto de teste, usado como turma de demonstração
├── modelos/
│   ├── modelo.joblib       # pipeline treinado + limiares + valores de referência
│   └── metricas.json       # todas as métricas usadas no app
├── pyproject.toml          # dependências (uv)
├── uv.lock                 # versões travadas (uv)
├── .python-version         # Python 3.12
└── requirements.txt        # alternativa para pip/Colab
```

## Limitações

- Dados de **uma única instituição portuguesa**. Antes de usar em outro contexto, seria preciso
  retreinar com dados locais.
- O modelo usa informações **até o fim do 1º ano** (notas do 2º semestre), então o alerta vale a
  partir desse momento. Uma versão para o momento da matrícula precisaria ser treinada sem essas
  colunas e teria desempenho menor.
- O modelo encontra **correlações, não causas**. A explicação "e se" mostra o que o modelo
  aprendeu, não o motivo real da evasão.
- Atributos sensíveis (gênero, estado civil, nacionalidade) entram no modelo. Num uso real, seria
  preciso auditar se há tratamento desigual entre grupos. O sistema deve **apoiar** a decisão da
  coordenação, não substituí-la.
