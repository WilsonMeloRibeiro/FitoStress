# 🌱 FitoStress

### Ferramenta para Cálculo, Visualização e Relato de Índices de Estresse Abiótico em Plantas

O **FitoStress** é uma aplicação desktop desenvolvida em Python para automatizar o cálculo, a visualização e a geração de relatórios de índices de estresse abiótico em plantas.

A ferramenta foi projetada para reunir diferentes etapas da análise em um único ambiente, permitindo o processamento de dados experimentais, o cálculo de índices e a interpretação dos resultados por meio de recursos gráficos e relatórios.

Um dos principais diferenciais do FitoStress é seu funcionamento **totalmente offline**, dispensando conexão com a internet e comunicação com servidores externos. Dessa forma, os dados são processados localmente, proporcionando maior autonomia ao pesquisador e possibilitando o uso da ferramenta em ambientes com conectividade limitada.

---

## 🚀 Funcionalidades

- **Cálculo automatizado:** processamento de índices de estresse abiótico a partir de dados de produtividade em condições com e sem estresse.
- **Visualização gráfica:** representação dos resultados por meio de gráficos e outras ferramentas de visualização.
- **Heatmap:** visualização comparativa dos índices calculados para diferentes genótipos.
- **PCA-biplot:** recurso de análise multivariada para explorar relações entre genótipos e índices.
- **Geração de relatórios:** exportação dos resultados em diferentes formatos.
- **Validação de dados:** mecanismos para verificar as informações fornecidas pelo usuário.
- **Funcionamento offline:** processamento local, sem necessidade de conexão com a internet.

## 📊 Índices de estresse

O FitoStress implementa índices utilizados na avaliação e comparação do desempenho de genótipos vegetais sob diferentes condições ambientais.

Entre os índices contemplados estão:

| Índice | Descrição |
|---|---|
| RC | Índice de redução |
| TOL | Tolerância |
| MP | Média de produtividade |
| GMP | Média geométrica de produtividade |
| HM | Média harmônica |
| SSI | Índice de suscetibilidade ao estresse |
| STI | Índice de tolerância ao estresse |
| YI | Índice de rendimento |
| YSI | Índice de estabilidade de rendimento |
| RSI | Índice relativo de suscetibilidade |

Os cálculos utilizam dados de produtividade em condições favoráveis (Yp) e sob estresse (Ys).

## 🖥️ Tecnologias utilizadas

O projeto foi desenvolvido utilizando Python e bibliotecas voltadas ao processamento de dados, cálculos numéricos, visualização e desenvolvimento de interfaces gráficas.

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem de programação |
| Pandas | Manipulação e organização de dados |
| NumPy | Operações e cálculos numéricos |
| Matplotlib | Criação de gráficos |
| Tkinter | Desenvolvimento da interface gráfica |
| SciPy | Recursos de computação científica |
| Scikit-learn | Análise multivariada e PCA |
| Pytest | Testes e validação das funções |
| PyInstaller | Empacotamento da aplicação |

## 📥 Download

A versão 1.0.0 do FitoStress está disponível para download na página de lançamentos do GitHub.

[⬇️ Baixar FitoStress — Versão 1.0.0](https://github.com/WilsonMeloRibeiro/FitoStress/releases/tag/v1.0.0)

Acesse a página de releases para baixar o executável e consultar os arquivos disponibilizados para essa versão.

## 🎯 Objetivo do projeto

O FitoStress tem como objetivo facilitar o processamento e a interpretação de dados relacionados à tolerância de plantas ao estresse abiótico, oferecendo uma alternativa desktop integrada para pesquisadores, estudantes e profissionais envolvidos em pesquisas de melhoramento vegetal.

Além da praticidade, sua arquitetura offline busca reduzir a dependência de serviços externos e ampliar as possibilidades de utilização em diferentes contextos de pesquisa.
