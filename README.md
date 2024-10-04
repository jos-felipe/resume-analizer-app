# Analisador de Currículos com RAG

Este projeto implementa um analisador de currículos usando Streamlit, processamento de linguagem natural e Retrieval-Augmented Generation (RAG). A aplicação permite aos usuários fazer upload de currículos em PDF, processá-los e fazer perguntas sobre as qualificações dos candidatos.

## Componentes Principais

1. **Streamlit**: Framework para criar a interface web da aplicação.
2. **PyPDF**: Biblioteca para extrair texto de arquivos PDF.
3. **LangChain**: Conjunto de ferramentas para processamento de linguagem natural e RAG.
4. **FAISS**: Biblioteca para busca eficiente de vetores similares.
5. **Hugging Face**: Plataforma para acessar modelos de linguagem pré-treinados.

## Fluxo de Funcionamento

1. **Upload do Currículo**: O usuário faz upload de um arquivo PDF contendo o currículo.

2. **Extração de Texto**: O texto é extraído do PDF usando a biblioteca PyPDF.

3. **Processamento do Texto**:
   - O texto é dividido em chunks menores usando `RecursiveCharacterTextSplitter`.
   - Cada chunk é convertido em um embedding usando `HuggingFaceEmbeddings`.
   - Os embeddings são armazenados em um índice FAISS para busca eficiente.

4. **Configuração do Modelo de Linguagem**: É utilizado o modelo "google/flan-t5-large" da Hugging Face para gerar respostas.

5. **Interface de Perguntas e Respostas**:
   - O usuário digita uma pergunta sobre o currículo.
   - A pergunta é processada pelo sistema RAG:
     a. Busca os chunks mais relevantes no índice FAISS.
     b. Usa o modelo de linguagem para gerar uma resposta com base nos chunks recuperados e na pergunta.
   - A resposta é exibida ao usuário, junto com os trechos do currículo usados como fonte.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Streamlit**: Para criar a interface web interativa.
- **LangChain**: Para implementar o pipeline de RAG.
- **FAISS**: Para armazenamento e busca eficiente de embeddings.
- **Hugging Face**: Para acessar modelos de linguagem pré-treinados.
- **PyPDF**: Para extrair texto de arquivos PDF.

## Vantagens da Abordagem RAG

1. **Respostas Contextualizadas**: As respostas são geradas com base no conteúdo específico do currículo.
2. **Flexibilidade**: Pode responder a uma ampla variedade de perguntas sem necessidade de treinamento específico.
3. **Transparência**: Mostra as fontes das informações usadas para gerar as respostas.
4. **Eficiência**: Usa busca vetorial para rapidamente encontrar informações relevantes em currículos longos.

## Possíveis Melhorias

1. **Suporte a Múltiplos Currículos**: Permitir o upload e análise de vários currículos simultaneamente.
2. **Análise Comparativa**: Implementar funcionalidades para comparar diferentes candidatos.
3. **Extração de Informações Estruturadas**: Adicionar capacidade de extrair e organizar informações específicas (ex: habilidades, experiência) de forma estruturada.
4. **Interface de Usuário Aprimorada**: Adicionar visualizações e filtros mais avançados para melhorar a experiência do usuário.
5. **Modelos de Linguagem Mais Avançados**: Experimentar com modelos mais recentes e potentes para melhorar a qualidade das respostas.

Este projeto demonstra uma aplicação prática de técnicas avançadas de NLP e IA para resolver um problema real de análise de currículos, oferecendo uma ferramenta poderosa para profissionais de RH e recrutadores.
