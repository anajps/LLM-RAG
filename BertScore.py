from bert_score import score
import pandas as pd

# Perguntas
questions = [
    "Qual é o nome do autor da Aula 11 e o autor da Aula 03?",
    "Quais são os tópicos abordados na Aula 03?",
    "O que é o núcleo kernel?",
    "De acordo com a aula 03 e a aula 11 o que é sistema de entrada e saída?",
    "O que é modo usuário e modo supervisor de acordo com o slide 11?",
    "O que é a CPU e qual sua função?",
    "O que é o mode bit mencionado nos slides?",
    "Quais são as camadas da arquitetura de um sistema computacional de acordo com as aulas?",
    "Quais são as principais características do Linux?",
    "O que é uma chamada de sistema e como funciona?"
]

# =====================================================
# GROUND TRUTH (substitua pelas respostas oficiais)
# =====================================================
ground_truths = [
    "Jordana Sarmenghi Salamon é autora da Aula 11 e Prof. Jó Ueyama é autor da Aula 03 e a apresentação baseada nos slides da Profa. Dra. Kalinka Castelo Branco, do Prof. Dr. Antônio Carlos Sementille, da Profa. Dra. Luciana A. F. Martimiano",
    "A Aula 03 aborda estruturas de sistemas operacionais, CPU, memória, BIOS, arquitetura do sistema e chamadas de sistema.",
    "O kernel é o núcleo do sistema operacional responsável pelo gerenciamento dos recursos essenciais da máquina, como CPU, memória e dispositivos conectados , ele é uma camada que fica entre o hardware do computador e os softwares executados nele ",
    "O sistema de entrada e saída gerencia a comunicação entre dispositivos externos como pendrive, mouse e monitor e o sistema operacional",
    "Modo usuário possui acesso restrito a certos recurso do sistema e modo supervisor que tambem pode ser chamado de modo kernel possui acesso privilegiado aos recursos do sistema pode acessar qualquer recurso",
    "A CPU é a unidade central de processamento responsável por executar instruções.",
    "O mode bit indica se o processador está operando em modo usuário ou modo supervisor.",
    "A arquitetura é composta por hardware, microarquitetura, linguagem de máquina, sistema operacional, programas do sistema e aplicações.",
    "Linux é um sistema operacional livre, multitarefa, multiusuário, seguro e modular.",
    " é o mecanismo que permite que um programa que esteja em modo usuário peça serviços ao sistema operacional"
]

# =====================================================
# RESPOSTAS NOTEBOOKLM
# =====================================================
notebooklm_answers = [
    """Aula 11: Jordana Sarmenghi Salamon.
Aula 03: Prof. Jó Ueyama.""",

    """Estruturas de SOs, CPU, memória, dispositivos de E/S,
BIOS, arquitetura do sistema, modos de acesso e chamadas de sistema.""",

    """O kernel é a base principal de um sistema operacional,
responsável pela gerência do processador, interrupções,
comunicação e sincronização entre processos.""",

    """O sistema de E/S controla operações de entrada e saída,
gerenciando dispositivos através de controladoras e drivers.""",

    """Modo usuário executa processos com acesso restrito.
Modo supervisor executa o sistema operacional com acesso total ao hardware.""",

    """A CPU é o cérebro do computador e executa instruções,
utilizando registradores e ciclos de clock.""",

    """O mode bit indica o modo de operação atual do processador,
diferenciando modo usuário e modo supervisor.""",

    """A arquitetura possui hardware, microarquitetura,
linguagem de máquina, sistema operacional,
programas do sistema, aplicativos e usuários.""",

    """Linux é livre, multitarefa, multiusuário, seguro,
modular e suporta múltiplas CPUs.""",

    """Uma chamada de sistema é a interface entre aplicações
e o sistema operacional, utilizando TRAP para acessar o modo kernel."""
]

# =====================================================
# RESPOSTAS DO LLM_RAG
# =====================================================
llm_rag_answers = [
    "O autor da Aula 11 é Jordana Sarmenghi Salamon, e o autor da Aula 03 é o Prof. Jó Ueyama.",

    """Os tópicos abordados na Aula 03 são:
    Estruturas de Sistemas Operacionais,
    Componentes Básicos,
    BIOS,
    Arquitetura do Sistema
    e Processos.""",

    """O núcleo, ou kernel, é a base principal de um sistema operacional.
    Ele gerencia recursos do sistema e faz a ponte entre software e hardware.""",

    """Os dispositivos de entrada e saída permitem a comunicação entre
    o computador e o ambiente externo por meio de teclados,
    mouses, impressoras e monitores.""",

    """Modo usuário permite acesso apenas aos próprios dados e instruções.
    Modo supervisor possui acesso ao hardware e ao kernel.""",

    """A CPU é a Unidade Central de Processamento responsável
    por buscar, decodificar e executar instruções.""",

    """O mode bit é um conjunto de bits localizado no PSW que
    determina se o processador opera em modo usuário ou kernel.""",

    """A arquitetura é composta por Hardware,
    Micro Arquitetura,
    Linguagem de Máquina,
    Sistema Operacional,
    Programas do Sistema e Aplicativos.""",

    """Linux é livre, multitarefa, multiusuário,
    possui sistema avançado de permissões,
    modularização e suporte a múltiplas CPUs.""",

    """Uma chamada de sistema permite que uma aplicação solicite
    serviços ao sistema operacional, mudando do modo usuário para o modo kernel."""
]

# =====================================================
# BERTScore NOTEBOOKLM
# =====================================================
P_nb, R_nb, F1_nb = score(
    notebooklm_answers,
    ground_truths,
    lang="pt",
    verbose=True
)

# =====================================================
# BERTScore LLM_RAG
# =====================================================
P_rag, R_rag, F1_rag = score(
    llm_rag_answers,
    ground_truths,
    lang="pt",
    verbose=True
)

# =====================================================
# RESULTADOS
# =====================================================
df = pd.DataFrame({
    "Pergunta": questions,
    "BERTScore_NotebookLM": [round(x.item(), 4) for x in F1_nb],
    "BERTScore_LLM_RAG": [round(x.item(), 4) for x in F1_rag]
})

print(df)

media_nb = F1_nb.mean().item()
media_rag = F1_rag.mean().item()

print("\n==============================")
print(f"Média NotebookLM: {media_nb:.4f}")
print(f"Média LLM_RAG:    {media_rag:.4f}")
print("==============================")

if media_nb > media_rag:
    print("NotebookLM teve melhor desempenho.")
elif media_rag > media_nb:
    print("LLM_RAG teve melhor desempenho.")
else:
    print("Empate.")