# Relatório: Implementação do Problema do Robô Reciclador com Q-Learning

**Disciplina:** Aprendizado por Reforço  
**Exemplo:** 3.3 - Reinforcement Learning: An Introduction (Sutton & Barto, 2018)  
**Data:** Setembro 2025

## 1. Introdução

Este relatório apresenta a implementação e análise do problema clássico do **Robô Reciclador** utilizando o algoritmo **Q-Learning** (Temporal Difference Learning). O objetivo é desenvolver um agente que aprenda a política ótima para maximizar recompensas ao coletar latas de alumínio, considerando o gerenciamento eficiente da bateria.

## 2. Descrição do Problema

### 2.1 Ambiente
O robô opera em um ambiente com dois estados de bateria:
- **HIGH**: Bateria com carga alta
- **LOW**: Bateria com carga baixa

### 2.2 Ações Disponíveis
- **Estado HIGH**: `search` (procurar latas), `wait` (esperar)
- **Estado LOW**: `search`, `wait`, `reload` (recarregar bateria)

### 2.3 Sistema de Recompensas
- **r_search**: +10 (encontrar latas)
- **r_wait**: +2 (esperar/descansar)
- **r_recharge**: 0 (recarregar)
- **r_battery_dead**: -100 (bateria esgotada durante busca)

### 2.4 Dinâmica de Transição
- **α = 0.6**: Probabilidade de manter bateria HIGH após search
- **β = 0.8**: Probabilidade de manter bateria LOW após search (sem esgotar)

## 3. Implementação do Algoritmo

### 3.1 Classe do Ambiente
```python
class RecyclingRobotEnvironment:
    def __init__(self, alpha=0.6, beta=0.8, r_search=10, r_wait=2)
```
Implementa a lógica de transição de estados e sistema de recompensas conforme especificação do Exemplo 3.3.

### 3.2 Classe do Agente TD
```python
class TDAgent:
    def __init__(self, env, learning_rate=0.15, discount_factor=0.95, 
                 epsilon=0.3, epsilon_decay=0.998, lr_decay=0.9995)
```

### 3.3 Algoritmo Q-Learning
A atualização da Q-table segue a equação:

**Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]**

Onde:
- **α**: Taxa de aprendizado (learning rate)
- **γ**: Fator de desconto (discount factor)
- **r**: Recompensa imediata
- **s'**: Próximo estado
- **max Q(s',a')**: Máximo Q-value do próximo estado

## 4. Decisões de Implementação

### 4.1 Parâmetros Iniciais
Após análise do problema original que apresentava estagnação no aprendizado, foram implementadas as seguintes melhorias:

| Parâmetro | Valor Original | Valor Otimizado | Justificativa |
|-----------|----------------|-----------------|---------------|
| Learning Rate | 0.1 | 0.15 | Aprendizado inicial mais rápido |
| Discount Factor | 0.9 | 0.95 | Maior importância às recompensas futuras |
| Epsilon | 0.1 | 0.3 | Maior exploração inicial |
| Epsilon Decay | - | 0.998 | Redução gradual da exploração |
| LR Decay | - | 0.9995 | Convergência mais fina |

### 4.2 Estratégia de Decaimento Adaptivo
- **Epsilon Decay**: Reduz exploração gradualmente de 30% para 1%
- **Learning Rate Decay**: Diminui taxa de aprendizado para convergência estável
- **Limites Mínimos**: Evitam convergência prematura

### 4.3 Configuração de Treinamento
- **Épocas**: 500
- **Passos por época**: 1000
- **Total de interações**: 500.000 steps

## 5. Resultados Obtidos

### 5.1 Q-Table Final (Política Ótima)

| Estado | Ação | Q-Value | Status |
|--------|------|---------|--------|
| HIGH | search | 144.60 | **ÓTIMA**|
| HIGH | wait | 139.83 | - |
| LOW | search | 122.62 | - |
| LOW | wait | 132.21 | - |
| LOW | reload | 137.69 | **ÓTIMA**|

### 5.2 Política Aprendida
- **Estado HIGH** → **SEARCH** (144.60 > 139.83)
- **Estado LOW** → **RELOAD** (137.69 > 132.21 > 122.62)

### 5.3 Análise de Performance
- **Recompensa total final**: 3.500.000
- **Média por época**: 7.000
- **Média por step**: 7.00
- **Melhoria vs implementação original**: ~40%

## 6. Análise dos Gráficos

### 6.1 Optimal Policy Heat Map
O mapa de calor confirma que o agente identificou corretamente a política ótima:
- Valores mais altos (azul escuro) indicam ações preferenciais
- **HIGH-search** e **LOW-reload** apresentam os maiores Q-values
- A diferenciação clara entre ações ótimas e subótimas comprova convergência

### 6.2 Total Cumulative Reward by Epoch
A curva de recompensa acumulativa apresenta características ideais:
- **Crescimento linear**: Indica aprendizado estável sem oscilações
- **Ausência de plateaus**: Confirma que não houve estagnação
- **Inclinação consistente**: Demonstra política convergida desde estágios iniciais

## 7. Validação Teórica

### 7.1 Consistência com o Exemplo 3.3
A política aprendida está **totalmente alinhada** com a solução teórica:
- Com bateria alta, é racional procurar latas (maior recompensa esperada)
- Com bateria baixa, é prudente recarregar (evita penalização de -100)

### 7.2 Convergência do Algoritmo
O algoritmo demonstrou convergência através de:
- **Estabilização dos Q-values**: Sem variações significativas nas últimas épocas
- **Política determinística**: Ações claramento definidas para cada estado
- **Performance consistente**: Recompensa média estável

## 8. Conclusões

### 8.1 Objetivos Alcançados
 **Implementação bem-sucedida** do algoritmo Q-Learning  
 **Identificação da política ótima** para o problema do robô reciclador  
 **Convergência estável** sem oscilações ou estagnação  
 **Performance superior** comparada à implementação inicial  

### 8.2 Lições Aprendidas
1. **Importância do decaimento adaptivo**: Epsilon e learning rate decay foram cruciais para convergência
2. **Balanceamento exploração vs exploitação**: Epsilon inicial alto seguido de decaimento gradual
3. **Sintonia de hiperparâmetros**: Pequenos ajustes geraram melhorias significativas
4. **Validação empírica**: Gráficos lineares confirmam aprendizado correto

### 8.3 Trabalhos Futuros
- Implementação de diferentes algoritmos (SARSA, Double Q-Learning)
- Análise de sensibilidade dos hiperparâmetros
- Extensão para ambientes mais complexos (múltiplos estados de bateria)
- Comparação de performance com métodos de planejamento dinâmico

## 9. Referências

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Exemplo 3.3: "Recycling Robot", páginas 52-53.

---

**Código fonte completo disponível nos arquivos:**
- `robot.py`: Classes do ambiente e agente
- `plot.py`: Script de treinamento e visualização