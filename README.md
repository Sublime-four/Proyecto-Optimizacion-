# Proyecto-Optimizacion-
OptiBus Unillanos es una aplicación web para diseñar y analizar rutas de buses universitarios de forma inteligente.
El backend implementa distintas metaheurísticas para resolver una variante del problema del viajante (TSP) con capacidad, y el frontend permite visualizar las rutas y consultar una IA explicadora que detalla paso a paso qué hizo cada algoritmo.

🎯 Objetivo

Encontrar rutas de bus que:

Minimicen la distancia total recorrida (km).

Respeten la capacidad del bus (o penalicen el sobrecupo).

Sean interpretables para usuarios no expertos gracias a la IA explicadora.

🧠 Metaheurísticas implementadas

Greedy (vecino más cercano)
Construye rutas eligiendo siempre el siguiente paradero más cercano desde el actual.

Algoritmo Genético (GA)

Población de rutas aleatorias.

Evaluación con función de fitness basada en distancia + penalización por sobrecupo.

Selección por ruleta, crossover tipo OX simplificado y mutación por swaps.

Recocido Simulado (SA)

Parte de la solución Greedy.

Genera vecinos haciendo swaps entre paraderos internos.

Acepta soluciones peores con probabilidad dependiente de la temperatura.

📊 Métricas clave

Para cada ruta se calculan, entre otras:

total_distance_km – distancia total recorrida.

total_demand – estudiantes atendidos.

over_capacity – exceso respecto a la capacidad del bus.

penalty_km – penalización por sobrecupo.

objective_km – función objetivo = distancia + penalización.

fitness – calidad de la ruta (1 / (1 + objective_km)).

time_min – tiempo estimado según velocidad promedio.

🤖 IA explicadora

El proyecto incluye un endpoint /ask_ai que usa Gemini para:

Explicar cómo funciona cada método (Greedy, GA, SA).

Interpretar los resultados de una corrida (/solve o /solve_multi).

Generar explicaciones en español, paso a paso, usando el JSON de los steps.

En el frontend se muestra como un widget de chat (“IA explicadora”) que permite hacer preguntas sobre:

La ruta encontrada.

Penalizaciones.

Iteraciones de GA/SA.

Métricas como objective_km, fitness, etc.

🛠️ Stack tecnológico

Backend: Python, FastAPI, Pydantic

Optimización: Metaheurísticas (Greedy, GA, SA)

IA: Gemini (Google Generative Language API)

Frontend: HTML/CSS/JS, widget de chat propio

Visualización: Mapa con paraderos y rutas (biblioteca de mapas en el frontend)
