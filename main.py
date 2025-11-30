from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import math
import random
import time
import os
import json
import requests
import re

# ============================================================
# 0. CONFIG: FRONT (index.html + styles.css)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
CSS_FILE = BASE_DIR / "styles.css"

app = FastAPI(title="Rutas Unillanos - Metaheurísticas API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 1. DATASET DE PARADEROS
# ============================================================

STOPS = {
    "UNILLANOS": {"name": "Unillanos - Sede Barcelona", "lat": 4.07438, "lng": -73.58331},

    "VIVA": {"name": "Viva", "lat": 4.125961, "lng": -73.638722},
    "VILLACENTRO": {"name": "Villacentro", "lat": 4.1350, "lng": -73.6400},
    "POSTOBON": {"name": "Postobón", "lat": 4.1300, "lng": -73.6250},
    "GRAMA": {"name": "Grama", "lat": 4.157975, "lng": -73.638278},
    "HATO_GRANDE": {"name": "Hato Grande", "lat": 4.1500, "lng": -73.6300},
    "COVISAN": {"name": "Covisán", "lat": 4.1450, "lng": -73.6150},
    "LOS_MARACOS": {"name": "Los Maracos", "lat": 4.1475, "lng": -73.6420},

    "LA_MADRID": {"name": "La Madrid", "lat": 4.057538, "lng": -73.668749},
    "LA_ROCHELA": {"name": "La Rochela", "lat": 4.1100, "lng": -73.6200},

    "TERMINAL": {"name": "Terminal", "lat": 4.13226, "lng": -73.60359},
    "MANANTIAL": {"name": "Manantial", "lat": 4.1200, "lng": -73.6050},

    "AMARILO": {"name": "Amarilo", "lat": 4.1200, "lng": -73.6150},
    "SERRAMONTE": {"name": "Serramonte", "lat": 4.1250, "lng": -73.6180},

    "PARQUE_ESTUDIANTES": {"name": "Parque de los Estudiantes", "lat": 4.1350, "lng": -73.6280},
    "CENTRO": {"name": "Centro", "lat": 4.1400, "lng": -73.6300},
    "CAMPANARIO": {"name": "CAMPANARIO", "lat": 4.132950, "lng": -73.56535632730824},

    "KIRPA_20SUR": {
        "name": "Kirpa (Calle 20 Sur)",
        "lat": 4.12198,
        "lng": -73.58295
    },
    "VALLES_CAROLINA_20SUR": {
        "name": "Valles de la Carolina (Calle 20 Sur)",
        "lat": 4.11867,
        "lng": -73.58839
    },
    "GAVIOTAS_20SUR": {
        "name": "Las Gaviotas (Calle 20 Sur)",
        "lat": 4.11572,
        "lng": -73.60384
    },
    "ACAPULCO_20SUR": {
        "name": "Acapulco (Calle 20 Sur)",
        "lat": 4.12118,
        "lng": -73.59349
    },

    "MARCO_ANTONIO_PINILLA": {
        "name": "Marco Antonio Pinilla",
        "lat": 4.12999,
        "lng": -73.58778
    },
    "SANTA_CATALINA": {
        "name": "Santa Catalina",
        "lat": 4.13688,
        "lng": -73.59076
    },
    "PORFIA": {
        "name": "Porfía",
        "lat": 4.088173,
        "lng": -73.670235
    },
}

STOPS_IDS = list(STOPS.keys())
IDX = {sid: i for i, sid in enumerate(STOPS_IDS)}

# ============================================================
# 2. RUTAS OFICIALES (cada una será un bus)
# ============================================================

ROUTE_DEFS = {
    "UNILLANOS_VIVA": {
        "label": "Unillanos → Viva",
        "stops": ["UNILLANOS", "VIVA"]
    },
    "UNILLANOS_VILLACENTRO": {
        "label": "Unillanos → Villacentro",
        "stops": ["UNILLANOS", "VILLACENTRO"]
    },
    "UNILLANOS_CENTRO": {
        "label": "Unillanos → Centro",
        "stops": ["UNILLANOS", "PARQUE_ESTUDIANTES", "CENTRO"]
    },
    "UNILLANOS_POSTOBON_GRAMA_AM": {
        "label": "Unillanos → Postobón → Grama",
        "stops": ["UNILLANOS", "POSTOBON", "GRAMA"]
    },
    "UNILLANOS_HATO_GRANDE_AM": {
        "label": "Unillanos → Hato Grande",
        "stops": ["UNILLANOS", "HATO_GRANDE"]
    },

    # COVISÁN AM por C20 Sur + Campanario, terminando en Covisán
    "UNILLANOS_COVISAN_AM": {
        "label": "Unillanos → C20 Sur → Campanario → Covisán",
        "stops": [
            "UNILLANOS",
            "ACAPULCO_20SUR",
            "VALLES_CAROLINA_20SUR",
            "KIRPA_20SUR",
            "CAMPANARIO",
            "COVISAN"
        ]
    },

    # LOS MARACOS AM
    "UNILLANOS_LOS_MARACOS_AM": {
        "label": "Unillanos → C20 Sur → Marco A. Pinilla → Terminal → Los Maracos",
        "stops": [
            "UNILLANOS",
            "ACAPULCO_20SUR",
            "VALLES_CAROLINA_20SUR",
            "KIRPA_20SUR",
            "MARCO_ANTONIO_PINILLA",
            "TERMINAL",
            "LOS_MARACOS"
        ]
    },

    "LA_MADRID_UNILLANOS": {
        "label": "La Madrid → Unillanos",
        "stops": ["LA_MADRID", "UNILLANOS"]
    },
    "LA_ROCHELA_UNILLANOS": {
        "label": "La Rochela → Unillanos",
        "stops": ["LA_ROCHELA", "UNILLANOS"]
    },
    "VIVA_UNILLANOS": {
        "label": "Viva → Unillanos",
        "stops": ["VIVA", "UNILLANOS"]
    },
    "VILLACENTRO_UNILLANOS": {
        "label": "Villacentro → Unillanos",
        "stops": ["VILLACENTRO", "UNILLANOS"]
    },
    "GRAMA_UNILLANOS": {
        "label": "Grama → Unillanos",
        "stops": ["GRAMA", "UNILLANOS"]
    },
    "HATO_GRANDE_UNILLANOS": {
        "label": "Hato Grande → Unillanos",
        "stops": ["HATO_GRANDE", "UNILLANOS"]
    },

    # COVISÁN → Unillanos
    "COVISAN_UNILLANOS": {
        "label": "Covisán → Campanario → Kirpa (C20 Sur) → Unillanos",
        "stops": [
            "COVISAN",
            "CAMPANARIO",
            "KIRPA_20SUR",
            "VALLES_CAROLINA_20SUR",
            "ACAPULCO_20SUR",
            "UNILLANOS"
        ]
    },

    # Los Maracos → Unillanos
    "LOS_MARACOS_UNILLANOS": {
        "label": "Los Maracos → Terminal → Marco A. Pinilla → C20 Sur → Unillanos",
        "stops": [
            "LOS_MARACOS",
            "TERMINAL",
            "MARCO_ANTONIO_PINILLA",
            "KIRPA_20SUR",
            "VALLES_CAROLINA_20SUR",
            "ACAPULCO_20SUR",
            "UNILLANOS"
        ]
    },

    # Porfía: solo ida
    "UNILLANOS_PORFIA": {
        "label": "Unillanos → Porfía",
        "stops": ["UNILLANOS", "PORFIA"]
    },

    # Porfía: solo regreso
    "PORFIA_UNILLANOS": {
        "label": "Porfía → Unillanos",
        "stops": ["PORFIA", "UNILLANOS"]
    },
}

# ============================================================
# 3. MATRIZ DE DISTANCIAS (HAVERSINE)
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def build_distance_matrix():
    n = len(STOPS_IDS)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, sid_i in enumerate(STOPS_IDS):
        for j, sid_j in enumerate(STOPS_IDS):
            if i == j:
                matrix[i][j] = 0.0
            else:
                a = STOPS[sid_i]
                b = STOPS[sid_j]
                matrix[i][j] = haversine(a["lat"], a["lng"], b["lat"], b["lng"])
    return matrix


DIST_MATRIX = build_distance_matrix()

# ============================================================
# 4. EVALUACIÓN DE UNA RUTA
# ============================================================

PENALTY_PER_STUDENT_KM = 2.0  # km virtuales por estudiante extra
AVG_SPEED_KMH = 35.0          # velocidad promedio


def evaluate_route(route: List[str], demands: Dict[str, int], capacity: int) -> Dict:
    total_distance = 0.0
    for i in range(len(route) - 1):
        a = IDX[route[i]]
        b = IDX[route[i + 1]]
        total_distance += DIST_MATRIX[a][b]

    total_demand = sum(
        demands.get(stop_id, 0)
        for stop_id in route
        if stop_id != "UNILLANOS"
    )

    over_capacity = max(0, total_demand - capacity)
    penalty_km = over_capacity * PENALTY_PER_STUDENT_KM
    objective_km = total_distance + penalty_km
    fitness = 1.0 / (1.0 + objective_km)
    time_min = (total_distance / AVG_SPEED_KMH) * 60.0 if total_distance > 0 else 0.0

    return {
        "route": route,
        "total_distance_km": total_distance,
        "total_demand": total_demand,
        "over_capacity": over_capacity,
        "penalty_km": penalty_km,
        "objective_km": objective_km,
        "fitness": fitness,
        "time_min": time_min,
    }

# ============================================================
# 5. PARADEROS A VISITAR (para bus global)
# ============================================================

def get_stops_from_active_routes(active_routes: List[str],
                                 demands: Dict[str, int]) -> List[str]:
    """
    Une los paraderos de las rutas activas y además incluye
    cualquier paradero que tenga demanda > 0.
    """
    stops_set = set()

    for rid in active_routes:
        if rid not in ROUTE_DEFS:
            continue
        for s in ROUTE_DEFS[rid]["stops"]:
            stops_set.add(s)

    for sid, d in demands.items():
        if d > 0 and sid in STOPS:
            stops_set.add(sid)

    stops_set.add("UNILLANOS")

    ordered = [s for s in STOPS_IDS if s in stops_set]
    return ordered

# ============================================================
# 6. METAHEURÍSTICAS (con paso a paso + FÓRMULAS)
# ============================================================

def make_random_route(stops_to_visit: List[str]) -> List[str]:
    start = "UNILLANOS"
    inner = [s for s in stops_to_visit if s != start]
    random.shuffle(inner)
    return [start] + inner + [start]


def neighbor_swap(route: List[str]) -> List[str]:
    route = route[:]
    if len(route) <= 3:
        return route
    i = random.randint(1, len(route) - 2)
    j = random.randint(1, len(route) - 2)
    route[i], route[j] = route[j], route[i]
    return route


# ---------- Helper: construir fórmulas detalladas ----------
def build_formulas_for_route(route: List[str],
                             demands: Dict[str, int],
                             capacity: int) -> (Dict, Dict[str, str]):
    """
    Devuelve:
      - info numérica (igual a evaluate_route)
      - un dict con fórmulas en texto para mostrar en el frontend
    """
    info = evaluate_route(route, demands, capacity)

    # --- Distancia ---
    dist_terms = []
    total_distance = 0.0
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        d = DIST_MATRIX[IDX[a]][IDX[b]]
        total_distance += d
        dist_terms.append(f"d({a},{b}) = {d:.3f}")

    if dist_terms:
        dist_expr = " + ".join(dist_terms) + f" = {total_distance:.3f} km"
    else:
        dist_expr = "0 km"

    # --- Demanda ---
    demand_terms = []
    total_demand = 0
    for stop_id in route:
        if stop_id == "UNILLANOS":
            continue
        d = demands.get(stop_id, 0)
        total_demand += d
        demand_terms.append(f"dem({stop_id}) = {d}")

    if demand_terms:
        demand_expr = " + ".join(demand_terms) + f" = {total_demand}"
    else:
        demand_expr = "0"

    # --- Penalización por sobrecupo ---
    over_capacity = max(0, total_demand - capacity)
    over_expr = f"max(0, {total_demand} - {capacity}) = {over_capacity}"

    penalty_km = over_capacity * PENALTY_PER_STUDENT_KM
    penalty_expr = f"{over_capacity} * {PENALTY_PER_STUDENT_KM} = {penalty_km:.3f} km"

    # --- Objetivo y fitness ---
    objective_km = info["objective_km"]
    objective_expr = f"{total_distance:.3f} + {penalty_km:.3f} = {objective_km:.3f} km"

    fitness = info["fitness"]
    fitness_expr = f"1 / (1 + {objective_km:.3f}) = {fitness:.6f}"

    formulas = {
        "distance": dist_expr,
        "demand": demand_expr,
        "over_capacity": over_expr,
        "penalty": penalty_expr,
        "objective": objective_expr,
        "fitness": fitness_expr,
    }
    return info, formulas


# ---------- Greedy con fórmulas paso a paso ----------

def greedy_tsp(stops_to_visit: List[str],
               demands: Dict[str, int],
               capacity: int) -> Dict:
    start = "UNILLANOS"
    remaining = [s for s in stops_to_visit if s != start]
    route = [start]
    current = start

    steps = []
    iteration = 0

    while remaining:
        best_next = None
        best_dist = float("inf")
        candidate_distances = []

        for cand in remaining:
            d = DIST_MATRIX[IDX[current]][IDX[cand]]
            candidate_distances.append({
                "from": current,
                "to": cand,
                "distance_km": d
            })
            if d < best_dist:
                best_dist = d
                best_next = cand

        route.append(best_next)
        remaining.remove(best_next)
        current = best_next

        iteration += 1
        partial_route = route + ([start] if remaining else [])

        # info + fórmulas de esta ruta parcial
        info_partial, formulas = build_formulas_for_route(
            partial_route, demands, capacity
        )

        steps.append({
            "method": "greedy",
            "iteration": iteration,
            "phase": "build_route",
            "from_stop": route[-2],
            "chosen_next": best_next,
            "candidate_distances": candidate_distances,
            "route": info_partial["route"],
            "distance_km": info_partial["total_distance_km"],
            "objective_km": info_partial["objective_km"],
            "fitness": info_partial["fitness"],
            "temperature": None,
            "best_so_far": True,
            "details": (
                f"Desde {route[-2]} se evaluaron {len(candidate_distances)} candidatos "
                f"y se eligió {best_next} con distancia {best_dist:.3f} km."
            ),
            # NUEVO: fórmulas explícitas
            "formulas": formulas,
        })

    if route[-1] != start:
        route.append(start)

    final_info, _ = build_formulas_for_route(route, demands, capacity)
    final_info["steps"] = steps
    return final_info


# ---------- Algoritmo genético con paso a paso ----------

def crossover(parent1: List[str], parent2: List[str]) -> (List[str], Dict):
    """
    Crossover tipo OX simplificado.
    Devuelve el hijo y un log con los puntos de corte y el segmento heredado.
    """
    if len(parent1) <= 3:
        return parent1[:], {
            "type": "noop",
            "operator": "none",
            "reason": "route_too_short",
            "cut_points": None,
            "segment": [],
        }

    start_idx = 1
    end_idx = len(parent1) - 1  # último índice interno

    i = random.randint(start_idx, end_idx - 2)
    j = random.randint(i + 1, end_idx - 1)

    # Segmento que se copia tal cual desde parent1
    middle = parent1[i:j]

    # Resto de genes se completa respetando el orden de parent2
    rest = [x for x in parent2 if x not in middle or x == "UNILLANOS"]
    child_inner = [s for s in rest if s != "UNILLANOS"]
    child_inner = child_inner[:1] + middle + child_inner[1:]

    child = ["UNILLANOS"] + [s for s in child_inner if s != "UNILLANOS"] + ["UNILLANOS"]

    log = {
        "type": "crossover",
        "operator": "OX_simplificado",
        "cut_points": (i, j),
        "segment": middle,
        "parent1": parent1,
        "parent2": parent2,
        "child_after_crossover": child,
    }
    return child, log


def mutate(route: List[str], mutation_rate: float = 0.15) -> (List[str], Dict):
    """
    Mutación por intercambio de posiciones internas.
    Devuelve la ruta mutada y un log con los swaps realizados.
    """
    route = route[:]
    inner_indices = list(range(1, len(route) - 1))
    swaps = []

    for i in inner_indices:
        if random.random() < mutation_rate:
            j = random.choice(inner_indices)
            if i == j:
                continue
            route[i], route[j] = route[j], route[i]
            swaps.append((i, j))

    log = {
        "type": "mutation",
        "operator": "swap_indices",
        "mutation_rate": mutation_rate,
        "swaps": swaps,  # lista de pares (i, j) intercambiados
    }
    return route, log


def ga_tsp(
    stops_to_visit: List[str],
    demands: Dict[str, int],
    capacity: int,
    population_size: int = 40,
    generations: int = 80,
    mutation_rate: float = 0.15,
) -> Dict:
    """
    GA para TSP con log de:
      - mejor, promedio y peor fitness por generación
      - ejemplo de crossover (padres, cortes, segmento)
      - ejemplo de mutación (índices intercambiados)
      - fórmulas detalladas del mejor individuo, incluyendo:
          distance, demand, over_capacity, penalty, objective, fitness,
          ga_selection, ga_crossover, ga_mutation.
    """

    def fitness_of(rt: List[str]) -> float:
        return evaluate_route(rt, demands, capacity)["fitness"]

    # Población inicial
    population = [make_random_route(stops_to_visit) for _ in range(population_size)]
    steps: List[Dict] = []

    for gen in range(1, generations + 1):
        # ---------- EVALUACIÓN ----------
        scored = [(fitness_of(r), r) for r in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        best_fit, best_route = scored[0]
        best_info, formulas_best = build_formulas_for_route(
            best_route, demands, capacity
        )

        avg_fit = sum(f for f, _ in scored) / len(scored)
        worst_fit, worst_route = scored[-1]

        # ---------- SELECCIÓN ----------
        elites = [r for _, r in scored[:5]]  # top 5 como élite
        total_fit = sum(f for f, _ in scored) or 1e-6

        def select() -> List[str]:
            """Ruleta proporcional al fitness: P(rt) = f(rt) / total_fit."""
            r = random.random() * total_fit
            acc = 0.0
            for f, rt in scored:
                acc += f
                if acc >= r:
                    return rt
            return scored[-1][1]

        new_pop: List[List[str]] = elites[:]

        # Para el paso a paso: solo loggeamos UN ejemplo de hijo por generación
        ga_events = None

        # ---------- REPRODUCCIÓN ----------
        while len(new_pop) < population_size:
            parent1 = select()
            parent2 = select()

            # CROSSOVER
            child, cross_log = crossover(parent1, parent2)

            # MUTACIÓN
            child_before_mut = child[:]
            child, mut_log = mutate(child, mutation_rate)

            new_pop.append(child)

            if ga_events is None:
                child_info, child_formulas = build_formulas_for_route(
                    child, demands, capacity
                )
                ga_events = {
                    "parent1": parent1,
                    "parent2": parent2,
                    "child_before_mutation": child_before_mut,
                    "child_after_mutation": child,
                    "crossover": cross_log,
                    "mutation": mut_log,
                    "child_formulas": child_formulas,
                }

        population = new_pop

        # ---------- TEXTO GA PARA EL OVERLAY ----------
        formulas_best = dict(formulas_best)  # copia defensiva

        # Selección (genérico)
        ga_selection_str = (
            "Selección por ruleta proporcional al fitness:\n"
            "P(ruta_i) = f(ruta_i) / Σ_j f(ruta_j).\n"
            f"En esta generación participaron {len(scored)} individuos "
            f"con fitness total Σ f = {total_fit:.4f}.\n"
            "Las rutas con mayor fitness tienen mayor probabilidad de ser "
            "elegidas como padres."
        )

        ga_crossover_str = ""
        ga_mutation_str = ""

        if ga_events is not None:
            # ---- Crossover ----
            cross = ga_events["crossover"]
            p1_str = " → ".join(ga_events["parent1"])
            p2_str = " → ".join(ga_events["parent2"])

            cut_points = cross.get("cut_points")
            segment_list = cross.get("segment", []) or []
            segment = " → ".join(segment_list) if segment_list else "(segmento vacío)"

            # Puede ser NOOP (len<=3): cut_points = None, sin child_after_crossover
            if (
                isinstance(cut_points, (tuple, list))
                and len(cut_points) == 2
                and cross.get("type") != "noop"
            ):
                cut_i, cut_j = cut_points
                child_cross = " → ".join(
                    cross.get("child_after_crossover", ga_events["child_before_mutation"])
                )
                ga_crossover_str = (
                    f"Padre 1: {p1_str}\n"
                    f"Padre 2: {p2_str}\n"
                    f"Crossover OX simplificado con cortes en posiciones [{cut_i}, {cut_j}).\n"
                    f"Segmento heredado del padre 1: {segment}.\n"
                    f"Hijo después de crossover (antes de mutación): {child_cross}."
                )
            else:
                reason = cross.get("reason", "sin detalle")
                ga_crossover_str = (
                    "En esta generación el crossover de ejemplo no se aplicó "
                    "como OX estándar (probablemente ruta demasiado corta).\n"
                    f"Motivo registrado: {reason}.\n"
                    f"Padre 1: {p1_str}\n"
                    f"Padre 2: {p2_str}"
                )

            # ---- Mutación ----
            mut = ga_events["mutation"]
            swaps = mut.get("swaps", [])
            before_mut_str = " → ".join(ga_events["child_before_mutation"])
            after_mut_str = " → ".join(ga_events["child_after_mutation"])

            if swaps:
                swaps_str = ", ".join(f"({i} ↔ {j})" for i, j in swaps)
                ga_mutation_str = (
                    f"Operador: {mut['operator']} (rate={mut['mutation_rate']:.2f}).\n"
                    f"Intercambios aplicados en índices internos: {swaps_str}.\n"
                    f"Ruta antes de mutación:  {before_mut_str}\n"
                    f"Ruta después de mutación: {after_mut_str}"
                )
            else:
                ga_mutation_str = (
                    f"Operador: {mut['operator']} (rate={mut['mutation_rate']:.2f}).\n"
                    "En esta generación, el hijo de ejemplo no sufrió ningún swap "
                    "(la mutación no modificó la ruta).\n"
                    f"Ruta resultante: {after_mut_str}"
                )

        # Inyectamos en formulas_best para que el front lo pinte
        formulas_best["ga_selection"] = ga_selection_str
        if ga_crossover_str:
            formulas_best["ga_crossover"] = ga_crossover_str
        if ga_mutation_str:
            formulas_best["ga_mutation"] = ga_mutation_str

        # ---------- LOG DE LA GENERACIÓN ----------
        steps.append({
            "method": "ga",
            "iteration": gen,
            "phase": "generation",
            "route": best_info["route"],               # mejor ruta de la generación
            "distance_km": best_info["total_distance_km"],
            "objective_km": best_info["objective_km"],
            "fitness": best_info["fitness"],
            "temperature": None,
            "best_so_far": True,
            # métricas de la población
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "worst_fitness": worst_fit,
            "population_size": len(population),
            # ejemplo detallado de GA
            "ga_events": ga_events,
            # texto descriptivo
            "details": (
                f"Gen {gen}: mejor f={best_fit:.4f}, "
                f"promedio={avg_fit:.4f}, peor={worst_fit:.4f}."
            ),
            # fórmulas del mejor individuo (incluyendo GA_*)
            "formulas": formulas_best,
        })

    # ---------- Mejor solución final ----------
    best_final = max(population, key=lambda r: fitness_of(r))
    final_info, _ = build_formulas_for_route(best_final, demands, capacity)
    final_info["steps"] = steps
    return final_info


# ---------- Recocido simulado con paso a paso DETALLADO ----------

def sa_tsp(
    stops_to_visit: List[str],
    demands: Dict[str, int],
    capacity: int,
    initial_temp: float = 1.0,
    final_temp: float = 0.01,
    alpha: float = 0.95,
    steps_per_temp: int = 20,
    max_logged_steps: int = 1000,
) -> Dict:
    """
    Recocido simulado con log detallado:
      - delta de costo
      - probabilidad de aceptación e^(-delta/T)
      - si se aceptó o no
      - qué posiciones se intercambiaron (swap)
      - fórmulas de distancia / penalización / fitness
      - fórmulas SA: sa_delta, sa_acceptance (para el overlay)
    """

    # Helper para ver qué swap se hizo entre dos rutas
    def detect_swap(before: List[str], after: List[str]) -> Dict:
        diff_idx = [i for i in range(len(before)) if before[i] != after[i]]
        if len(diff_idx) == 2:
            i, j = diff_idx
            return {
                "type": "swap",
                "indices": (i, j),
                "stops_before": (before[i], before[j]),
                "stops_after": (after[i], after[j]),
            }
        return {
            "type": "unknown",
            "indices": diff_idx,
        }

    # Punto de partida: solución greedy (solo usamos la ruta final)
    greedy_info = greedy_tsp(stops_to_visit, demands, capacity)
    start_route = greedy_info["route"]
    current_info, current_formulas = build_formulas_for_route(
        start_route, demands, capacity
    )
    current_route = current_info["route"]
    current_score = current_info["objective_km"]

    best_route = current_route[:]
    best_info = current_info
    best_score = current_score

    T = initial_temp
    steps: List[Dict] = []
    iteration = 0

    # ---------- Paso 0: solución inicial ----------
    initial_formulas = dict(current_formulas)
    initial_formulas["sa_delta"] = (
        "Paso inicial: se parte de la solución Greedy.\n"
        "No hay vecino que comparar, por lo tanto Δ = 0."
    )
    initial_formulas["sa_acceptance"] = (
        f"T0 = {T:.4f}.\n"
        "La solución inicial se acepta por definición (probabilidad = 1)."
    )

    steps.append({
        "method": "sa",
        "iteration": iteration,
        "phase": "initial_greedy",
        "route": best_info["route"],
        "distance_km": best_info["total_distance_km"],
        "objective_km": best_info["objective_km"],
        "fitness": best_info["fitness"],
        "temperature": T,
        "best_so_far": True,
        "delta": 0.0,
        "accept_prob": 1.0,
        "accepted": True,
        "sa_move": None,
        "details": (
            f"Solución inicial Greedy. Obj = {best_info['objective_km']:.3f} "
            "km-equivalentes."
        ),
        "formulas": initial_formulas,
    })

    # ---------- Bucle principal de recocido ----------
    while T > final_temp and len(steps) < max_logged_steps:
        for _ in range(steps_per_temp):
            if len(steps) >= max_logged_steps:
                break

            iteration += 1

            # Propuesta de vecino
            before_route = current_route[:]
            prev_score = current_score

            candidate_route = neighbor_swap(current_route)
            move_info = detect_swap(before_route, candidate_route)

            cand_info, cand_formulas = build_formulas_for_route(
                candidate_route, demands, capacity
            )
            cand_score = cand_info["objective_km"]

            delta = cand_score - prev_score
            accepted = False
            phase = "rejected_worse"
            prob = 0.0

            if delta < 0:
                # Mejora: se acepta siempre
                accepted = True
                prob = 1.0
                phase = "improvement"
                current_route = candidate_route
                current_info = cand_info
                current_score = cand_score
                current_formulas = cand_formulas
            else:
                # Peor solución: se acepta con probabilidad e^(-delta/T)
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accepted = True
                    phase = "accepted_worse"
                    current_route = candidate_route
                    current_info = cand_info
                    current_score = cand_score
                    current_formulas = cand_formulas

            # Actualizar mejor global
            best_so_far_flag = False
            if current_score < best_score:
                best_route = current_route[:]
                best_info = current_info
                best_score = current_score
                best_so_far_flag = True

            # Ruta y fórmulas que se muestran en este paso
            info_step, formulas_step = build_formulas_for_route(
                current_route if accepted else before_route,
                demands,
                capacity,
            )

            # ---------- Texto SA para el overlay (sa_delta / sa_acceptance) ----------
            formulas_step = dict(formulas_step)

            # sa_delta
            formulas_step["sa_delta"] = (
                f"Δ = obj_candidato - obj_actual = "
                f"{cand_score:.3f} - {prev_score:.3f} = {delta:.3f}."
            )

            # sa_acceptance
            if delta < 0:
                formulas_step["sa_acceptance"] = (
                    f"T = {T:.4f}.\n"
                    "Como Δ < 0, la solución candidata es mejor y se acepta "
                    "siempre (probabilidad = 1)."
                )
            else:
                decision = "ACEPTADA" if accepted else "RECHAZADA"
                formulas_step["sa_acceptance"] = (
                    f"T = {T:.4f}.\n"
                    f"Δ ≥ 0, se calcula probabilidad de aceptación:\n"
                    f"p = exp(-Δ/T) = exp(-{delta:.4f}/{T:.4f}) = {prob:.4f}.\n"
                    f"En este paso la solución fue {decision}."
                )

            # ---------- Log del paso ----------
            steps.append({
                "method": "sa",
                "iteration": iteration,
                "phase": phase,
                "route": info_step["route"],
                "distance_km": info_step["total_distance_km"],
                "objective_km": info_step["objective_km"],
                "fitness": info_step["fitness"],
                "temperature": T,
                "best_so_far": best_so_far_flag,
                # datos específicos de SA
                "delta": float(delta),
                "accept_prob": float(prob),
                "accepted": bool(accepted),
                "sa_move": {
                    "before_route": before_route,
                    "candidate_route": candidate_route,
                    "move": move_info,
                },
                "details": (
                    f"T={T:.4f}, delta={delta:.4f}, "
                    f"p_aceptar=exp(-{delta:.4f}/{T:.4f})={prob:.4f}, "
                    f"{'ACEPTADA' if accepted else 'RECHAZADA'}; "
                    f"mejor_global={'sí' if best_so_far_flag else 'no'}."
                ),
                "formulas": formulas_step,
            })

        T *= alpha

    best_info["steps"] = steps
    return best_info


# ============================================================
# 6b. CONOCIMIENTO EXPLÍCITO DE LOS MÉTODOS (para la IA)
# ============================================================

ALGORITHMS_KNOWLEDGE = {
    "evaluate_route": {
        "description": (
            "Evalúa una ruta calculando:\n"
            "- total_distance_km: suma de distancias físicas entre paraderos consecutivos.\n"
            "- total_demand: suma de demandas de todos los paraderos (excepto UNILLANOS).\n"
            "- over_capacity: max(0, total_demand - capacidad_bus).\n"
            "- penalty_km: penalización por sobrecupo = over_capacity * PENALTY_PER_STUDENT_KM.\n"
            "- objective_km: función objetivo = total_distance_km + penalty_km.\n"
            "- fitness: 1 / (1 + objective_km).\n"
            "- time_min: tiempo estimado en minutos usando AVG_SPEED_KMH."
        ),
        "formulas": {
            "distance": "total_distance_km = Σ d(stop_i, stop_{i+1})",
            "demand": "total_demand = Σ dem(stop) para todos los paraderos ≠ UNILLANOS",
            "over_capacity": "over_capacity = max(0, total_demand - capacity)",
            "penalty": "penalty_km = over_capacity * PENALTY_PER_STUDENT_KM",
            "objective": "objective_km = total_distance_km + penalty_km",
            "fitness": "fitness = 1 / (1 + objective_km)",
        },
    },
    "greedy": {
        "description": (
            "Construye una ruta usando el vecino más cercano:\n"
            "1) Empieza en UNILLANOS.\n"
            "2) Mientras haya paraderos sin visitar, elige el más cercano al actual.\n"
            "3) Cuando se acaban, vuelve a UNILLANOS.\n"
            "En cada iteración guarda un paso en 'steps' con:\n"
            "- candidate_distances: distancias desde el paradero actual a cada candidato.\n"
            "- chosen_next: paradero elegido.\n"
            "- formulas: distance, demand, over_capacity, penalty, objective, fitness."
        )
    },
    "ga": {
        "description": (
            "Algoritmo Genético para TSP:\n"
            "1) Población inicial: rutas aleatorias (siempre empiezan y terminan en UNILLANOS).\n"
            "2) Evaluación: se calcula fitness con evaluate_route.\n"
            "3) Selección: ruleta proporcional al fitness.\n"
            "4) Crossover: operador OX simplificado.\n"
            "5) Mutación: swaps aleatorios de paraderos internos.\n"
            "6) Se repite por varias generaciones.\n"
            "En 'steps' por generación se guarda:\n"
            "- best_fitness, avg_fitness, worst_fitness.\n"
            "- la mejor ruta de la generación.\n"
            "- ga_events: ejemplo de cruce y mutación.\n"
            "- formulas: distance, demand, over_capacity, penalty, objective, fitness,\n"
            "            ga_selection, ga_crossover, ga_mutation."
        )
    },
    "sa": {
        "description": (
            "Recocido Simulado para TSP:\n"
            "1) Parte de la solución Greedy como ruta inicial.\n"
            "2) En cada paso propone un vecino haciendo un swap entre dos paraderos internos.\n"
            "3) Calcula obj_actual y obj_candidato con evaluate_route.\n"
            "4) Δ = obj_candidato - obj_actual.\n"
            "   - Si Δ < 0, se acepta siempre.\n"
            "   - Si Δ ≥ 0, se acepta con probabilidad p = exp(-Δ / T).\n"
            "5) La temperatura T se enfría multiplicando por alpha.\n"
            "En 'steps' se guarda por paso:\n"
            "- delta, accept_prob, accepted, temperatura T.\n"
            "- sa_move: before_route, candidate_route, descripción del swap.\n"
            "- formulas: distance, demand, over_capacity, penalty, objective, fitness,\n"
            "            sa_delta, sa_acceptance."
        )
    },
}


# ============================================================
# 6c. HELPER: Llamar a Gemini (reemplaza Ollama)
# ============================================================

# Variables de entorno para configurar Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # usa uno estable
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL",
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
)


def _strip_deepseek_thinking(text: str) -> str:
    """
    Antes limpiábamos bloques <think>...</think> de DeepSeek.
    Lo dejamos por compatibilidad: si algún modelo los añadiera, se limpian;
    para Gemini no afecta.
    """
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def call_ollama(prompt: str) -> str:
    """
    Mantiene el nombre original para no romper la estructura,
    pero ahora llama a la API de Gemini.
    """
    if not GEMINI_API_KEY:
        return "[ERROR IA] Falta GEMINI_API_KEY en variables de entorno."

    try:
        resp = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": 512,
                    "temperature": 0.2,
                    "topP": 0.9,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # ==========================
        # 1) Intento estándar Gemini
        # ==========================
        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []

            # Concatenar TODOS los 'text' que aparezcan en parts
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    text += p.get("text", "")

        text = (text or "").strip()

        # ==========================
        # 2) Fallbacks alternativos
        # ==========================
        if not text:
            if isinstance(data, dict) and "text" in data:
                text = str(data["text"]).strip()

        # ==========================
        # 3) Si sigue vacío: devolver JSON para debug
        # ==========================
        if not text:
            try:
                raw = json.dumps(data, ensure_ascii=False)
                return "[DEBUG GEMINI] " + raw[:3500]
            except Exception:
                return "[ERROR IA] Respuesta vacía desde Gemini (y no se pudo serializar el JSON)."

        # Por compatibilidad con el helper viejo (<think>...</think>)
        return _strip_deepseek_thinking(text)

    except Exception as e:
        return f"[ERROR IA] No se pudo llamar a Gemini: {e}"

# ============================================================
# 7. MODELOS Pydantic
# ============================================================

class SolveRequest(BaseModel):
    method: str
    active_routes: List[str]
    demands: Dict[str, int]
    capacity: int = 40


class Step(BaseModel):
    # --------- Genérico (todos los métodos) ----------
    method: str
    iteration: int
    phase: str
    route: List[str]
    distance_km: float
    objective_km: float
    fitness: float
    temperature: Optional[float] = None
    best_so_far: bool = False

    # --------- Campos extra para GREEDY ----------
    from_stop: Optional[str] = None
    to_stop: Optional[str] = None
    edge_distance_km: Optional[float] = None
    cumulative_distance_km: Optional[float] = None
    remaining_stops_count: Optional[int] = None
    event: Optional[str] = None

    # --------- Campos para GA ----------
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    worst_fitness: Optional[float] = None
    population_size: Optional[int] = None
    ga_events: Optional[Dict] = None

    # --------- Campos para SA ----------
    delta: Optional[float] = None
    accept_prob: Optional[float] = None
    accepted: Optional[bool] = None
    sa_move: Optional[Dict] = None

    # --------- Campo común para texto explicativo ----------
    details: Optional[str] = None

    # --------- NUEVO: fórmulas usadas en el cálculo ----------
    formulas: Optional[Dict[str, str]] = None


class SolveResponse(BaseModel):
    method: str
    route: List[str]
    label: str
    distance_km: float
    time_min: float
    demand: int
    capacity: int
    over_capacity: int
    penalty_km: float
    objective_km: float
    fitness: float
    cpu_ms: float
    steps: List[Step]


class BusSolution(BaseModel):
    bus_id: str
    label: str
    route: List[str]
    distance_km: float
    time_min: float
    demand: int
    capacity: int
    over_capacity: int
    penalty_km: float
    objective_km: float
    fitness: float
    cpu_ms: float
    steps: List[Step]


class MultiSolveResponse(BaseModel):
    method: str
    buses: List[BusSolution]
    global_bus: Optional[BusSolution] = None


# ============================================================
# 7b. MODELOS: IA explicadora
# ============================================================

class AskAIRequest(BaseModel):
    question: str                         # pregunta en lenguaje natural
    method: Optional[str] = None          # "greedy" | "ga" | "sa" | None
    solve_context: Optional[Dict] = None  # SolveResponse o MultiSolveResponse (JSON)


class AskAIResponse(BaseModel):
    answer: str


# ============================================================
# 8. ENDPOINTS FRONT
# ============================================================

@app.get("/", response_class=HTMLResponse)
def serve_front():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado")
    return INDEX_FILE.read_text(encoding="utf-8")


@app.get("/styles.css")
def serve_styles():
    if not CSS_FILE.exists():
        raise HTTPException(status_code=404, detail="styles.css no encontrado")
    return FileResponse(CSS_FILE, media_type="text/css")


# ============================================================
# 9. ENDPOINT /solve (bus global único, para compatibilidad)
# ============================================================

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    t0 = time.time()

    method = req.method.lower()
    active_routes = req.active_routes
    demands = req.demands
    capacity = req.capacity

    if capacity <= 0:
        raise HTTPException(status_code=400, detail="Capacidad debe ser mayor a 0.")

    stops_to_visit = get_stops_from_active_routes(active_routes, demands)

    if len(stops_to_visit) < 2:
        raise HTTPException(status_code=400, detail="No hay suficientes paraderos activos.")

    random.seed(42)

    if method == "greedy":
        info = greedy_tsp(stops_to_visit, demands, capacity)
    elif method == "ga":
        info = ga_tsp(stops_to_visit, demands, capacity)
    elif method == "sa":
        info = sa_tsp(stops_to_visit, demands, capacity)
    else:
        raise HTTPException(status_code=400, detail=f"Método no soportado: {req.method}")

    cpu_ms = (time.time() - t0) * 1000.0

    route = info["route"]
    label = " → ".join(route)

    return SolveResponse(
        method=method,
        route=route,
        label=label,
        distance_km=info["total_distance_km"],
        time_min=info["time_min"],
        demand=info["total_demand"],
        capacity=capacity,
        over_capacity=info["over_capacity"],
        penalty_km=info["penalty_km"],
        objective_km=info["objective_km"],
        fitness=info["fitness"],
        cpu_ms=cpu_ms,
        steps=[Step(**s) for s in info.get("steps", [])]
    )


# ============================================================
# 10. ENDPOINT /solve_multi (un bus por ruta + bus global)
# ============================================================

@app.post("/solve_multi", response_model=MultiSolveResponse)
def solve_multi(req: SolveRequest):
    method = req.method.lower()
    active_routes = req.active_routes
    demands = req.demands
    capacity = req.capacity

    if capacity <= 0:
        raise HTTPException(status_code=400, detail="Capacidad debe ser mayor a 0.")

    if not active_routes:
        raise HTTPException(status_code=400, detail="Debes seleccionar al menos una ruta oficial.")

    random.seed(42)

    def run_solver(stops_to_visit: List[str]) -> Dict:
        if method == "greedy":
            return greedy_tsp(stops_to_visit, demands, capacity)
        elif method == "ga":
            return ga_tsp(stops_to_visit, demands, capacity)
        elif method == "sa":
            return sa_tsp(stops_to_visit, demands, capacity)
        else:
            raise HTTPException(status_code=400, detail=f"Método no soportado: {req.method}")

    buses: List[BusSolution] = []

    # 1) Un bus por cada ruta oficial activa
    for rid in active_routes:
        if rid not in ROUTE_DEFS:
            continue
        route_stops = ROUTE_DEFS[rid]["stops"]
        if len(route_stops) < 2:
            continue

        t0 = time.time()
        info = run_solver(route_stops)
        cpu_ms = (time.time() - t0) * 1000.0

        buses.append(
            BusSolution(
                bus_id=rid,
                label=ROUTE_DEFS[rid]["label"],
                route=info["route"],
                distance_km=info["total_distance_km"],
                time_min=info["time_min"],
                demand=info["total_demand"],
                capacity=capacity,
                over_capacity=info["over_capacity"],
                penalty_km=info["penalty_km"],
                objective_km=info["objective_km"],
                fitness=info["fitness"],
                cpu_ms=cpu_ms,
                steps=[Step(**s) for s in info.get("steps", [])],
            )
        )

    if not buses:
        raise HTTPException(status_code=400, detail="No se pudo optimizar ninguna ruta activa.")

    # 2) Bus GLOBAL que pasa por todos los paraderos combinados
    all_stops = get_stops_from_active_routes(active_routes, demands)
    t0 = time.time()
    global_info = run_solver(all_stops)
    cpu_ms_global = (time.time() - t0) * 1000.0

    global_bus = BusSolution(
        bus_id="GLOBAL",
        label="Bus global (todas las rutas)",
        route=global_info["route"],
        distance_km=global_info["total_distance_km"],
        time_min=global_info["time_min"],
        demand=global_info["total_demand"],
        capacity=capacity,
        over_capacity=global_info["over_capacity"],
        penalty_km=global_info["penalty_km"],
        objective_km=global_info["objective_km"],
        fitness=global_info["fitness"],
        cpu_ms=cpu_ms_global,
        steps=[Step(**s) for s in global_info.get("steps", [])],
    )

    return MultiSolveResponse(
        method=method,
        buses=buses,
        global_bus=global_bus,
    )


# ============================================================
# 11. ENDPOINT /ask_ai: IA que explica métodos y resultados
# ============================================================

@app.post("/ask_ai", response_model=AskAIResponse)
def ask_ai(req: AskAIRequest):
    """
    IA explicadora: 
      - Conoce cómo funcionan evaluate_route, greedy, ga, sa.
      - Opcionalmente recibe el JSON de una corrida (/solve o /solve_multi)
        y explica paso a paso qué pasó.
    """

    # 1) Conocimiento estático de algoritmos
    algorithms_text = []
    for key, info in ALGORITHMS_KNOWLEDGE.items():
        algorithms_text.append(f"== {key} ==\n{info.get('description', '')}\n")
        formulas = info.get("formulas")
        if formulas:
            algorithms_text.append("Fórmulas asociadas:\n")
            for name, formula in formulas.items():
                algorithms_text.append(f"- {name}: {formula}")
        algorithms_text.append("\n")

    algorithms_block = "\n".join(algorithms_text)

    # 2) Contexto de la corrida (si viene)
    solve_block = ""
    if req.solve_context is not None:
        try:
            solve_block = json.dumps(req.solve_context, ensure_ascii=False)
            if len(solve_block) > 6000:
                solve_block = solve_block[:6000] + "\n... [truncado para la IA] ..."
        except Exception:
            solve_block = str(req.solve_context)

    # 3) Método (si se especifica)
    method_info = f"Método principal: {req.method}" if req.method else "Método principal: (no especificado)."

    # 4) Prompt completo
    prompt = f"""
Eres una IA experta en metaheurísticas para ruteo de buses universitarios.
El sistema está implementado en Python/FastAPI y utiliza estos métodos:

{algorithms_block}

Reglas:
- Explica SIEMPRE en español, claro y didáctico.
- Cuando hables de fórmulas, usa los nombres: distance_km, objective_km,
  over_capacity, penalty_km, fitness, delta, accept_prob, etc.
- No inventes números: si no ves un valor en el contexto, explica la fórmula de forma general.
- Si hay 'steps', puedes usar iteration, phase, delta, accept_prob, details, formulas, etc.
- Si la pregunta es teórica (sin solve_context), responde solo con la información de los métodos.
- Si la pregunta es sobre una corrida (con solve_context), usa ese JSON para explicar qué pasó.

{method_info}

Pregunta del usuario:
\"\"\"{req.question}\"\"\"


Contexto de resultados (SolveResponse o MultiSolveResponse):
\"\"\"json
{solve_block}
\"\"\"


Responde en lenguaje humano, como profesor, pero sin perder rigor técnico.
"""

    answer = call_ollama(prompt)
    return AskAIResponse(answer=answer)
