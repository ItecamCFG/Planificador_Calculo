import json
import pandas as pd
import plotly.express as px
import plotly.io as pio
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import date, timedelta, datetime
import random
import math
import copy

# Configuración de Plotly
pio.renderers.default = "browser"

# --- ESTRUCTURAS DE DATOS ---
@dataclass
class Resource:
    name: str
    availability: Dict[str, float] = field(default_factory=dict)

@dataclass
class Task:
    id: str; name: str; project_id: str; hours: float; sequence: int
    compatible_resources: List[str] = field(default_factory=list)
    subcontractable: bool = False
    requires_client_validation: bool = False
    validation_delay_days: int = 0
    predecessors: List[str] = field(default_factory=list)

@dataclass
class Project:
    id: str
    name: str

# --- CLASE DEL OPTIMIZADOR HEURÍSTICO ---
class OptimizadorProyectosHeuristico:
    def __init__(self, proyectos, tareas, recursos, params):
        self.proyectos = proyectos
        self.tareas = tareas
        self.recursos = recursos
        self.params = params
        self.task_map = {t.id: t for t in self.tareas}
        self.project_map = {p.id: p for p in self.proyectos}
        self.project_tasks = {p.id: sorted([t for t in self.tareas if t.project_id == p.id], key=lambda x: x.sequence) for p in self.proyectos}
        self.resource_map = {r.name: r for r in self.recursos}

    @classmethod
    def desde_json(cls, ruta_fichero: str):
        with open(ruta_fichero, 'r') as f: data = json.load(f)
        proyectos = [Project(**p) for p in data['proyectos']]
        tareas = [Task(**t) for t in data['tareas']]
        recursos = [Resource(**r) for r in data['recursos']]
        params = data['parametros']
        params['fecha_inicio'] = datetime.strptime(params['fecha_inicio'], '%Y-%m-%d').date()
        return cls(proyectos, tareas, recursos, params)

    def _calcular_disponibilidad_diaria(self):
        # Pre-calcula la disponibilidad para cada recurso y día numérico
        disponibilidad = {}
        for r in self.recursos:
            disp_recurso = {}
            for d in range(1, self.params['horizonte_dias'] + 1):
                fecha = self.params['fecha_inicio'] + timedelta(days=d-1)
                dia_semana = fecha.strftime('%A') # Lunes, Martes...
                disp_recurso[d] = r.availability.get(dia_semana, 0)
            disponibilidad[r.name] = disp_recurso
        return disponibilidad

    def generar_solucion_inicial(self):
        """Crea una primera planificación factible de forma greedy."""
        planificacion = {}
        # Un calendario para cada recurso para saber qué horas tiene ocupadas cada día
        calendario_recursos = {r.name: {d: 0 for d in range(1, self.params['horizonte_dias'] + 1)} for r in self.recursos}
        
        tareas_ordenadas = sorted(self.tareas, key=lambda t: t.sequence)
        
        for tarea in tareas_ordenadas:
            recurso_asignado = random.choice(tarea.compatible_resources)
            horas_restantes = tarea.hours
            
            dia_inicio_min = 1
            for pred_id in tarea.predecessors:
                if pred_id in planificacion:
                    pred = planificacion[pred_id]
                    retraso = self.task_map[pred_id].validation_delay_days if self.task_map[pred_id].requires_client_validation else 0
                    dia_inicio_min = max(dia_inicio_min, pred['fin'] + 1 + retraso)

            dia_actual = dia_inicio_min
            dia_inicio_real = -1
            dias_trabajados = []

            while horas_restantes > 0 and dia_actual <= self.params['horizonte_dias']:
                disp_diaria = self.resource_map[recurso_asignado].availability.get((self.params['fecha_inicio'] + timedelta(days=dia_actual-1)).strftime('%A'), 0)
                horas_libres_recurso = disp_diaria - calendario_recursos[recurso_asignado].get(dia_actual, 0)
                
                if horas_libres_recurso > 0:
                    if dia_inicio_real == -1:
                        dia_inicio_real = dia_actual

                    horas_a_trabajar = min(horas_restantes, horas_libres_recurso)
                    calendario_recursos[recurso_asignado][dia_actual] += horas_a_trabajar
                    horas_restantes -= horas_a_trabajar
                    dias_trabajados.append(dia_actual)
                
                dia_actual += 1

            if horas_restantes == 0:
                planificacion[tarea.id] = {
                    'recurso': recurso_asignado,
                    'inicio': dia_inicio_real,
                    'fin': max(dias_trabajados),
                    'dias_trabajados': dias_trabajados
                }
        return planificacion, calendario_recursos

    def calcular_coste(self, planificacion):
        """Calcula el 'coste' (beneficio negativo) de una solución."""
        if not planificacion: return -float('inf')

        beneficio = 0
        gamma = 0.1 # Penalización por duración
        
        for p in self.proyectos:
            tareas_p = self.project_tasks[p.id]
            completado = True
            proyecto_sub = False
            max_fin_dia = 0
            
            for t in tareas_p:
                if t.id not in planificacion:
                    completado = False
                    break
                max_fin_dia = max(max_fin_dia, planificacion[t.id]['fin'])
            
            if completado and max_fin_dia <= self.params['horizonte_dias']:
                # Aquí simplificamos, asumiendo que si se planifica no se subcontrata
                beneficio += self.params['alpha']
        
        # Penalización total por duración
        penalizacion = sum((plan['fin'] - plan['inicio'] + 1) for plan in planificacion.values())
        
        return -(beneficio - gamma * penalizacion)

    def obtener_vecino(self, planificacion, calendario):
        """Genera una planificación vecina haciendo un pequeño cambio."""
        plan_vecino = copy.deepcopy(planificacion)
        cal_vecino = copy.deepcopy(calendario)
        
        if not plan_vecino: return plan_vecino, cal_vecino
            
        tarea_a_modificar_id = random.choice(list(plan_vecino.keys()))
        tarea_obj = self.task_map[tarea_a_modificar_id]
        plan_original = plan_vecino[tarea_a_modificar_id]
        
        # Deshacer la asignación actual
        recurso_original = plan_original['recurso']
        for dia in plan_original['dias_trabajados']:
            horas_trabajadas = tarea_obj.hours / len(plan_original['dias_trabajados']) # Simplificación
            cal_vecino[recurso_original][dia] -= horas_trabajadas

        # Re-planificar la tarea
        recurso_nuevo = random.choice(tarea_obj.compatible_resources)
        horas_restantes = tarea_obj.hours
        
        dia_inicio_min = 1
        for pred_id in tarea_obj.predecessors:
            if pred_id in plan_vecino:
                pred = plan_vecino[pred_id]
                retraso = self.task_map[pred_id].validation_delay_days if self.task_map[pred_id].requires_client_validation else 0
                dia_inicio_min = max(dia_inicio_min, pred['fin'] + 1 + retraso)
        
        dia_actual = dia_inicio_min
        dia_inicio_real = -1
        dias_trabajados = []
        
        while horas_restantes > 0 and dia_actual <= self.params['horizonte_dias']:
            disp_diaria = self.resource_map[recurso_nuevo].availability.get((self.params['fecha_inicio'] + timedelta(days=dia_actual-1)).strftime('%A'), 0)
            horas_libres_recurso = disp_diaria - cal_vecino[recurso_nuevo].get(dia_actual, 0)
            
            if horas_libres_recurso > 0:
                if dia_inicio_real == -1: dia_inicio_real = dia_actual
                horas_a_trabajar = min(horas_restantes, horas_libres_recurso)
                cal_vecino[recurso_nuevo][dia_actual] += horas_a_trabajar
                horas_restantes -= horas_a_trabajar
                dias_trabajados.append(dia_actual)
            
            dia_actual += 1

        if horas_restantes == 0:
            plan_vecino[tarea_a_modificar_id] = {'recurso': recurso_nuevo, 'inicio': dia_inicio_real, 'fin': max(dias_trabajados), 'dias_trabajados': dias_trabajados}
        else: # Si no se pudo re-planificar, se revierte al estado anterior
            return planificacion, calendario

        return plan_vecino, cal_vecino

    def resolver(self, iteraciones=10000, temp_inicial=100, cooling_rate=0.999):
        """Ejecuta el algoritmo de Recocido Simulado."""
        solucion_actual, calendario_actual = self.generar_solucion_inicial()
        coste_actual = self.calcular_coste(solucion_actual)
        
        mejor_solucion = copy.deepcopy(solucion_actual)
        mejor_coste = coste_actual
        
        temp = temp_inicial
        
        for i in range(iteraciones):
            solucion_vecina, cal_vecino = self.obtener_vecino(solucion_actual, calendario_actual)
            coste_vecino = self.calcular_coste(solucion_vecina)
            
            delta_coste = coste_vecino - coste_actual
            
            if delta_coste < 0 or random.uniform(0, 1) < math.exp(-delta_coste / temp):
                solucion_actual, calendario_actual, coste_actual = solucion_vecina, cal_vecino, coste_vecino
                
            if coste_actual < mejor_coste:
                mejor_solucion, mejor_coste = solucion_actual, coste_actual

            temp *= cooling_rate
            if i % 1000 == 0:
                print(f"Iteración {i}/{iteraciones} - Mejor Beneficio: {-mejor_coste:.2f}")

        return mejor_solucion

    def visualizar_gantt(self, planificacion):
        fecha_inicio = self.params["fecha_inicio"]
        dias_lab = self.params["dias_laborables"]
        gantt_data = []

        for t_id, plan in planificacion.items():
            task = self.task_map[t_id]
            start_date = fecha_inicio + timedelta(days=plan['inicio'] - 1)
            finish_date = fecha_inicio + timedelta(days=plan['fin'] - 1)
            
            gantt_data.append(dict(
                Task=f"{self.project_map[task.project_id].name}: {task.name}",
                Start=start_date, Finish=finish_date,
                Project=self.project_map[task.project_id].name,
                Resource=plan['recurso']
            ))

        if not gantt_data: print("\nNo hay tareas para mostrar."); return
        
        df = pd.DataFrame(gantt_data)
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Project", title="Diagrama de Gantt (Heurística)", hover_data=["Resource"])
        fig.update_yaxes(categoryorder="total ascending")
        fig.show()


# 3. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    fichero_instancia = 'instancia_real.json'
    optimizador = OptimizadorProyectosHeuristico.desde_json(fichero_instancia)
    
    print("Resolviendo con Recocido Simulado...")
    solucion_final = optimizador.resolver(iteraciones=20000)
    
    print("\n--- RESULTADOS DE LA HEURÍSTICA ---")
    print(f"Beneficio Final Estimado: {-optimizador.calcular_coste(solucion_final):.2f}")
    
    print("\nGenerando diagrama de Gantt...")
    optimizador.visualizar_gantt(solucion_final)