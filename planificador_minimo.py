import pulp
import pandas as pd
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import date, timedelta, datetime
import json
import plotly.io as pio

# Le decimos a Plotly que el renderizador por defecto es el navegador
pio.renderers.default = "browser"
# 1. DEFINICIÓN DE ESTRUCTURAS DE DATOS (Corregidas)
@dataclass
class Resource:
    name: str
    availability: Dict[str, float] = field(default_factory=dict)

@dataclass
class Task:
    id: str
    name: str
    project_id: str
    hours: float
    sequence: int
    compatible_resources: List[str] = field(default_factory=list)
    subcontractable: bool = False
    requires_client_validation: bool = False
    validation_delay_days: int = 0
    predecessors: List[str] = field(default_factory=list)

@dataclass
class Project:
    id: str
    name: str

# 2. CLASE DEL OPTIMIZADOR CON PULP
class OptimizadorProyectos:
    def __init__(self, proyectos, tareas, recursos, params):
        self.proyectos = proyectos
        self.tareas = tareas
        self.recursos = recursos
        self.params = params
        self.task_map = {t.id: t for t in self.tareas}
        self.project_map = {p.id: p for p in self.proyectos}
        self.model = pulp.LpProblem("Maximizacion_Proyectos", pulp.LpMaximize)
        self.vars = {}

    @classmethod
    def desde_json(cls, ruta_fichero: str):
        print(f"Cargando instancia desde: {ruta_fichero}")
        with open(ruta_fichero, 'r') as f:
            data = json.load(f)
        proyectos = [Project(**p) for p in data['proyectos']]
        tareas = [Task(**t) for t in data['tareas']]
        recursos = [Resource(**r) for r in data['recursos']]
        params = data['parametros']
        params['fecha_inicio'] = datetime.strptime(params['fecha_inicio'], '%Y-%m-%d').date()
        return cls(proyectos, tareas, recursos, params)

    def construir_modelo(self):
        """Traduce el modelo matemático a código PuLP (versión corregida y mejorada)."""
        H = self.params["horizonte_dias"]
        DIAS = list(range(1, H + 1))
        
        # --- Variables de Decisión ---
        self.vars['x'] = pulp.LpVariable.dicts("horas", ((t.id, r.name, d) for t in self.tareas for r in self.recursos for d in DIAS), lowBound=0, cat='Continuous')
        self.vars['s'] = pulp.LpVariable.dicts("inicio_tarea", (t.id for t in self.tareas), lowBound=1, upBound=H, cat='Integer')
        self.vars['e'] = pulp.LpVariable.dicts("fin_tarea", (t.id for t in self.tareas), lowBound=1, upBound=H, cat='Integer')
        self.vars['SubT'] = pulp.LpVariable.dicts("subcontrata_tarea", (t.id for t in self.tareas), cat='Binary')
        self.vars['C'] = pulp.LpVariable.dicts("completa_proyecto", (p.id for p in self.proyectos), cat='Binary')
        self.vars['SubP'] = pulp.LpVariable.dicts("subcontrata_proyecto", (p.id for p in self.proyectos), cat='Binary')
        self.vars['CI'] = pulp.LpVariable.dicts("completa_interno", (p.id for p in self.proyectos), cat='Binary')
        self.vars['CS'] = pulp.LpVariable.dicts("completa_subcontratado", (p.id for p in self.proyectos), cat='Binary')
        
        # --- Función Objetivo ---
        alpha = self.params['alpha']
        beta = self.params['beta']
        self.model += pulp.lpSum(alpha * self.vars['CI'][p.id] + beta * self.vars['CS'][p.id] for p in self.proyectos), "Beneficio_Total"

        # --- Restricciones ---
        M = H * 10  # Big M
        for t in self.tareas:
            # Variable auxiliar z_td: 1 si la tarea t está activa en el día d
            z_td = pulp.LpVariable.dicts(f"activa_{t.id}", DIAS, cat='Binary')

            # 1. Horas y Subcontratación
            self.model += pulp.lpSum(self.vars['x'][t.id, r.name, d] for r in self.recursos if r.name in t.compatible_resources for d in DIAS) == t.hours * (1 - self.vars['SubT'][t.id]), f"Horas_{t.id}"
            if not t.subcontractable:
                self.model += self.vars['SubT'][t.id] == 0, f"No_Sub_{t.id}"

            # 2. Lógica de días de inicio y fin (Formulación Robusta)
            for d in DIAS:
                # Vincula las horas con la variable de actividad z_td
                self.model += pulp.lpSum(self.vars['x'][t.id, r.name, d] for r in self.recursos) <= M * z_td[d], f"Link_activa_{t.id}_{d}"
                
                # El día de fin es mayor o igual que cualquier día activo
                self.model += self.vars['e'][t.id] >= d * z_td[d], f"Def_Fin_{t.id}_{d}"
                
                # El día de inicio es menor o igual que cualquier día activo
                self.model += self.vars['s'][t.id] <= d + M * (1 - z_td[d]), f"Def_Inicio_{t.id}_{d}"
            
            # Lógicamente, el inicio es antes o el mismo día que el fin
            self.model += self.vars['s'][t.id] <= self.vars['e'][t.id], f"Inicio_antes_de_Fin_{t.id}"

            # 3. Secuencialidad y Validación de Cliente
            for pred_id in t.predecessors:
                pred_task = self.task_map[pred_id]
                retraso_validacion = pred_task.validation_delay_days if pred_task.requires_client_validation else 0
                self.model += self.vars['s'][t.id] >= self.vars['e'][pred_id] + 1 + retraso_validacion, f"Secuencia_{pred_id}_{t.id}"

        # 4. Disponibilidad de Recursos
        for r in self.recursos:
            for d in DIAS:
                dia_semana = self.params["dias_laborables"][(d-1) % len(self.params["dias_laborables"])]
                disponibilidad = r.availability.get(dia_semana, 0)
                self.model += pulp.lpSum(self.vars['x'][t.id, r.name, d] for t in self.tareas) <= disponibilidad, f"Disp_{r.name}_{d}"

        # 5. Lógica de Proyectos
        for p in self.proyectos:
            tareas_del_proyecto = [t for t in self.tareas if t.project_id == p.id]
            if tareas_del_proyecto:
                # Un proyecto se completa si TODAS sus tareas (no subcontratadas) finalizan en el horizonte
                for t in tareas_del_proyecto:
                    self.model += self.vars['e'][t.id] - M * self.vars['SubT'][t.id] <= H + M * (1 - self.vars['C'][p.id]), f"Completitud_{p.id}_{t.id}"
                
                # Lógica de proyecto subcontratado
                self.model += pulp.lpSum(self.vars['SubT'][t.id] for t in tareas_del_proyecto) <= M * self.vars['SubP'][p.id], f"Define_SubP_upper_{p.id}"
                for t in tareas_del_proyecto:
                    self.model += self.vars['SubP'][p.id] >= self.vars['SubT'][t.id], f"Define_SubP_lower_{p.id}_{t.id}"

            # 6. Linealización
            self.model += self.vars['CI'][p.id] <= self.vars['C'][p.id], f"LinCI1_{p.id}"
            self.model += self.vars['CI'][p.id] <= 1 - self.vars['SubP'][p.id], f"LinCI2_{p.id}"
            self.model += self.vars['CI'][p.id] >= self.vars['C'][p.id] - self.vars['SubP'][p.id], f"LinCI3_{p.id}"
            self.model += self.vars['CS'][p.id] <= self.vars['C'][p.id], f"LinCS1_{p.id}"
            self.model += self.vars['CS'][p.id] <= self.vars['SubP'][p.id], f"LinCS2_{p.id}"
            self.model += self.vars['CS'][p.id] >= self.vars['C'][p.id] + self.vars['SubP'][p.id] - 1, f"LinCS3_{p.id}"

    def resolver(self, solver=None):
        self.model.solve(solver)
        return pulp.LpStatus[self.model.status]

    def mostrar_resultados(self):
        print("\n--- RESULTADOS DE LA OPTIMIZACIÓN ---")
        print(f"Estado: {pulp.LpStatus[self.model.status]}")
        print(f"Beneficio Total Óptimo: {pulp.value(self.model.objective):.2f}")
        print("\n**Estado de los Proyectos:**")
        for p in self.proyectos:
            if pulp.value(self.vars['C'][p.id]) == 1:
                tipo = "Subcontratado" if pulp.value(self.vars['SubP'][p.id]) == 1 else "Interno"
                print(f"- Proyecto '{p.name}': COMPLETADO ({tipo})")
            else:
                print(f"- Proyecto '{p.name}': NO COMPLETADO")
        print("\n**Detalle de Tareas:**")
        for t in self.tareas:
            if pulp.value(self.vars['SubT'][t.id]) == 1:
                print(f"- Tarea '{t.name}' ({self.task_map[t.id].project_id}): SUBCONTRATADA")
            else:
                inicio = pulp.value(self.vars['s'][t.id]) if self.vars['s'][t.id].varValue is not None else 'N/A'
                fin = pulp.value(self.vars['e'][t.id]) if self.vars['e'][t.id].varValue is not None else 'N/A'
                print(f"- Tarea '{t.name}' ({self.task_map[t.id].project_id}): Inicia día {inicio:.0f}, Finaliza día {fin:.0f}")
        print("\n**Asignación de Horas (resumen):**")
        for t in self.tareas:
            if pulp.value(self.vars['SubT'][t.id]) == 0:
                for r in self.recursos:
                    horas_asignadas = sum(pulp.value(self.vars['x'][t.id, r.name, d]) for d in range(1, self.params['horizonte_dias'] + 1))
                    if horas_asignadas > 0.1:
                        print(f"  - Recurso '{r.name}' trabaja {horas_asignadas:.1f}h en la tarea '{t.name}'")

    # La función visualizar_gantt NO necesita cambios, ya que la configuración se hizo al principio del script.
    def visualizar_gantt(self):
        """Prepara los datos y genera un diagrama de Gantt con Plotly."""
        
        if self.model.status != pulp.LpStatusOptimal:
            print("\nNo se puede generar el Gantt: no se encontró una solución óptima.")
            return

        fecha_inicio = self.params["fecha_inicio"]
        gantt_data = []

        for t in self.tareas:
            if pulp.value(self.vars['SubT'][t.id]) == 0:
                start_day = pulp.value(self.vars['s'][t.id])
                finish_day = pulp.value(self.vars['e'][t.id])
                if start_day is None or finish_day is None: continue
                
                assigned_resource = "N/A"
                for r in self.recursos:
                    if sum(pulp.value(self.vars['x'][t.id, r.name, d]) for d in range(1, self.params['horizonte_dias'] + 1)) > 0.1:
                        assigned_resource = r.name
                        break
                
                gantt_data.append(dict(
                    Task=f"{self.project_map[t.project_id].name}: {t.name}",
                    Start=(fecha_inicio + timedelta(days=int(start_day)-1)),
                    Finish=(fecha_inicio + timedelta(days=int(finish_day))),
                    Project=self.project_map[t.project_id].name,
                    Resource=assigned_resource
                ))

        if not gantt_data:
            print("\nNo hay tareas internas planificadas para mostrar en el Gantt.")
            return
            
        df = pd.DataFrame(gantt_data)
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Project",
                          title="Diagrama de Gantt de la Planificación",
                          hover_data=["Resource"])
        
        fig.update_yaxes(categoryorder="total ascending")
        fig.show()

# 3. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    fichero_instancia = r'instancias\instancia_mediana.json'
    optimizador = OptimizadorProyectos.desde_json(fichero_instancia)
    print("Construyendo el modelo de optimización...")
    optimizador.construir_modelo()
    print("Resolviendo...")
    estado = optimizador.resolver()
    if estado == 'Optimal':
        optimizador.mostrar_resultados()
        optimizador.visualizar_gantt()
    else:
        print(f"\nNo se encontró una solución óptima. Estado: {estado}")