import pulp
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# 1. DEFINICIÓN DE ESTRUCTURAS DE DATOS (Adaptado de data_models.py)
# --------------------------------------------------------------------
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
    predecessors: List[str] = field(default_factory=list) # Lista de IDs de tareas predecesoras

@dataclass
class Project:
    id: str
    name: str

# 2. INSTANCIA DEL PROBLEMA (Datos de prueba sencillos)
# --------------------------------------------------------------------
def get_instancia_sencilla():
    """Devuelve un conjunto de datos simple para probar el modelo."""
    proyectos = [
        Project(id="P1", name="Proyecto Alpha"),
        Project(id="P2", name="Proyecto Beta")
    ]

    tareas = [
        # Tareas del Proyecto Alpha
        Task(id="T1", name="Análisis Alpha", project_id="P1", hours=8, sequence=1, compatible_resources=["Recurso A", "Recurso B"]),
        Task(id="T2", name="Diseño Alpha", project_id="P1", hours=12, sequence=2, compatible_resources=["Recurso A"], predecessors=["T1"]),
        # Tareas del Proyecto Beta
        Task(id="T3", name="Investigación Beta", project_id="P2", hours=10, sequence=1, compatible_resources=["Recurso A", "Recurso B"], subcontractable=True),
        Task(id="T4", name="Prototipo Beta", project_id="P2", hours=16, sequence=2, compatible_resources=["Recurso B"], predecessors=["T3"])
    ]

    recursos = [
        Resource(name="Recurso A", availability={"Lunes": 8, "Martes": 8, "Miercoles": 8, "Jueves": 8, "Viernes": 4}),
        Resource(name="Recurso B", availability={"Lunes": 8, "Martes": 8, "Miercoles": 8, "Jueves": 8, "Viernes": 8})
    ]
    
    # Parámetros del modelo
    parametros = {
        "horizonte_dias": 20, # Planificamos para 20 días
        "alpha": 100, # Beneficio por proyecto interno
        "beta": 50, # Beneficio por proyecto subcontratado
        "dias_laborables": ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]
    }

    return proyectos, tareas, recursos, parametros

# 3. CLASE DEL OPTIMIZADOR CON PULP
# --------------------------------------------------------------------
class OptimizadorProyectos:
    """
    Implementa el modelo de optimización para maximizar proyectos completados.
    """
    def __init__(self, proyectos, tareas, recursos, params):
        self.proyectos = proyectos
        self.tareas = tareas
        self.recursos = recursos
        self.params = params
        
        # Estructuras de datos para fácil acceso
        self.task_map = {t.id: t for t in self.tareas}
        
        # Modelo PuLP
        self.model = pulp.LpProblem("Maximizacion_Proyectos", pulp.LpMaximize)
        self.vars = {}

    def construir_modelo(self):
        """Traduce el modelo matemático a código PuLP."""
        H = self.params["horizonte_dias"]
        DIAS = list(range(1, H + 1))
        
        # --- Variables de Decisión ---
        self.vars['x'] = pulp.LpVariable.dicts("horas",
            ((t.id, r.name, d) for t in self.tareas for r in self.recursos for d in DIAS),
            lowBound=0, cat='Continuous')
        
        self.vars['s'] = pulp.LpVariable.dicts("inicio_tarea", (t.id for t in self.tareas), lowBound=1, cat='Integer')
        self.vars['e'] = pulp.LpVariable.dicts("fin_tarea", (t.id for t in self.tareas), lowBound=1, cat='Integer')
        
        self.vars['SubT'] = pulp.LpVariable.dicts("subcontrata_tarea", (t.id for t in self.tareas), cat='Binary')
        self.vars['C'] = pulp.LpVariable.dicts("completa_proyecto", (p.id for p in self.proyectos), cat='Binary')
        self.vars['SubP'] = pulp.LpVariable.dicts("subcontrata_proyecto", (p.id for p in self.proyectos), cat='Binary')
        
        # Variables para linealización del objetivo
        self.vars['CI'] = pulp.LpVariable.dicts("completa_interno", (p.id for p in self.proyectos), cat='Binary')
        self.vars['CS'] = pulp.LpVariable.dicts("completa_subcontratado", (p.id for p in self.proyectos), cat='Binary')
        
        # --- Función Objetivo ---
        alpha = self.params['alpha']
        beta = self.params['beta']
        self.model += pulp.lpSum(alpha * self.vars['CI'][p.id] + beta * self.vars['CS'][p.id] for p in self.proyectos), "Beneficio_Total_Proyectos"

        # --- Restricciones ---
        
        # 1. Dedicación de Horas
        for t in self.tareas:
            self.model += pulp.lpSum(self.vars['x'][t.id, r.name, d] for r in self.recursos if r.name in t.compatible_resources for d in DIAS) == t.hours * (1 - self.vars['SubT'][t.id]), f"Horas_Tarea_{t.id}"

        # 2. Disponibilidad de Recursos
        for r in self.recursos:
            for d in DIAS:
                # Mapear día numérico a nombre de día
                dia_semana = self.params["dias_laborables"][(d-1) % len(self.params["dias_laborables"])]
                disponibilidad = r.availability.get(dia_semana, 0)
                self.model += pulp.lpSum(self.vars['x'][t.id, r.name, d] for t in self.tareas) <= disponibilidad, f"Disponibilidad_{r.name}_{d}"
        
        # 3. Secuencialidad
        for t in self.tareas:
            for pred_id in t.predecessors:
                self.model += self.vars['s'][t.id] >= self.vars['e'][pred_id] + 1, f"Secuencia_{pred_id}_{t.id}"

        # 4. Lógica de días de inicio y fin (versión simplificada)
        M = H * 10 # Big M
        for t in self.tareas:
            for d in DIAS:
                # Si se trabaja en el día d, el día de fin es al menos d
                self.model += self.vars['e'][t.id] >= d * pulp.lpSum(self.vars['x'][t.id, r.name, d] for r in self.recursos) / M, f"Def_Fin_{t.id}_{d}"
            # El inicio es antes o igual que el fin
            self.model += self.vars['s'][t.id] <= self.vars['e'][t.id], f"Inicio_antes_de_Fin_{t.id}"

        # 5. Completitud del Proyecto
        for p in self.proyectos:
            tareas_del_proyecto = [t for t in self.tareas if t.project_id == p.id]
            if tareas_del_proyecto:
                # El proyecto se completa si todas sus tareas finalizan dentro del horizonte
                for t in tareas_del_proyecto:
                     self.model += t.hours <= M * self.vars['C'][p.id] + M * self.vars['SubT'][t.id], f"Completitud_{p.id}_{t.id}"
                     #self.model += self.vars['e']

        # 6. Lógica de Subcontratación
        for t in self.tareas:
            if not t.subcontractable:
                self.model += self.vars['SubT'][t.id] == 0, f"No_Subcontratable_{t.id}"
        
        for p in self.proyectos:
            tareas_del_proyecto = [t for t in self.tareas if t.project_id == p.id]
            self.model += pulp.lpSum(self.vars['SubT'][t.id] for t in tareas_del_proyecto) <= M * self.vars['SubP'][p.id], f"Define_SubP_upper_{p.id}"
            for t in tareas_del_proyecto:
                self.model += self.vars['SubP'][p.id] >= self.vars['SubT'][t.id], f"Define_SubP_lower_{p.id}_{t.id}"

        # 7. Restricciones de linealización del objetivo
        for p in self.proyectos:
            self.model += self.vars['CI'][p.id] <= self.vars['C'][p.id], f"LinCI1_{p.id}"
            self.model += self.vars['CI'][p.id] <= 1 - self.vars['SubP'][p.id], f"LinCI2_{p.id}"
            self.model += self.vars['CI'][p.id] >= self.vars['C'][p.id] - self.vars['SubP'][p.id], f"LinCI3_{p.id}"
            self.model += self.vars['CS'][p.id] <= self.vars['C'][p.id], f"LinCS1_{p.id}"
            self.model += self.vars['CS'][p.id] <= self.vars['SubP'][p.id], f"LinCS2_{p.id}"
            self.model += self.vars['CS'][p.id] >= self.vars['C'][p.id] + self.vars['SubP'][p.id] - 1, f"LinCS3_{p.id}"

    def resolver(self):
        """Resuelve el modelo y devuelve el estado."""
        self.model.solve()
        return pulp.LpStatus[self.model.status]

    def mostrar_resultados(self):
        """Imprime los resultados de la optimización en la consola."""
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
                print(f"- Tarea '{t.name}' ({t.project_id}): SUBCONTRATADA")
            else:
                inicio = pulp.value(self.vars['s'][t.id])
                fin = pulp.value(self.vars['e'][t.id])
                print(f"- Tarea '{t.name}' ({t.project_id}): Inicia día {inicio:.0f}, Finaliza día {fin:.0f}")

        print("\n**Asignación de Horas (resumen):**")
        for t in self.tareas:
            if pulp.value(self.vars['SubT'][t.id]) == 0:
                for r in self.recursos:
                    horas_asignadas = sum(pulp.value(self.vars['x'][t.id, r.name, d]) for d in range(1, self.params['horizonte_dias'] + 1))
                    if horas_asignadas > 0:
                        print(f"  - Recurso '{r.name}' trabaja {horas_asignadas:.1f}h en la tarea '{t.name}'")


# 4. EJECUCIÓN PRINCIPAL
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Cargar datos
    proyectos, tareas, recursos, params = get_instancia_sencilla()
    
    # 2. Crear y construir el modelo
    optimizador = OptimizadorProyectos(proyectos, tareas, recursos, params)
    print("Construyendo el modelo de optimización...")
    optimizador.construir_modelo()
    
    # 3. Resolver el problema
    print("Resolviendo... (esto puede tardar unos segundos)")
    estado = optimizador.resolver()
    
    # 4. Mostrar resultados
    if estado == 'Optimal':
        optimizador.mostrar_resultados()
        print(optimizador.vars['x'])
    else:
        print(f"\nNo se encontró una solución óptima. Estado: {estado}")