#   Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s', a') − Q(s, a) ]
#
#   Donde:
#     Q(s, a) = valor estimado de tomar acción 'a' en estado 's'
#     α (alpha) = tasa de aprendizaje (qué tan rápido aprende)
#     r = recompensa recibida
#     γ (gamma) = factor de descuento (cuánto valoran el futuro)
#     max Q(s', a') = mejor acción posible en el estado siguiente

import numpy as np
import random
import json
import os


class AgenteQLearning:


    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Table: empieza vacía, se va llenando durante el entrenamiento
        # Usamos defaultdict-like behavior con un diccionario normal
        self.q_table = {}

        # Estadísticas de entrenamiento
        self.episodios_entrenados = 0
        self.recompensas_por_episodio = []



    def _obtener_q_valor(self, estado_vector, accion):
        """
        Retorna el Q-valor de un par (estado, acción).
        Si nunca se vio ese par, retorna 0 (optimismo inicial neutro).
        """
        estado_key = tuple(estado_vector)
        if estado_key not in self.q_table:
            return 0.0
        return self.q_table[estado_key].get(tuple(accion), 0.0)

    def _obtener_mejor_q_valor(self, estado_vector, acciones_validas):
        """
        Retorna el Q-valor de la MEJOR acción posible en este estado.
        Se usa en la ecuación de Bellman para estimar el futuro.
        """
        if not acciones_validas:
            return 0.0
        valores = [self._obtener_q_valor(estado_vector, a) for a in acciones_validas]
        return max(valores)

    def _actualizar_q_tabla(self, estado, accion, recompensa, estado_siguiente,
                             acciones_siguientes):

        # Valor actual estimado
        q_actual = self._obtener_q_valor(estado, accion)

        # Mejor valor futuro posible (según lo que sabemos ahora)
        mejor_q_futuro = self._obtener_mejor_q_valor(estado_siguiente, acciones_siguientes)

        # Objetivo: recompensa inmediata + valor descontado del futuro
        objetivo = recompensa + self.gamma * mejor_q_futuro

        # Nuevo Q-valor: nos movemos α% hacia el objetivo
        q_nuevo = q_actual + self.alpha * (objetivo - q_actual)

        # Guardar en la tabla
        estado_key = tuple(estado)
        if estado_key not in self.q_table:
            self.q_table[estado_key] = {}
        self.q_table[estado_key][tuple(accion)] = q_nuevo


    def elegir_accion(self, estado_vector, acciones_validas):
        """
        Decide qué acción tomar usando la política epsilon-greedy:

          Con probabilidad ε   → EXPLORAR: elige acción aleatoria
                                  (para descubrir cosas nuevas)
          Con probabilidad 1-ε → EXPLOTAR: elige la mejor acción conocida
                                  (para aprovechar lo aprendido)

        Al inicio ε=1.0 (100% exploración), va bajando con el entrenamiento.
        """
        if not acciones_validas:
            return (-1, 0, 0)  # esperar si no hay acciones válidas

        # ¿Explorar o explotar?
        if random.random() < self.epsilon:
            # EXPLORAR: acción completamente aleatoria
            return random.choice(acciones_validas)
        else:
            # EXPLOTAR: la acción con mayor Q-valor
            return self._mejor_accion(estado_vector, acciones_validas)

    def _mejor_accion(self, estado_vector, acciones_validas):
        """
        Retorna la acción con el Q-valor más alto.
        Si hay empate, elige aleatoriamente entre las mejores.
        """
        valores = [(a, self._obtener_q_valor(estado_vector, a)) for a in acciones_validas]
        max_valor = max(v for _, v in valores)

        # Todas las acciones con el valor máximo (puede haber empate)
        mejores = [a for a, v in valores if v == max_valor]
        return random.choice(mejores)

    def elegir_accion_greedy(self, estado_vector, acciones_validas):
        """
        Versión 100% codiciosa (sin exploración).
        Se usa para EVALUAR el agente entrenado, no para entrenar.
        """
        if not acciones_validas:
            return (-1, 0, 0)
        return self._mejor_accion(estado_vector, acciones_validas)


    def entrenar_episodio(self, entorno):
        """
        Corre un episodio completo de entrenamiento.
        El agente interactúa con el entorno turno a turno,
        actualizando la Q-Table con cada experiencia.

        Retorna la recompensa total del episodio.
        """
        # Reiniciar el entorno
        estado = entorno.reset()
        estado_vec = entorno.estado_a_vector(estado)

        recompensa_total = 0
        terminado = False

        while not terminado:
            # Obtener acciones válidas en el estado actual
            acciones_validas = entorno.obtener_acciones_validas()

            # Elegir acción (explorar o explotar)
            accion = self.elegir_accion(estado_vec, acciones_validas)

            # Ejecutar la acción en el entorno
            estado_sig, recompensa, terminado, info = entorno.step(accion)
            estado_sig_vec = entorno.estado_a_vector(estado_sig)

            # Obtener las acciones válidas en el estado siguiente
            acciones_sig = entorno.obtener_acciones_validas() if not terminado else []

            # Actualizar la Q-Table con esta experiencia
            self._actualizar_q_tabla(
                estado_vec, accion, recompensa, estado_sig_vec, acciones_sig
            )

            # Avanzar al siguiente estado
            estado_vec = estado_sig_vec
            recompensa_total += recompensa

        # Al final del episodio, reducir epsilon (explorar menos)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodios_entrenados += 1
        self.recompensas_por_episodio.append(recompensa_total)

        return recompensa_total


    def guardar(self, ruta="q_table.json"):
        """
        Guarda la Q-Table en un archivo JSON.
        """
        # Convertir las tuplas a strings (JSON no soporta tuplas como claves)
        datos = {
            "q_table":            {str(k): {str(a): v for a, v in acciones.items()}
                                   for k, acciones in self.q_table.items()},
            "epsilon":            self.epsilon,
            "episodios":          self.episodios_entrenados,
            "recompensas":        self.recompensas_por_episodio,
            "hiperparametros":    {
                "alpha":          self.alpha,
                "gamma":          self.gamma,
                "epsilon_min":    self.epsilon_min,
                "epsilon_decay":  self.epsilon_decay,
            }
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)
        print(f"Q-Table guardada en '{ruta}' "
              f"({len(self.q_table)} estados conocidos)")

    def cargar(self, ruta="q_table.json"):
        """
        Carga una Q-Table previamente guardada.
        """
        if not os.path.exists(ruta):
            print(f"No se encontró '{ruta}'. Empezando desde cero.")
            return

        with open(ruta, "r", encoding="utf-8") as f:
            datos = json.load(f)

        # Reconstruir las tuplas desde strings
        self.q_table = {}
        for estado_str, acciones in datos["q_table"].items():
            estado_tuple = eval(estado_str)
            self.q_table[estado_tuple] = {
                eval(accion_str): valor
                for accion_str, valor in acciones.items()
            }

        self.epsilon              = datos["epsilon"]
        self.episodios_entrenados = datos["episodios"]
        self.recompensas_por_episodio = datos["recompensas"]

        print(f"Q-Table cargada desde '{ruta}'")
        print(f"Episodios previos: {self.episodios_entrenados}")
        print(f"Estados conocidos: {len(self.q_table)}")
        print(f"Epsilon actual:    {self.epsilon:.4f}")



    def resumen(self):
        """Muestra un resumen del estado del agente."""
        print(f"\n{'='*50}")
        print(f"  AGENTE Q-LEARNING")
        print(f"{'='*50}")
        print(f"  Episodios entrenados: {self.episodios_entrenados}")
        print(f"  Estados en Q-Table:   {len(self.q_table)}")
        print(f"  Epsilon actual:       {self.epsilon:.4f}")
        print(f"  Alpha (aprendizaje):  {self.alpha}")
        print(f"  Gamma (descuento):    {self.gamma}")

        if self.recompensas_por_episodio:
            ultimas = self.recompensas_por_episodio[-100:]
            print(f"  Recompensa promedio (últimos 100): {np.mean(ultimas):.1f}")
            print(f"  Recompensa máxima lograda:         {max(self.recompensas_por_episodio):.1f}")
        print()

    def explicar_decision(self, estado_vector, acciones_validas, entorno):
        """
        Explica en lenguaje natural por qué el agente elige una acción.
        Muy útil para la demostración en la feria (XRL / Explicabilidad).
        """
        if not acciones_validas:
            print("No hay acciones válidas disponibles.")
            return

        # Calcular Q-valores para todas las acciones válidas
        valores = []
        for accion in acciones_validas:
            q_val = self._obtener_q_valor(estado_vector, accion)
            valores.append((accion, q_val))

        # Ordenar de mayor a menor Q-valor
        valores.sort(key=lambda x: x[1], reverse=True)

        print("EXPLICACIÓN DE DECISIÓN DEL AGENTE:")
        print(f"Epsilon actual: {self.epsilon:.2f} "
              f"({'explorando' if self.epsilon > 0.3 else 'explotando conocimiento'})")
        print(f"Top 5 acciones evaluadas:")

        nombres_v = {0: "esperar", 1: "patrulla", 2: "moto", 3: "furgón"}
        for i, (accion, q_val) in enumerate(valores[:5]):
            bloqueo_id, tipo_v, oficiales = accion
            if bloqueo_id == -1:
                desc = "  Esperar (no hacer nada)"
            else:
                nombre_b = entorno.bloqueos[bloqueo_id]["nombre"]
                desc = f"  Enviar {nombres_v[tipo_v]} con {oficiales} oficial/es → {nombre_b}"
            marca = " ← ELEGIDA" if i == 0 else ""
            print(f"   {i+1}. Q={q_val:+.2f} | {desc}{marca}")