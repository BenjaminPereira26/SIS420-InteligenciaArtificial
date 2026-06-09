import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.bloqueo_env import BloqueoEntorno
from agent.q_learning import AgenteQLearning
from visualization.mapa import MapaSimulacion


def demo_con_mapa(episodios_entrenamiento=20000, pausa_visual=0.1):
    """
    1. Entrena el agente rápidamente (sin visual para ir más rápido)
    2. Luego corre un episodio de demostración CON el mapa animado
    """

    # -----------------------------------------------------------------------
    # FASE 1: Entrenamiento rápido (sin visualización)
    # -----------------------------------------------------------------------
    print("=" * 55)
    print("  FASE 1: Entrenando agente...")
    print(f"  ({episodios_entrenamiento} episodios, sin visual para mayor velocidad)")
    print("=" * 55)

    entorno = BloqueoEntorno(semilla=42)
    agente  = AgenteQLearning(
        alpha=0.1, gamma=0.95,
        epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995
    )

    # Intentar cargar Q-Table existente para no repetir entrenamiento
    if os.path.exists("q_table.json"):
        agente.cargar("q_table.json")
        print("  Q-Table encontrada. Saltando entrenamiento.\n")
    else:
        for ep in range(1, episodios_entrenamiento + 1):
            agente.entrenar_episodio(entorno)
            if ep % 500 == 0:
                import numpy as np
                r_prom = np.mean(agente.recompensas_por_episodio[-100:])
                print(f"  Ep {ep:5d} | Recompensa prom.: {r_prom:8.1f} | "
                      f"ε={agente.epsilon:.3f} | "
                      f"Estados Q: {len(agente.q_table)}")
        agente.guardar("q_table.json")

    print("Entrenamiento completado.")
    agente.resumen()

    # -----------------------------------------------------------------------
    # FASE 2: Demo visual (agente experto)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  FASE 2: Abriendo mapa visual...")
    print("  El agente usará lo que aprendió (sin exploración)")
    print("=" * 55 + "\n")

    agente.epsilon = 0.0   # sin exploración → solo decisiones aprendidas
    entorno.reset()

    # Crear el mapa (abre la ventana)
    mapa = MapaSimulacion(entorno, pausa=pausa_visual)

    terminado = False
    while not terminado:
        estado_vec     = entorno.estado_a_vector()
        acciones_valid = entorno.obtener_acciones_validas()

        # El agente elige la mejor acción conocida
        accion = agente.elegir_accion_greedy(estado_vec, acciones_valid)

        # Mostrar en consola qué decidió
        _imprimir_decision(accion, entorno)

        # Ejecutar la acción
        _, recompensa, terminado, _ = entorno.step(accion)

        # Actualizar el mapa
        mapa.actualizar(entorno)

        print(f"  → Recompensa este turno: {recompensa:+.1f} | "
              f"Total: {entorno.recompensa_total:.0f} pts")

    # Resultado final
    print("\n" + "=" * 55)
    if entorno.bloqueos_resueltos == 6:
        print(" ¡TODOS LOS BLOQUEOS RESUELTOS!")
    else:
        print(f"Tiempo agotado. Resueltos: {entorno.bloqueos_resueltos}/6")
    print(f"Puntuación final: {entorno.recompensa_total:.0f} pts")
    print("=" * 55)

    input("\nPresiona ENTER para cerrar el mapa...")
    mapa.cerrar()


def _imprimir_decision(accion, entorno):
    """Imprime en consola la decisión del agente de forma legible."""
    bloqueo_id, tipo_v, oficiales = accion
    nombres_v = {0: "—", 1: "Patrulla", 2: "Moto", 3: "Furgón"}

    print(f"\n[Turno {entorno.turno_actual}] Decisión del agente:")
    if bloqueo_id == -1:
        print("  → ESPERAR (conservar recursos)")
    else:
        nombre_b = entorno.bloqueos[bloqueo_id]["nombre"]
        print(f"  → Enviar {nombres_v[tipo_v]} "
              f"con {oficiales} oficial(es) a: {nombre_b}")


# -----------------------------------------------------------------------------
# PUNTO DE ENTRADA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo_con_mapa(
        episodios_entrenamiento=20000,   # reduce si quieres que abra más rápido
        pausa_visual=1.0                # segundos entre turnos (1.0 = cómodo de ver)
    )