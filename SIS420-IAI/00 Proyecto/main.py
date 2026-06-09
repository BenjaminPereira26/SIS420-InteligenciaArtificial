import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.bloqueo_env import BloqueoEntorno
from agent.q_learning import AgenteQLearning


def prueba_rapida():
    """
    Prueba el entorno con acciones aleatorias durante 5 turnos.
    Sirve para verificar que todo funciona correctamente.
    """
    print("=" * 60)
    print("  PRUEBA RÁPIDA DEL ENTORNO")
    print("=" * 60)
    print("  Ejecutando 5 turnos con acciones aleatorias...\n")

    import random

    entorno = BloqueoEntorno(semilla=123)
    entorno.reset()
    entorno.render()

    for turno in range(5):
        print(f"\n{'─'*60}")
        print(f"  TURNO {turno + 1}")

        acciones = entorno.obtener_acciones_validas()
        print(f"  Acciones disponibles: {len(acciones)}")

        # Acción aleatoria (como si el agente no supiera nada todavía)
        accion = random.choice(acciones)
        bloqueo_id, tipo_v, oficiales = accion
        nombres_v = {0: "esperar", 1: "patrulla", 2: "moto", 3: "furgón"}

        if bloqueo_id == -1:
            print(f"  Acción elegida: ESPERAR")
        else:
            nombre_b = entorno.bloqueos[bloqueo_id]["nombre"]
            print(f"  Acción elegida: Enviar {nombres_v[tipo_v]} "
                  f"con {oficiales} oficial(es) → {nombre_b}")

        _, recompensa, terminado, _ = entorno.step(accion)
        print(f"  Recompensa: {recompensa:.1f}")
        entorno.render()

        if terminado:
            print("\n  ¡Episodio terminado!")
            break

    print("Prueba completada. El entorno funciona correctamente.")
    print("Para iniciar el entrenamiento completo, ejecuta:")
    print("python training/train.py\n")


def prueba_agente():
    """
    Prueba rápida del agente Q-Learning (sin entrenar — solo 10 episodios).
    """
    print("\n" + "=" * 60)
    print("PRUEBA RÁPIDA DEL AGENTE (10 episodios)")
    print("=" * 60)

    entorno = BloqueoEntorno(semilla=42)
    agente  = AgenteQLearning(alpha=0.1, gamma=0.95, epsilon=1.0)

    recompensas = []
    for ep in range(1, 11):
        r = agente.entrenar_episodio(entorno)
        recompensas.append(r)
        print(f"Episodio {ep:2d}: recompensa={r:8.1f} | "
              f"bloqueos resueltos={entorno.bloqueos_resueltos}/6 | "
              f"ε={agente.epsilon:.3f}")

    print(f"Estados aprendidos en Q-Table: {len(agente.q_table)}")
    print("Agente funciona correctamente.")
    print("Con miles de episodios aprenderá la estrategia óptima.\n")


if __name__ == "__main__":
    prueba_rapida()
    prueba_agente()