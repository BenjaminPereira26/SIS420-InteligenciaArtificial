import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sin ventana (compatible con VS Code)

from env.bloqueo_env import BloqueoEntorno
from agent.q_learning import AgenteQLearning



EPISODIOS_ENTRENAMIENTO = 50000  # cuántos episodios entrenar
EPISODIOS_EVALUACION    = 10000    # cuántos episodios usar para evaluar al final
INTERVALO_LOG           = 5000    # cada cuántos episodios mostrar progreso
GUARDAR_Q_TABLE         = True   # guardar la Q-Table al finalizar


def entrenar():
    print("=" * 60)
    print("  INICIANDO ENTRENAMIENTO - BLOQUEOS RL")
    print("=" * 60)
    print(f"  Episodios: {EPISODIOS_ENTRENAMIENTO}")
    print(f"  El agente explorará y aprenderá por sí solo.\n")

    # Crear entorno y agente
    entorno = BloqueoEntorno(semilla=42)
    agente  = AgenteQLearning(
        alpha=0.1,           # tasa de aprendizaje
        gamma=0.95,          # descuento de recompensas futuras
        epsilon=1.0,         # empieza explorando 100%
        epsilon_min=0.05,    # nunca baja de 5% de exploración
        epsilon_decay=0.995, # decae suavemente
    )

    # Métricas de entrenamiento
    recompensas        = []
    bloqueos_resueltos = []
    epsilons           = []
    ventana            = 100  # tamaño de ventana para promedio móvil


    for ep in range(1, EPISODIOS_ENTRENAMIENTO + 1):

        # Un episodio completo de entrenamiento
        recompensa_ep = agente.entrenar_episodio(entorno)
        recompensas.append(recompensa_ep)
        bloqueos_resueltos.append(entorno.bloqueos_resueltos)
        epsilons.append(agente.epsilon)

        # Mostrar progreso cada INTERVALO_LOG episodios
        if ep % INTERVALO_LOG == 0:
            r_promedio  = np.mean(recompensas[-ventana:])
            b_promedio  = np.mean(bloqueos_resueltos[-ventana:])
            estados_q   = len(agente.q_table)
            print(f"  Ep {ep:5d}/{EPISODIOS_ENTRENAMIENTO} | "
                  f"Recomp. (últ.{ventana}): {r_promedio:8.1f} | "
                  f"Bloqueos resueltos: {b_promedio:.2f}/6 | "
                  f"ε={agente.epsilon:.3f} | "
                  f"Estados Q: {estados_q}")

    print("Entrenamiento completado.")
    agente.resumen()


    # GUARDAR Q-TABLE
    if GUARDAR_Q_TABLE:
        agente.guardar("q_table.json")


    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN FINAL ({EPISODIOS_EVALUACION} episodios sin exploración)")
    print(f"{'='*60}")

    epsilon_original = agente.epsilon
    agente.epsilon = 0.0  # desactivar exploración para evaluar

    recompensas_eval  = []
    bloqueos_eval     = []

    for _ in range(EPISODIOS_EVALUACION):
        recompensa_ep = agente.entrenar_episodio(entorno)
        recompensas_eval.append(recompensa_ep)
        bloqueos_eval.append(entorno.bloqueos_resueltos)

    agente.epsilon = epsilon_original

    print(f"  Recompensa promedio:    {np.mean(recompensas_eval):.1f}")
    print(f"  Bloqueos resueltos/ep:  {np.mean(bloqueos_eval):.2f} / 6")
    print(f"  Mejor episodio:         {max(recompensas_eval):.1f}")
    print(f"  Peor episodio:          {min(recompensas_eval):.1f}")


    graficar_aprendizaje(recompensas, bloqueos_resueltos, epsilons, ventana)

    return agente, entorno


def graficar_aprendizaje(recompensas, bloqueos_resueltos, epsilons, ventana=100):
    """
    Genera gráficas que muestran cómo aprendió el agente a lo largo del tiempo.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("Curvas de Aprendizaje – Agente Q-Learning\nGestión de Bloqueos de Carreteras",
                 fontsize=14, fontweight='bold')

    episodios = range(1, len(recompensas) + 1)

    # --- Gráfica 1: Recompensa por episodio ---
    ax1 = axes[0]
    ax1.plot(episodios, recompensas, alpha=0.3, color='steelblue', linewidth=0.5,
             label='Recompensa por episodio')
    # Promedio móvil para ver la tendencia
    if len(recompensas) >= ventana:
        promedio = [np.mean(recompensas[max(0, i-ventana):i+1])
                    for i in range(len(recompensas))]
        ax1.plot(episodios, promedio, color='navy', linewidth=2,
                 label=f'Promedio móvil ({ventana} ep.)')
    ax1.set_ylabel("Recompensa total")
    ax1.set_title("Recompensa por episodio (sube = el agente aprende mejor)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Gráfica 2: Bloqueos resueltos por episodio ---
    ax2 = axes[1]
    ax2.plot(episodios, bloqueos_resueltos, alpha=0.3, color='green', linewidth=0.5)
    if len(bloqueos_resueltos) >= ventana:
        promedio_b = [np.mean(bloqueos_resueltos[max(0, i-ventana):i+1])
                      for i in range(len(bloqueos_resueltos))]
        ax2.plot(episodios, promedio_b, color='darkgreen', linewidth=2,
                 label=f'Promedio móvil ({ventana} ep.)')
    ax2.axhline(y=6, color='red', linestyle='--', alpha=0.5, label='Máximo (6 bloqueos)')
    ax2.set_ylabel("Bloqueos resueltos")
    ax2.set_title("Bloqueos resueltos por episodio (máximo = 6)")
    ax2.set_ylim(0, 7)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --- Gráfica 3: Evolución de Epsilon ---
    ax3 = axes[2]
    ax3.plot(episodios, epsilons, color='orange', linewidth=1.5)
    ax3.fill_between(episodios, epsilons, alpha=0.2, color='orange')
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_title("Epsilon: exploración → explotación (baja con el tiempo)")
    ax3.set_ylim(0, 1.05)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    ruta = "curvas_aprendizaje.png"
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"Gráficas guardadas en '{ruta}'")
    plt.close()


def demo_episodio(agente, entorno):
    """
    Corre un episodio de demostración mostrando cada turno.
    """
    print("\n" + "="*60)
    print("  DEMOSTRACIÓN: AGENTE EXPERTO EN ACCIÓN")
    print("="*60)
    print("  (Epsilon = 0, el agente usa solo lo que aprendió)\n")

    agente.epsilon = 0.0  # sin exploración
    estado = entorno.reset()
    estado_vec = entorno.estado_a_vector(estado)
    entorno.render()

    terminado = False
    turno = 0

    while not terminado and turno < 10:  # mostramos los primeros 10 turnos
        acciones_validas = entorno.obtener_acciones_validas()

        # Mostrar qué piensa el agente
        agente.explicar_decision(estado_vec, acciones_validas, entorno)

        # El agente elige la mejor acción
        accion = agente.elegir_accion_greedy(estado_vec, acciones_validas)

        # Ejecutar
        estado, recompensa, terminado, info = entorno.step(accion)
        estado_vec = entorno.estado_a_vector(estado)

        print(f"\n  ↪ Recompensa de este turno: {recompensa:.1f}")
        entorno.render()
        turno += 1

        input("\n  Presiona ENTER para el siguiente turno...")



if __name__ == "__main__":
    agente, entorno = entrenar()

    # Preguntar si quiere ver la demo
    respuesta = input("\n¿Deseas ver una demostración del agente entrenado? (s/n): ")
    if respuesta.lower() == 's':
        demo_episodio(agente, entorno)