import matplotlib
# Intentar backends en orden de compatibilidad con Windows
for _backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Posición fija del recinto en el mapa (centro-izquierda)
POS_RECINTO = (0.15, 0.50)

# Posiciones fijas de los 6 bloqueos distribuidos en el mapa
POSICIONES_BLOQUEOS = [
    (0.55, 0.82),   # 0 - Norte       (arriba)
    (0.55, 0.55),   # 1 - Sur         (centro-arriba)
    (0.75, 0.72),   # 2 - Este        (derecha-arriba)
    (0.72, 0.35),   # 3 - Oeste       (derecha-abajo)
    (0.88, 0.60),   # 4 - Noreste     (extremo derecha)
    (0.88, 0.28),   # 5 - Sureste     (extremo derecha-abajo)
]

# Colores según nivel de tensión
COLOR_TENSION = {
    0: "#2ecc71",   # verde  - tensión baja
    1: "#f39c12",   # naranja - tensión media
    2: "#e74c3c",   # rojo   - tensión alta
}
LABEL_TENSION = {0: "BAJA", 1: "MEDIA", 2: "ALTA"}

# Iconos/etiquetas por tipo de vehículo
ICONO_VEHICULO = {
    1: "🚓",   # patrulla
    2: "🏍",   # moto
    3: "🚐",   # furgón
}


class MapaSimulacion:
    """
    Ventana gráfica que muestra el estado del entorno en tiempo real.

    Cómo funciona:
      - Al crear el mapa se abre una ventana de Matplotlib
      - Cada vez que el agente actúa, llamas a mapa.actualizar(entorno)
      - El mapa redibuja todo: bloqueos, vehículos en tránsito, recursos
    """

    def __init__(self, entorno, pausa=0.8):
        """
        entorno: instancia de BloqueoEntorno (ya inicializada)
        pausa:   segundos entre turnos (0.8 = se ve bien, 0.3 = rápido)
        """
        self.pausa = pausa
        plt.ion()   # modo interactivo: no bloquea la ejecución

        # Crear figura con dos zonas: mapa grande + panel lateral
        self.fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("Sistema de Gestión de Bloqueos - RL")

        # Área del mapa (80% del ancho)
        self.ax_mapa = self.fig.add_axes([0.01, 0.05, 0.62, 0.90])
        # Panel lateral de recursos (20% del ancho)
        self.ax_panel = self.fig.add_axes([0.65, 0.05, 0.33, 0.90])

        self._configurar_ejes()
        self._dibujar_fondo_mapa()

        # Primer dibujo
        self.actualizar(entorno)

    # -------------------------------------------------------------------------
    # CONFIGURACIÓN INICIAL
    # -------------------------------------------------------------------------

    def _configurar_ejes(self):
        """Fondo oscuro, sin ejes visibles — estilo dashboard."""
        for ax in [self.ax_mapa, self.ax_panel]:
            ax.set_facecolor("#16213e")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

    def _dibujar_fondo_mapa(self):
        """Dibuja elementos estáticos del mapa (solo una vez)."""
        # Título del mapa
        self.ax_mapa.text(0.50, 0.97,
                          "MAPA DE OPERACIONES — RECINTO POLICIAL",
                          ha='center', va='top',
                          color='white', fontsize=11, fontweight='bold',
                          transform=self.ax_mapa.transAxes)

        # Líneas de cuadrícula tenues (dan sensación de mapa)
        for x in np.arange(0.1, 1.0, 0.15):
            self.ax_mapa.axvline(x, color='#ffffff08', linewidth=0.5)
        for y in np.arange(0.1, 1.0, 0.15):
            self.ax_mapa.axhline(y, color='#ffffff08', linewidth=0.5)

    # -------------------------------------------------------------------------
    # ACTUALIZACIÓN PRINCIPAL (llamar en cada turno)
    # -------------------------------------------------------------------------

    def actualizar(self, entorno):
        """
        Redibuja el mapa completo con el estado actual del entorno.
        Llama a esta función después de cada entorno.step(accion).
        """
        self.ax_mapa.cla()
        self.ax_panel.cla()
        self._configurar_ejes()
        self._dibujar_fondo_mapa()

        self._dibujar_recinto(entorno)
        self._dibujar_bloqueos(entorno)
        self._dibujar_vehiculos_transito(entorno)
        self._dibujar_panel_recursos(entorno)
        self._dibujar_turno(entorno)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.pausa)

    # -------------------------------------------------------------------------
    # DIBUJAR RECINTO
    # -------------------------------------------------------------------------

    def _dibujar_recinto(self, entorno):
        """Dibuja el ícono del recinto policial en el mapa."""
        rx, ry = POS_RECINTO

        # Círculo de fondo azul
        circulo = plt.Circle((rx, ry), 0.07,
                              color="#0066cc", zorder=5, alpha=0.9)
        self.ax_mapa.add_patch(circulo)

        # Ícono y nombre
        self.ax_mapa.text(rx, ry + 0.01, "🏛", ha='center', va='center',
                          fontsize=22, zorder=10)
        self.ax_mapa.text(rx, ry - 0.085, "RECINTO\nPOLICIAL",
                          ha='center', va='top', color='#7ecfff',
                          fontsize=7.5, fontweight='bold', zorder=10)

        # Oficiales disponibles
        r = entorno.recursos
        self.ax_mapa.text(rx, ry - 0.145,
                          f"👮 {r['oficiales_disponibles']} disponibles",
                          ha='center', va='top', color='white',
                          fontsize=7, zorder=10)

    # -------------------------------------------------------------------------
    # DIBUJAR BLOQUEOS
    # -------------------------------------------------------------------------

    def _dibujar_bloqueos(self, entorno):
        """Dibuja cada bloqueo con su estado actual."""
        for i, bloqueo in enumerate(entorno.bloqueos):
            bx, by = POSICIONES_BLOQUEOS[i]

            if bloqueo["resuelto"]:
                self._dibujar_bloqueo_resuelto(bx, by, bloqueo)
            else:
                self._dibujar_bloqueo_activo(bx, by, bloqueo, i)

    def _dibujar_bloqueo_activo(self, bx, by, bloqueo, idx):
        """Bloqueo activo: coloreado según tensión, con datos."""
        tension = bloqueo["tension"]
        color   = COLOR_TENSION[tension]

        # Fondo del bloqueo (cuadrado redondeado)
        rect = mpatches.FancyBboxPatch(
            (bx - 0.09, by - 0.06), 0.18, 0.12,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor='white',
            linewidth=1.5, alpha=0.85, zorder=5
        )
        self.ax_mapa.add_patch(rect)

        # Nombre del bloqueo (primera parte, corto)
        nombre_corto = bloqueo["nombre"].split("(")[0].strip()
        self.ax_mapa.text(bx, by + 0.035, nombre_corto,
                          ha='center', va='center',
                          color='white', fontsize=6.5, fontweight='bold', zorder=10)

        # Tensión
        self.ax_mapa.text(bx, by + 0.005,
                          f"Tensión: {LABEL_TENSION[tension]}",
                          ha='center', va='center',
                          color='white', fontsize=6, zorder=10)

        # Manifestantes y policías presentes
        self.ax_mapa.text(bx, by - 0.022,
                          f"👥 {bloqueo['manifestantes']}  👮 {bloqueo['oficiales_asignados']}",
                          ha='center', va='center',
                          color='white', fontsize=6, zorder=10)

        # Distancia
        self.ax_mapa.text(bx, by - 0.048,
                          f"📍 {bloqueo['distancia_km']} km",
                          ha='center', va='center',
                          color='#cccccc', fontsize=5.5, zorder=10)

        # Línea al recinto (tenue)
        rx, ry = POS_RECINTO
        self.ax_mapa.plot([rx, bx], [ry, by],
                          color='#ffffff18', linewidth=0.8,
                          linestyle='--', zorder=2)

    def _dibujar_bloqueo_resuelto(self, bx, by, bloqueo):
        """Bloqueo resuelto: gris con checkmark."""
        rect = mpatches.FancyBboxPatch(
            (bx - 0.09, by - 0.06), 0.18, 0.12,
            boxstyle="round,pad=0.01",
            facecolor="#2c2c2c", edgecolor="#555555",
            linewidth=1, alpha=0.7, zorder=5
        )
        self.ax_mapa.add_patch(rect)

        self.ax_mapa.text(bx, by + 0.015, "✅ RESUELTO",
                          ha='center', va='center',
                          color='#55ff55', fontsize=7, fontweight='bold', zorder=10)

        nombre_corto = bloqueo["nombre"].split("(")[0].strip()
        self.ax_mapa.text(bx, by - 0.025, nombre_corto,
                          ha='center', va='center',
                          color='#888888', fontsize=6, zorder=10)

    # -------------------------------------------------------------------------
    # DIBUJAR VEHÍCULOS EN TRÁNSITO
    # -------------------------------------------------------------------------

    def _dibujar_vehiculos_transito(self, entorno):
        """
        Dibuja los vehículos yendo al bloqueo (amarillo →)
        y los que regresan al recinto (verde ←).
        """
        # --- Vehículos yendo al bloqueo ---
        destinos = {}
        for v in entorno.vehiculos_en_transito:
            destinos.setdefault(v["bloqueo_destino"], []).append(v)

        for bloqueo_id, vehiculos in destinos.items():
            rx, ry = POS_RECINTO
            bx, by = POSICIONES_BLOQUEOS[bloqueo_id]
            dist_km   = entorno.bloqueos[bloqueo_id]["distancia_km"]
            turnos_max = max(1, int(np.ceil(dist_km / 20.0)))

            # Flecha de ruta (una sola por destino)
            self.ax_mapa.annotate(
                "", xy=(bx, by), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="-|>", color='#ffdd5780', lw=1.2),
                zorder=3
            )

            for j, v in enumerate(vehiculos):
                fraccion = max(0.15, min(0.90, 1.0 - v["turnos_restantes"] / turnos_max))
                vx = rx + fraccion * (bx - rx)
                vy = ry + fraccion * (by - ry) + j * 0.03

                icono = ICONO_VEHICULO.get(v["tipo"], "🚗")
                self.ax_mapa.text(vx, vy, icono, ha='center', va='center',
                                  fontsize=13, zorder=15)
                self.ax_mapa.text(vx, vy - 0.05,
                                  f"{v['oficiales']}👮 {v['turnos_restantes']}t",
                                  ha='center', va='top', color='#ffdd57',
                                  fontsize=5.5,
                                  bbox=dict(boxstyle='round,pad=0.2',
                                            facecolor='#00000099', edgecolor='none'),
                                  zorder=15)

        # --- Vehículos regresando al recinto ---
        for j, v in enumerate(entorno.vehiculos_regresando):
            rx, ry = POS_RECINTO
            bx, by = POSICIONES_BLOQUEOS[v["desde_bloqueo"]]
            dist_km   = entorno.bloqueos[v["desde_bloqueo"]]["distancia_km"]
            turnos_max = max(1, int(np.ceil(dist_km / 20.0)))

            # El regreso va de bloqueo → recinto, fracción invertida
            fraccion = max(0.10, min(0.85, v["turnos_restantes"] / turnos_max))
            vx = rx + fraccion * (bx - rx)
            vy = ry + fraccion * (by - ry) - j * 0.03  # offset hacia abajo

            # Flecha de retorno (punteada, verde)
            self.ax_mapa.annotate(
                "", xy=(rx, ry), xytext=(bx, by),
                arrowprops=dict(arrowstyle="-|>", color='#55ff5580', lw=1.0,
                                linestyle='dashed'),
                zorder=3
            )

            icono = ICONO_VEHICULO.get(v["tipo"], "🚗")
            self.ax_mapa.text(vx, vy, icono, ha='center', va='center',
                              fontsize=12, zorder=15, alpha=0.7)
            self.ax_mapa.text(vx, vy - 0.05,
                              f"↩ {v['oficiales']}👮 {v['turnos_restantes']}t",
                              ha='center', va='top', color='#55ff55',
                              fontsize=5.5,
                              bbox=dict(boxstyle='round,pad=0.2',
                                        facecolor='#00000099', edgecolor='none'),
                              zorder=15)

    # -------------------------------------------------------------------------
    # PANEL LATERAL DE RECURSOS
    # -------------------------------------------------------------------------

    def _dibujar_panel_recursos(self, entorno):
        """Panel derecho con recursos disponibles y estado de bloqueos."""
        ax = self.ax_panel
        r  = entorno.recursos

        # Título
        ax.text(0.5, 0.97, "RECURSOS DEL RECINTO",
                ha='center', va='top', color='white',
                fontsize=9, fontweight='bold',
                transform=ax.transAxes)

        # --- Recursos numéricos ---
        y = 0.88
        recursos_items = [
            ("👮 Oficiales disp.",  f"{r['oficiales_disponibles']} / 70"),
            ("🚓 Patrullas",        f"{r['patrullas_disponibles']} / 6"),
            ("🏍  Motos",           f"{r['motos_disponibles']} / 8"),
            ("🚐 Furgón",           f"{r['furgones_disponibles']} / 1"),
        ]
        for label, valor in recursos_items:
            ax.text(0.05, y, label, ha='left', va='center',
                    color='#aaaaaa', fontsize=8, transform=ax.transAxes)
            ax.text(0.95, y, valor, ha='right', va='center',
                    color='white', fontsize=8, fontweight='bold',
                    transform=ax.transAxes)
            y -= 0.07

        # Separador
        ax.plot([0.05, 0.95], [y + 0.02, y + 0.02], color='#444444',
                linewidth=0.8, transform=ax.transAxes)
        y -= 0.04

        # --- Estado de bloqueos ---
        ax.text(0.5, y, "ESTADO DE BLOQUEOS",
                ha='center', va='center', color='white',
                fontsize=8.5, fontweight='bold', transform=ax.transAxes)
        y -= 0.06

        for i, b in enumerate(entorno.bloqueos):
            nombre_corto = b["nombre"].split("(")[0].strip().replace("Bloqueo ", "")

            if b["resuelto"]:
                color_t = "#55ff55"
                estado  = "✅"
                detalle = "Resuelto"
            else:
                color_t = COLOR_TENSION[b["tension"]]
                estado  = ["🟢", "🟡", "🔴"][b["tension"]]
                detalle = f"{b['oficiales_asignados']}👮 asignados"

            ax.text(0.05, y, f"{estado} {nombre_corto}",
                    ha='left', va='center', color=color_t,
                    fontsize=7, transform=ax.transAxes)
            ax.text(0.95, y, detalle,
                    ha='right', va='center', color='#cccccc',
                    fontsize=6.5, transform=ax.transAxes)
            y -= 0.06

        # Separador
        ax.plot([0.05, 0.95], [y + 0.01, y + 0.01], color='#444444',
                linewidth=0.8, transform=ax.transAxes)
        y -= 0.04

        # --- Vehículos yendo al bloqueo ---
        nombres_v = {1: "Patrulla", 2: "Moto", 3: "Furgón"}
        if entorno.vehiculos_en_transito:
            ax.text(0.5, y, "→ YENDO AL BLOQUEO",
                    ha='center', va='center', color='#ffdd57',
                    fontsize=7.5, fontweight='bold', transform=ax.transAxes)
            y -= 0.055
            for v in entorno.vehiculos_en_transito:
                icono    = ICONO_VEHICULO.get(v["tipo"], "🚗")
                nombre_b = entorno.bloqueos[v["bloqueo_destino"]]["nombre"]
                nombre_b = nombre_b.split("(")[0].strip().replace("Bloqueo ", "")
                ax.text(0.05, y,
                        f"{icono} {nombres_v[v['tipo']]} → {nombre_b}",
                        ha='left', va='center', color='#ffdd57',
                        fontsize=6.5, transform=ax.transAxes)
                ax.text(0.95, y,
                        f"{v['oficiales']}👮 {v['turnos_restantes']}t",
                        ha='right', va='center', color='#aaaaaa',
                        fontsize=6.5, transform=ax.transAxes)
                y -= 0.052

        # --- Vehículos regresando al recinto ---
        if entorno.vehiculos_regresando:
            ax.text(0.5, y, "← REGRESANDO",
                    ha='center', va='center', color='#55ff55',
                    fontsize=7.5, fontweight='bold', transform=ax.transAxes)
            y -= 0.055
            for v in entorno.vehiculos_regresando:
                icono    = ICONO_VEHICULO.get(v["tipo"], "🚗")
                nombre_b = entorno.bloqueos[v["desde_bloqueo"]]["nombre"]
                nombre_b = nombre_b.split("(")[0].strip().replace("Bloqueo ", "")
                ax.text(0.05, y,
                        f"{icono} {nombres_v[v['tipo']]} ← {nombre_b}",
                        ha='left', va='center', color='#55ff55',
                        fontsize=6.5, transform=ax.transAxes)
                ax.text(0.95, y,
                        f"{v['oficiales']}👮 {v['turnos_restantes']}t",
                        ha='right', va='center', color='#aaaaaa',
                        fontsize=6.5, transform=ax.transAxes)
                y -= 0.052

        # --- Métricas finales ---
        ax.plot([0.05, 0.95], [0.12, 0.12], color='#444444',
                linewidth=0.8, transform=ax.transAxes)
        ax.text(0.05, 0.08, f"Recompensa total:",
                ha='left', va='center', color='#aaaaaa',
                fontsize=7.5, transform=ax.transAxes)
        ax.text(0.95, 0.08, f"{entorno.recompensa_total:.0f} pts",
                ha='right', va='center', color='#ffd700',
                fontsize=8, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.03, f"Resueltos:",
                ha='left', va='center', color='#aaaaaa',
                fontsize=7.5, transform=ax.transAxes)
        ax.text(0.95, 0.03, f"{entorno.bloqueos_resueltos} / 6",
                ha='right', va='center', color='#55ff55',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

    # -------------------------------------------------------------------------
    # TURNO ACTUAL
    # -------------------------------------------------------------------------

    def _dibujar_turno(self, entorno):
        """Muestra el turno actual en la parte inferior del mapa."""
        minutos = entorno.turno_actual * 30
        horas   = minutos // 60
        mins    = minutos % 60
        self.ax_mapa.text(0.5, 0.02,
                          f"Turno {entorno.turno_actual} / 48  —  "
                          f"Tiempo transcurrido: {horas:02d}h {mins:02d}min",
                          ha='center', va='bottom',
                          color='#888888', fontsize=8,
                          transform=self.ax_mapa.transAxes)

    def cerrar(self):
        """Cierra la ventana del mapa."""
        plt.ioff()
        plt.close(self.fig)