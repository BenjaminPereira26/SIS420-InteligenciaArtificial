import numpy as np
import random



# Flota de vehículos disponibles
NUM_PATRULLAS   = 6    
NUM_MOTOS       = 8    
NUM_FURGONES    = 1    

# Capacidad máxima de cada tipo de vehículo (en efectivos)
CAP_PATRULLA  = 4     
CAP_MOTO      = 1     
CAP_FURGON    = 12    

# Personal del recinto
OFICIALES_TOTAL = 70  

# Número de bloqueos activos en la simulación
NUM_BLOQUEOS = 6

# Tiempo máximo de un episodio (en turnos de 30 minutos)
MAX_TURNOS = 48       # equivale a 24 horas de simulación


SIN_VEHICULO = 0
PATRULLA     = 1
MOTO         = 2
FURGON       = 3

NOMBRES_VEHICULO = {
    SIN_VEHICULO: "Sin vehículo (esperar)",
    PATRULLA:     "Patrulla",
    MOTO:         "Moto",
    FURGON:       "Furgón",
}


TENSION_BAJA   = 0   # bloqueo pacífico, se puede esperar
TENSION_MEDIA  = 1   # bloqueo con riesgo de escalar
TENSION_ALTA   = 2   # bloqueo crítico, necesita intervención inmediata


class BloqueoEntorno:

    def __init__(self, semilla=None):

        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

        self.bloqueos_config = self._generar_bloqueos()

        self.estado = None
        self.turno_actual = 0

        self.historial = []


    def _generar_bloqueos(self):

        bloqueos = [
            {
                "nombre": "Bloqueo Norte (Av. Principal)",
                "distancia_km": 2.0,    # muy cerca → llega en 1 turno
                "manifestantes": 30,
                "tension_inicial": TENSION_MEDIA,
            },
            {
                "nombre": "Bloqueo Sur (Ruta 1)",
                "distancia_km": 5.5,    # cerca → llega en 1 turno
                "manifestantes": 80,
                "tension_inicial": TENSION_ALTA,
            },
            {
                "nombre": "Bloqueo Este (Carretera Interurbana)",
                "distancia_km": 12.0,   # moderado → llega en 2 turnos
                "manifestantes": 50,
                "tension_inicial": TENSION_MEDIA,
            },
            {
                "nombre": "Bloqueo Oeste (Camino Industrial)",
                "distancia_km": 18.0,   # lejos → llega en 3 turnos
                "manifestantes": 120,
                "tension_inicial": TENSION_ALTA,
            },
            {
                "nombre": "Bloqueo Noreste (Zona Rural)",
                "distancia_km": 25.0,   # muy lejos → llega en 4 turnos
                "manifestantes": 40,
                "tension_inicial": TENSION_BAJA,
            },
            {
                "nombre": "Bloqueo Sureste (Paso Fronterizo)",
                "distancia_km": 30.0,   # muy lejos → llega en 5 turnos
                "manifestantes": 200,
                "tension_inicial": TENSION_ALTA,
            },
        ]
        return bloqueos

    def _calcular_turnos_viaje(self, distancia_km):
        """
        Convierte distancia en turnos de viaje (cada turno = 30 min).
        Velocidad promedio en ciudad: 40 km/h → 20 km por turno.
        Mínimo 1 turno (siempre tarda algo en llegar).
        """
        turnos = max(1, int(np.ceil(distancia_km / 20.0)))
        return turnos



    def reset(self):

        self.turno_actual = 0
        self.historial = []

        # Cada bloqueo es un diccionario con su situación actual
        self.bloqueos = []
        for cfg in self.bloqueos_config:
            bloqueo = {
                "nombre":        cfg["nombre"],
                "distancia_km":  cfg["distancia_km"],
                "manifestantes": cfg["manifestantes"],
                "tension":       cfg["tension_inicial"],   # nivel 0/1/2
                "resuelto":      False,                    # ¿ya se resolvió?
                "oficiales_asignados": 0,                  # policías presentes
                "turnos_sin_atencion": 0,                  # tiempo ignorado
                "turnos_para_resolver": 0,                 # countdown al resolver
                "vehiculos_presentes":  [],                # vehículos físicamente en el bloqueo
            }
            self.bloqueos.append(bloqueo)

        # --- Recursos del recinto ---
        self.recursos = {
            "oficiales_disponibles": OFICIALES_TOTAL,
            "patrullas_disponibles": NUM_PATRULLAS,
            "motos_disponibles":     NUM_MOTOS,
            "furgones_disponibles":  NUM_FURGONES,
        }

        self.vehiculos_en_transito = []

        self.vehiculos_regresando = []


        self.recompensa_total = 0
        self.bloqueos_resueltos = 0

        return self._obtener_estado()


    def _discretizar_oficiales(self, n):
        """
        Convierte la cantidad exacta de oficiales en un nivel semántico.
        Los umbrales coinciden con las capacidades reales de los vehículos,
        así el agente distingue situaciones que realmente importan.

          0 → sin efectivos
          1 → solo alcanza para una patrulla (1 a 4)
          2 → patrullas posibles, no furgón lleno (5 a 12)
          3 → alcanza para un furgón lleno (13 a 25)
          4 → recursos moderados (26 a 50)
          5 → recursos abundantes (51+)
        """
        if n == 0:   return 0
        if n <= 4:   return 1
        if n <= 12:  return 2
        if n <= 25:  return 3
        if n <= 50:  return 4
        return 5

    def _obtener_estado(self):

        estado = {
            # Información de cada bloqueo (6 bloqueos × 4 variables)
            "bloqueos": [
                {
                    "tension":             b["tension"],
                    "manifestantes_nivel": min(2, b["manifestantes"] // 60),  # 0/1/2
                    "resuelto":            int(b["resuelto"]),
                    "oficiales_presentes": self._discretizar_oficiales(b["oficiales_asignados"]),
                }
                for b in self.bloqueos
            ],
            # Recursos disponibles en el recinto (discretizados)
            "patrullas":  self.recursos["patrullas_disponibles"],
            "motos":      self.recursos["motos_disponibles"],
            "furgones":   self.recursos["furgones_disponibles"],
            "oficiales":  self._discretizar_oficiales(self.recursos["oficiales_disponibles"]),
        }
        return estado

    def estado_a_vector(self, estado=None):
        """
        Convierte el estado (diccionario) a una tupla numérica.
        Las tuplas se usan como claves de la Q-Table (diccionario de Python).

        Ejemplo de tupla: (2, 1, 0, 1, 1, 0, 0, 0, ..., 6, 8, 1, 7)
        """
        if estado is None:
            estado = self._obtener_estado()

        vector = []
        for b in estado["bloqueos"]:
            vector.append(b["tension"])              # 0, 1 o 2
            vector.append(b["manifestantes_nivel"])  # 0, 1 o 2
            vector.append(b["resuelto"])             # 0 o 1
            vector.append(b["oficiales_presentes"])  # 0–5 (semántico)

        vector.append(estado["patrullas"])
        vector.append(estado["motos"])
        vector.append(estado["furgones"])
        vector.append(estado["oficiales"])    # 0–5 (semántico)

        return tuple(vector)



    def obtener_acciones_validas(self):
        """
        Retorna la lista de acciones que el agente puede tomar AHORA.
        Solo incluye acciones posibles con los recursos actuales.

        La acción "esperar" siempre está disponible como salida de emergencia.
        """
        acciones = []

        # Acción especial: esperar sin hacer nada
        acciones.append((-1, SIN_VEHICULO, 0))

        # Para cada bloqueo activo (no resuelto)
        for i, bloqueo in enumerate(self.bloqueos):
            if bloqueo["resuelto"]:
                continue  # ya resuelto, no necesita más recursos

            r = self.recursos  

            # --- Opción: enviar patrulla ---
            if r["patrullas_disponibles"] >= 1:
                max_of = min(CAP_PATRULLA, r["oficiales_disponibles"])
                for n in range(1, max_of + 1):
                    acciones.append((i, PATRULLA, n))

            # --- Opción: enviar moto ---
            if r["motos_disponibles"] >= 1 and r["oficiales_disponibles"] >= 1:
                acciones.append((i, MOTO, 1))

            # --- Opción: enviar furgón ---
            if r["furgones_disponibles"] >= 1:
                max_of = min(CAP_FURGON, r["oficiales_disponibles"])
                for n in range(1, max_of + 1):
                    acciones.append((i, FURGON, n))

        return acciones

    def step(self, accion):
        """
        Retorna:
          estado_siguiente: nuevo estado del entorno
          recompensa:       puntos ganados/perdidos por esta acción
          terminado:        True si el episodio acabó
          info:             diccionario con detalles (para debugging)
        """
        bloqueo_id, tipo_vehiculo, num_oficiales = accion

        recompensa = 0
        info = {"accion_tomada": accion, "turno": self.turno_actual}

        # --- FASE 1: Ejecutar la acción del agente ---
        if bloqueo_id >= 0 and tipo_vehiculo != SIN_VEHICULO:
            recompensa += self._despachar_vehiculo(bloqueo_id, tipo_vehiculo, num_oficiales, info)

        # --- FASE 2: Avanzar los vehículos en tránsito ---
        self._avanzar_transito()

        # --- FASE 3: Evolucionar el estado de cada bloqueo ---
        recompensa += self._evolucionar_bloqueos()

        # --- FASE 4: Avanzar el tiempo ---
        self.turno_actual += 1

        # --- FASE 5: Verificar si el episodio terminó ---
        terminado = self._verificar_fin()

        # Guardar en historial para análisis
        estado_sig = self._obtener_estado()
        self.recompensa_total += recompensa
        self.historial.append({
            "turno":      self.turno_actual,
            "accion":     accion,
            "recompensa": recompensa,
        })

        return estado_sig, recompensa, terminado, info


    def _despachar_vehiculo(self, bloqueo_id, tipo_vehiculo, num_oficiales, info):
        """
        Envía un vehículo con oficiales hacia el bloqueo indicado.
        El vehículo viajará varios turnos antes de llegar.
        Retorna la recompensa inmediata por esta decisión.
        """
        recompensa = 0
        r = self.recursos
        bloqueo = self.bloqueos[bloqueo_id]

        # Verificar capacidad del vehículo elegido
        capacidad = {PATRULLA: CAP_PATRULLA, MOTO: CAP_MOTO, FURGON: CAP_FURGON}[tipo_vehiculo]
        num_oficiales = min(num_oficiales, capacidad, r["oficiales_disponibles"])

        if num_oficiales <= 0:
            return -2  # penalización por acción inválida

        # Descontar recursos del recinto
        r["oficiales_disponibles"] -= num_oficiales
        if tipo_vehiculo == PATRULLA:
            r["patrullas_disponibles"] -= 1
        elif tipo_vehiculo == MOTO:
            r["motos_disponibles"] -= 1
        elif tipo_vehiculo == FURGON:
            r["furgones_disponibles"] -= 1

        # Calcular tiempo de viaje
        turnos_viaje = self._calcular_turnos_viaje(bloqueo["distancia_km"])

        # Registrar el vehículo en tránsito
        self.vehiculos_en_transito.append({
            "tipo":             tipo_vehiculo,
            "oficiales":        num_oficiales,
            "bloqueo_destino":  bloqueo_id,
            "turnos_restantes": turnos_viaje,
        })

        # Pequeña penalización por costo de operación
        costo = {PATRULLA: 1, MOTO: 0.5, FURGON: 2}[tipo_vehiculo]
        recompensa -= costo

        info["vehiculo_despachado"] = NOMBRES_VEHICULO[tipo_vehiculo]
        info["oficiales_enviados"]  = num_oficiales
        info["turnos_de_viaje"]     = turnos_viaje

        return recompensa

    def _avanzar_transito(self):
        """
        Descuenta un turno de viaje a cada vehículo.
        Si un vehículo llega (turnos_restantes == 0), lo aplica al bloqueo.
        """
        llegaron = []
        aun_viajando = []

        for vehiculo in self.vehiculos_en_transito:
            vehiculo["turnos_restantes"] -= 1

            if vehiculo["turnos_restantes"] <= 0:
                # El vehículo llegó al bloqueo
                llegaron.append(vehiculo)
            else:
                aun_viajando.append(vehiculo)

        # Actualizar la lista de vehículos en tránsito
        self.vehiculos_en_transito = aun_viajando

        # Aplicar los refuerzos que llegaron al bloqueo
        for vehiculo in llegaron:
            self._aplicar_refuerzo(vehiculo)

        # --- Avanzar vehículos que regresan al recinto ---
        aun_regresando = []
        for v in self.vehiculos_regresando:
            v["turnos_restantes"] -= 1
            if v["turnos_restantes"] <= 0:
                # Llegó al recinto: devolver vehículo y oficiales
                self.recursos["oficiales_disponibles"] += v["oficiales"]
                if v["tipo"] == PATRULLA:
                    self.recursos["patrullas_disponibles"] += 1
                elif v["tipo"] == MOTO:
                    self.recursos["motos_disponibles"] += 1
                elif v["tipo"] == FURGON:
                    self.recursos["furgones_disponibles"] += 1
            else:
                aun_regresando.append(v)
        self.vehiculos_regresando = aun_regresando

    def _aplicar_refuerzo(self, vehiculo):
        """
        Cuando un vehículo llega al bloqueo, suma los oficiales
        y calcula si el bloqueo puede resolverse.
        """
        bloqueo = self.bloqueos[vehiculo["bloqueo_destino"]]

        if bloqueo["resuelto"]:
            # El bloqueo ya se resolvió antes de que llegara → recursos regresan
            self._regresar_recursos(vehiculo)
            return

        # Sumar policías al bloqueo y registrar el vehículo presente
        bloqueo["oficiales_asignados"] += vehiculo["oficiales"]

        # Guardamos el vehículo que llegó físicamente (para devolverlo luego)
        bloqueo["vehiculos_presentes"].append({
            "tipo":      vehiculo["tipo"],
            "oficiales": vehiculo["oficiales"],
        })

        # ¿Los oficiales presentes son suficientes para resolver?
        # 1 oficial por cada 10 manifestantes, mínimo 3
        oficiales_necesarios = max(3, bloqueo["manifestantes"] // 10)
        if bloqueo["oficiales_asignados"] >= oficiales_necesarios:
            # Tiempo para resolver depende de la tensión
            turnos_resolucion = {
                TENSION_BAJA:  1,
                TENSION_MEDIA: 2,
                TENSION_ALTA:  3,
            }[bloqueo["tension"]]
            bloqueo["turnos_para_resolver"] = turnos_resolucion


    def _evolucionar_bloqueos(self):
        """
        Avanza el estado de cada bloqueo en un turno.
        - Si tiene suficientes oficiales y cuenta regresiva llegó a 0 → resuelto
        - Si lleva mucho tiempo sin atención → la tensión escala
        Retorna la recompensa/penalización acumulada de todos los bloqueos.
        """
        recompensa = 0

        for i, bloqueo in enumerate(self.bloqueos):
            if bloqueo["resuelto"]:
                continue  # ya resuelto, no hay nada que hacer

            # ¿Está en proceso de resolución? (policías presentes negociando)
            if bloqueo["turnos_para_resolver"] > 0:
                bloqueo["turnos_para_resolver"] -= 1

                if bloqueo["turnos_para_resolver"] == 0:
                    recompensa += self._resolver_bloqueo(i)
                    continue

                # Mientras los policías negocian, la tensión NO escala
                # y la penalización es menor (la situación está controlada)
                recompensa += -1   # penalización mínima: bloqueo aún activo

            else:
                # ¿Hay vehículos ya en camino hacia este bloqueo?
                vehiculo_en_camino = any(
                    v["bloqueo_destino"] == i
                    for v in self.vehiculos_en_transito
                )

                if vehiculo_en_camino:
                    # Hay refuerzo en camino: penalización leve, tensión no escala
                    recompensa += -2
                else:
                    # Nadie atendiendo y nadie en camino → escala
                    bloqueo["turnos_sin_atencion"] += 1

                    # Cada 3 turnos sin atención, la tensión sube un nivel
                    if bloqueo["turnos_sin_atencion"] % 3 == 0:
                        if bloqueo["tension"] < TENSION_ALTA:
                            bloqueo["tension"] += 1  # tensión escala

                    # Penalización completa según tensión
                    penalizacion_por_tension = {
                        TENSION_BAJA:  -1,
                        TENSION_MEDIA: -3,
                        TENSION_ALTA:  -8,
                    }[bloqueo["tension"]]
                    recompensa += penalizacion_por_tension

        return recompensa

    def _resolver_bloqueo(self, bloqueo_id):
        """
        Marca un bloqueo como resuelto y calcula la recompensa.
        Recompensa mayor si se resolvió rápido y con tensión alta
        """
        bloqueo = self.bloqueos[bloqueo_id]
        bloqueo["resuelto"] = True
        self.bloqueos_resueltos += 1

        # Recompensa base: más alta si se resolvió pronto
        # turno_actual + 1 para evitar división por cero
        rapidez = MAX_TURNOS / (self.turno_actual + 1)
        recompensa_base = 100 * rapidez

        # Bonus por resolver bloqueos difíciles
        bonus_tension = {
            TENSION_BAJA:  0,
            TENSION_MEDIA: 20,
            TENSION_ALTA:  50,
        }[bloqueo["tension"]]

        recompensa_total = recompensa_base + bonus_tension

        # Encolar viaje de regreso (los recursos tardan lo mismo en volver)
        self._encolar_regreso(bloqueo_id)

        return recompensa_total

    def _regresar_recursos(self, vehiculo):
        """
        Devuelve un vehículo que iba a un bloqueo ya resuelto.
        El bloqueo estaba resuelto antes de que llegara, así que
        el vehículo da media vuelta de inmediato.
        """
        self.recursos["oficiales_disponibles"] += vehiculo["oficiales"]
        if vehiculo["tipo"] == PATRULLA:
            self.recursos["patrullas_disponibles"] += 1
        elif vehiculo["tipo"] == MOTO:
            self.recursos["motos_disponibles"] += 1
        elif vehiculo["tipo"] == FURGON:
            self.recursos["furgones_disponibles"] += 1

    def _encolar_regreso(self, bloqueo_id):
        """
        Al resolver un bloqueo, todos los vehículos y oficiales presentes
        inician el viaje de regreso al recinto.

        Los recursos NO se liberan de inmediato — el agente debe esperar
        a que regresen antes de poder usarlos de nuevo. 
        """
        bloqueo = self.bloqueos[bloqueo_id]
        turnos_regreso = self._calcular_turnos_viaje(bloqueo["distancia_km"])

        # Encolar cada vehículo presente en el bloqueo
        for v in bloqueo.get("vehiculos_presentes", []):
            self.vehiculos_regresando.append({
                "tipo":             v["tipo"],
                "oficiales":        v["oficiales"],
                "desde_bloqueo":    bloqueo_id,   # solo para mostrar en render()
                "turnos_restantes": turnos_regreso,
            })

        # Limpiar el registro del bloqueo
        bloqueo["oficiales_asignados"] = 0
        bloqueo["vehiculos_presentes"] = []


    def _verificar_fin(self):
        """
        El episodio termina si:
          1. Se resolvieron todos los bloqueos (victoria total)
          2. Se agotó el tiempo máximo (derrota parcial)
        """
        todos_resueltos = all(b["resuelto"] for b in self.bloqueos)
        tiempo_agotado = self.turno_actual >= MAX_TURNOS
        return todos_resueltos or tiempo_agotado


    def render(self):

        print(f"\n{'='*60}")
        print(f"  TURNO {self.turno_actual} / {MAX_TURNOS}  "
              f"({self.turno_actual * 30} min transcurridos)")
        print(f"{'='*60}")

        print("BLOQUEOS ACTIVOS:")
        niveles_tension = {0: "🟢 BAJA", 1: "🟡 MEDIA", 2: "🔴 ALTA"}
        for i, b in enumerate(self.bloqueos):
            estado_str = "✅ RESUELTO" if b["resuelto"] else niveles_tension[b["tension"]]
            print(f"  [{i}] {b['nombre']}")
            print(f"       Estado: {estado_str} | "
                  f"Manifestantes: {b['manifestantes']} | "
                  f"Distancia: {b['distancia_km']} km | "
                  f"Policías presentes: {b['oficiales_asignados']}")

        print("RECURSOS EN RECINTO:")
        r = self.recursos
        print(f"Oficiales: {r['oficiales_disponibles']}/{OFICIALES_TOTAL} | "
              f"Patrullas: {r['patrullas_disponibles']}/{NUM_PATRULLAS} | "
              f"Motos: {r['motos_disponibles']}/{NUM_MOTOS} | "
              f"Furgones: {r['furgones_disponibles']}/{NUM_FURGONES}")

        if self.vehiculos_en_transito:
            print("YENDO AL BLOQUEO:")
            for v in self.vehiculos_en_transito:
                nombre_b = self.bloqueos[v["bloqueo_destino"]]["nombre"]
                print(f"  {NOMBRES_VEHICULO[v['tipo']]} con {v['oficiales']} policías "
                      f"→ {nombre_b} "
                      f"(llega en {v['turnos_restantes']} turno/s)")

        if self.vehiculos_regresando:
            print("REGRESANDO AL RECINTO:")
            for v in self.vehiculos_regresando:
                nombre_b = self.bloqueos[v["desde_bloqueo"]]["nombre"]
                print(f"  {NOMBRES_VEHICULO[v['tipo']]} con {v['oficiales']} policías "
                      f"← {nombre_b} "
                      f"(llega en {v['turnos_restantes']} turno/s)")

        print(f"Recompensa acumulada: {self.recompensa_total:.1f}")
        print(f"Bloqueos resueltos: {self.bloqueos_resueltos}/{NUM_BLOQUEOS}")