import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Discretización del espacio continuo ---
# MountainCar-v0 tiene observaciones continuas: [posicion, velocidad]
# Las dividimos en bins para poder usar una Q-table
NUM_BINS = 20  # bins por dimension
POS_BINS  = np.linspace(-1.2,  0.6,  NUM_BINS)
VEL_BINS  = np.linspace(-0.07, 0.07, NUM_BINS)

def discretizar(obs):
    """Convierte observacion continua en indice discreto (tupla)."""
    pos_idx = np.digitize(obs[0], POS_BINS) - 1
    vel_idx = np.digitize(obs[1], VEL_BINS) - 1
    return (
        np.clip(pos_idx, 0, NUM_BINS - 1),
        np.clip(vel_idx, 0, NUM_BINS - 1)
    )

def train(episodes):
    env = gym.make('MountainCar-v0')

    # Q-table: (bins_posicion x bins_velocidad x acciones)
    q_table = np.zeros((NUM_BINS, NUM_BINS, env.action_space.n))

    learning_rate       = 0.1
    discount_factor     = 0.95
    epsilon             = 1.0
    epsilon_decay_rate  = 0.0001
    rng = np.random.default_rng()

    # Array para almacenar si el agente llego a la meta en cada episodio
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):

        # Cada 1000 episodios muestra una ventana visual
        if (i + 1) % 1000 == 0:
            env.close()
            env = gym.make('MountainCar-v0', render_mode='human')
        else:
            env.close()
            env = gym.make('MountainCar-v0')

        obs, _ = env.reset()
        state = discretizar(obs)

        terminated = False
        truncated  = False

        while not terminated and not truncated:
            # Explorar o explotar
            if rng.random() < epsilon:
                action = env.action_space.sample()               # exploracion
            else:
                action = np.argmax(q_table[state[0], state[1], :])  # explotacion

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_state = discretizar(new_obs)

            # Actualizacion Q-Learning
            q_table[state[0], state[1], action] = (
                q_table[state[0], state[1], action]
                + learning_rate * (
                    reward
                    + discount_factor * np.max(q_table[new_state[0], new_state[1], :])
                    - q_table[state[0], state[1], action]
                )
            )

            state = new_state

        # Reducir epsilon con el tiempo
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        # Registrar exito: el auto llego a la meta (terminated=True, no truncated)
        if terminated and not truncated:
            rewards_per_episode[i] = 1

        if (i + 1) % 1000 == 0:
            print(f'Episodio {i + 1} | Exito: {int(rewards_per_episode[i])} | Epsilon: {epsilon:.4f}')

    env.close()

    print('\nTabla Q final (max por estado):')
    print(np.max(q_table, axis=2))  # valor maximo por cada estado discretizado

    # Suma de exitos en ventana de 100 episodios
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(sum_rewards)
    plt.xlabel('Episodio')
    plt.ylabel('Exitos en ultimos 100 episodios')
    plt.title('MountainCar-v0 con Q-Learning')
    plt.show()

if __name__ == '__main__':
    train(20000)